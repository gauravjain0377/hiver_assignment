"""
Vector Store — Phase 3
Manages the Qdrant vector database for RAG retrieval.
Supports both local ChromaDB (dev) and Qdrant Cloud (production).
"""
import sys
import json
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings
from src.rag.embeddings import embedder


class VectorStore:
    """
    Vector store abstraction.
    Uses ChromaDB locally, Qdrant Cloud in production.
    """

    def __init__(self):
        self.collection_name = settings.qdrant_collection_name
        self._client = None
        self._use_qdrant = bool(settings.qdrant_url)

    def _get_client(self):
        """Get the appropriate vector DB client."""
        if self._client is not None:
            return self._client

        if self._use_qdrant:
            logger.info(f"Connecting to Qdrant Cloud: {settings.qdrant_url}")
            from qdrant_client import QdrantClient
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            logger.info("Using local ChromaDB")
            import chromadb
            chroma_path = Path("data/chroma_db")
            chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(chroma_path))

        return self._client

    def build_index(self, conversations: list[dict], batch_size: int = 100):
        """
        Build the vector index from processed conversations.
        Embeds customer messages and stores with metadata.
        """
        client = self._get_client()
        texts = [c["customer_message"] for c in conversations]
        
        logger.info(f"Building vector index for {len(conversations):,} conversations...")
        logger.info("Computing embeddings (this runs locally, no API cost)...")

        if self._use_qdrant:
            self._build_qdrant(client, conversations, texts, batch_size)
        else:
            self._build_chroma(client, conversations, texts, batch_size)

        logger.success(f"Vector index built with {len(conversations):,} documents")

    def _build_qdrant(self, client, conversations, texts, batch_size):
        """Build Qdrant index."""
        from qdrant_client import models

        # Create collection if it doesn't exist
        existing = [c.name for c in client.get_collections().collections]
        if self.collection_name in existing:
            logger.info(f"Collection '{self.collection_name}' exists. Deleting and rebuilding.")
            client.delete_collection(self.collection_name)

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=embedder.dim,
                distance=models.Distance.COSINE,
            ),
        )

        # Index in batches
        for i in tqdm(range(0, len(conversations), batch_size), desc="Indexing batches"):
            batch_convs = conversations[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            embeddings = embedder.embed(batch_texts)

            points = [
                models.PointStruct(
                    id=i + j,
                    vector=embeddings[j].tolist(),
                    payload={
                        "customer_message": batch_convs[j]["customer_message"],
                        "brand_reply": batch_convs[j]["brand_reply"],
                        "brand": batch_convs[j].get("brand", settings.target_brand),
                        "conversation_id": batch_convs[j].get("conversation_id", ""),
                    },
                )
                for j in range(len(batch_convs))
            ]
            client.upsert(collection_name=self.collection_name, points=points)

    def _build_chroma(self, client, conversations, texts, batch_size):
        """Build ChromaDB index."""
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        for i in tqdm(range(0, len(conversations), batch_size), desc="Indexing batches"):
            batch_convs = conversations[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            embeddings = embedder.embed(batch_texts)

            collection.add(
                ids=[str(i + j) for j in range(len(batch_convs))],
                embeddings=embeddings.tolist(),
                documents=batch_texts,
                metadatas=[{
                    "customer_message": batch_convs[j]["customer_message"],
                    "brand_reply": batch_convs[j]["brand_reply"],
                    "brand": batch_convs[j].get("brand", settings.target_brand),
                    "conversation_id": batch_convs[j].get("conversation_id", ""),
                } for j in range(len(batch_convs))],
            )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Search for similar historical conversations.

        Args:
            query: Customer message to find similar cases for
            top_k: Number of results to return

        Returns:
            List of {customer_message, brand_reply, score} dicts
        """
        top_k = top_k or settings.rag_top_k
        client = self._get_client()
        query_embedding = embedder.embed_single(query)

        if self._use_qdrant:
            results = client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=top_k,
                with_payload=True,
            )
            hits = results.points
            return [
                {
                    "customer_message": hit.payload["customer_message"],
                    "brand_reply": hit.payload["brand_reply"],
                    "score": hit.score,
                }
                for hit in hits
            ]
        else:
            try:
                collection = client.get_collection(self.collection_name)
            except Exception:
                logger.warning("ChromaDB collection not found. Run: python src/rag/vector_store.py build")
                return []

            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            items = []
            for j in range(len(results["documents"][0])):
                meta = results["metadatas"][0][j]
                dist = results["distances"][0][j]
                items.append({
                    "customer_message": meta["customer_message"],
                    "brand_reply": meta["brand_reply"],
                    "score": 1 - dist,  # Convert distance to similarity
                })
            return items


# Global instance
vector_store = VectorStore()


if __name__ == "__main__":
    import typer
    app = typer.Typer()

    @app.command()
    def build(
        brand: str = typer.Option(settings.target_brand, "--brand"),
        sample: Optional[int] = typer.Option(None, "--sample", "-s"),
    ):
        """Build the vector index from processed conversations."""
        from src.data.preprocess import load_conversations
        convs = load_conversations(brand)
        if sample:
            import random
            random.seed(42)
            convs = random.sample(convs, min(sample, len(convs)))
        vector_store.build_index(convs)

    @app.command()
    def test(query: str = typer.Argument("I can't log into my account")):
        """Test the vector store with a sample query."""
        results = vector_store.search(query, top_k=3)
        for i, r in enumerate(results):
            print(f"\n--- Result {i+1} (score: {r['score']:.3f}) ---")
            print(f"Customer: {r['customer_message']}")
            print(f"Reply: {r['brand_reply']}")

    app()
