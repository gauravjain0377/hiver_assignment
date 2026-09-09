"""
Embedding Module — Phase 3
Local sentence-transformer embeddings, zero API cost.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


class EmbeddingModel:
    """Wraps sentence-transformers for local embedding generation."""

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None

    @classmethod
    def get_instance(cls) -> "EmbeddingModel":
        """Singleton pattern — load model once."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        """Lazy load the model (only when first used)."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.success(f"Model loaded. Embedding dim: {self._model.get_sentence_embedding_dimension()}")

    def embed(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of strings to embed
            batch_size: Batch size for encoding
            show_progress: Show tqdm progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Cosine similarity via dot product
            convert_to_numpy=True,
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single string."""
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        self._load()
        return self._model.get_sentence_embedding_dimension()


# Global instance
embedder = EmbeddingModel.get_instance()
