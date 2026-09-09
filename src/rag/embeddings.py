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
        """Lazy load the embedding model with ultra-low memory ONNX (FastEmbed) first."""
        if self._model is not None:
            return

        # Strategy 1: FastEmbed (ONNX Runtime, ~35MB RAM, ideal for Render 512MB free tier)
        try:
            logger.info("Loading ultra-lightweight ONNX embedder (fastembed)...")
            from fastembed import TextEmbedding
            # Map standard model names to FastEmbed supported models
            fe_name = "sentence-transformers/all-MiniLM-L6-v2"
            self._model = TextEmbedding(model_name=fe_name)
            self._mode = "fastembed"
            self._dim = 384
            logger.success("FastEmbed (ONNX) loaded successfully with ~35MB RAM usage.")
            return
        except Exception as e:
            logger.warning(f"FastEmbed not available or failed ({e}), falling back to sentence-transformers...")

        # Strategy 2: SentenceTransformers (PyTorch, ~450MB RAM)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._mode = "sentence_transformers"
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.success(f"SentenceTransformer loaded. Embedding dim: {self._dim}")
            return
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers: {e}")
            self._mode = "fallback"
            self._dim = 384

    def embed(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        Returns: numpy array of shape (len(texts), embedding_dim)
        """
        self._load()
        if self._mode == "fastembed":
            # fastembed returns generator of numpy arrays
            gen = self._model.embed(texts, batch_size=batch_size)
            arr = np.array(list(gen), dtype=np.float32)
            # Normalize vectors for cosine similarity
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms
        elif self._mode == "sentence_transformers":
            return self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        else:
            # Emergency zero-memory fallback
            return np.zeros((len(texts), self._dim), dtype=np.float32)

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
