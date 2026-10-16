"""Embedding Model Engine.

Provides an extensible embedder interface supporting:
1. FastEmbed (ONNX Runtime CPU - ~40MB, lightweight, no PyTorch/CUDA)
2. SentenceTransformers (BAAI/bge-small-en-v1.5)
3. MockEmbedder (for offline deterministic unit testing)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional

from app.config import settings
from app.logging_config import logger, timed_operation


class BaseEmbedder(ABC):
    """Abstract interface for dense vector embedding generation."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or HuggingFace ID of the embedding model."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the generated vector embeddings."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for a batch of text chunks."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate a dense embedding vector for a single search query."""
        pass


class MockEmbedder(BaseEmbedder):
    """Deterministic mock embedder for offline unit testing without weights download."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> List[float]:
        """Produce a pseudo-semantic deterministic unit-normalized vector from text."""
        # Seed pseudo vector using character frequencies
        vec = [0.0] * self._dimension
        if not text:
            vec[0] = 1.0
            return vec

        for idx, char in enumerate(text):
            slot = (ord(char) * 31 + idx * 17) % self._dimension
            vec[slot] += 1.0

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0
        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._generate_vector(query)


class FastEmbedEmbedder(BaseEmbedder):
    """Lightweight CPU embedder using FastEmbed (ONNX Runtime, ~40MB, no PyTorch)."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        from fastembed import TextEmbedding  # type: ignore

        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._dimension = settings.EMBEDDING_DIMENSION
        logger.info(f"Loading FastEmbed model '{self._model_name}' on CPU...")
        self._model = TextEmbedding(model_name=self._model_name)
        logger.info(f"FastEmbed model '{self._model_name}' initialized successfully.")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        with timed_operation("fastembed_batch", {"batch_size": len(texts)}):
            embeddings_gen = self._model.embed(texts)
            return [emb.tolist() for emb in embeddings_gen]

    def embed_query(self, query: str) -> List[float]:
        with timed_operation("fastembed_query"):
            embeddings_gen = self._model.embed([query])
            return next(embeddings_gen).tolist()


class SentenceTransformerEmbedder(BaseEmbedder):
    """Standard SentenceTransformer embedder using PyTorch."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._dimension = settings.EMBEDDING_DIMENSION
        logger.info(f"Loading SentenceTransformer '{self._model_name}' on CPU...")
        self._model = SentenceTransformer(self._model_name, device="cpu")
        logger.info(f"SentenceTransformer '{self._model_name}' initialized successfully.")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        with timed_operation("st_embed_batch", {"batch_size": len(texts)}):
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        with timed_operation("st_embed_query"):
            emb = self._model.encode(
                [query],
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return emb[0].tolist()


# Global cached embedder instance
_embedder_instance: Optional[BaseEmbedder] = None


def get_embedder(prefer_mock: bool = False) -> BaseEmbedder:
    """Retrieve or initialize the active embedding engine.

    Tries FastEmbed first (lightweight CPU), then SentenceTransformers,
    and falls back to MockEmbedder if libraries are not yet installed in host env.
    """
    global _embedder_instance
    if _embedder_instance is not None and not prefer_mock:
        return _embedder_instance

    if prefer_mock:
        return MockEmbedder()

    # Try FastEmbed
    try:
        import fastembed  # type: ignore # noqa: F401
        _embedder_instance = FastEmbedEmbedder()
        return _embedder_instance
    except ImportError:
        pass

    # Try SentenceTransformers
    try:
        import sentence_transformers  # type: ignore # noqa: F401
        _embedder_instance = SentenceTransformerEmbedder()
        return _embedder_instance
    except ImportError:
        pass

    logger.warning(
        "Neither FastEmbed nor SentenceTransformers is installed. Initializing MockEmbedder."
    )
    _embedder_instance = MockEmbedder()
    return _embedder_instance
