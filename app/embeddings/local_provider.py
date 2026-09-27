"""Local embedding provider using sentence-transformers.

Runs fully offline after the model is downloaded once (~90MB for MiniLM). No API
key, no per-call cost. This is the default so the demo is free to run.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """Wraps a sentence-transformers model (default: all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Lazy import: keeps the package importable without torch installed
        # (e.g. when EMBEDDING_PROVIDER=mock).
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        logger.info("Loading local embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        self._model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,  # cosine similarity == dot product
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def dimension(self) -> int:
        return self._dim

    def name(self) -> str:
        return f"local:{self._model_name}"
