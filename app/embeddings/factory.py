"""Embedding provider factory — maps config -> concrete embedding provider."""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)


def build_embeddings(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.lower().strip()

    if provider == "mock":
        from app.embeddings.mock_provider import MockEmbeddingProvider

        logger.info("Embedding provider: mock (deterministic, offline)")
        return MockEmbeddingProvider()

    if provider == "local":
        from app.embeddings.local_provider import LocalEmbeddingProvider

        return LocalEmbeddingProvider(settings.embedding_model)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={settings.embedding_provider!r}. "
        "Expected 'local' or 'mock'."
    )


@lru_cache
def get_embeddings() -> EmbeddingProvider:
    """Return a cached embedding provider built from active settings."""
    return build_embeddings(get_settings())
