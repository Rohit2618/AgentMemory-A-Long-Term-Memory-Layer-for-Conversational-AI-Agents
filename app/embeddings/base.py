"""Embedding provider interface.

Groq does not serve embeddings, so embeddings come from a separate provider.
Like the LLM layer, callers depend only on this abstract class; the concrete
model (local sentence-transformers, a hosted API, etc.) is chosen by config.
"""

from __future__ import annotations

import abc


class EmbeddingProvider(abc.ABC):
    """Turns text into fixed-length float vectors for similarity search."""

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        """Convenience: embed a single string."""
        return self.embed([text])[0]

    @abc.abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality (needed to size the vector index)."""
        raise NotImplementedError

    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError
