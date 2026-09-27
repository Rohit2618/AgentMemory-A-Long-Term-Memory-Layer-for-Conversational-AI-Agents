"""Deterministic mock embedding provider (no model download, instant).

Produces stable pseudo-embeddings from a hash of token n-grams. It is NOT
semantically meaningful the way a real model is, but it is deterministic and
gives *higher* similarity to texts that share words — enough to exercise and
unit-test the vector-search + ranking code paths with zero dependencies.
"""

from __future__ import annotations

import hashlib
import math
import re

from app.embeddings.base import EmbeddingProvider

_DIM = 64


class MockEmbeddingProvider(EmbeddingProvider):
    """Hash-based bag-of-words embedding. Fast, deterministic, dependency-free."""

    def __init__(self, dim: int = _DIM) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = _tokenize(text)
        if not tokens:
            return vec
        # Hash each token into a bucket; accumulate. Shared words -> shared
        # buckets -> higher cosine similarity.
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            bucket = h % self._dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[bucket] += sign
        # L2-normalize so dot product == cosine similarity.
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def dimension(self) -> int:
        return self._dim

    def name(self) -> str:
        return "mock"


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]
