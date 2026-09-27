"""In-memory MemoryStore implementation.

A dependency-free ``MemoryStore`` backed by plain dicts + the embedding
provider. It is useful in three places:

  * unit tests for the consolidation engine and retrieval ranking,
  * the "naive vector-only" baseline in the eval harness (with consolidation
    disabled), and
  * local development when you don't want to spin up Neo4j/Chroma.

The real graph (Neo4j/NetworkX) + Chroma implementation lives in Task 6, but it
honours the exact same ``MemoryStore`` interface, so nothing downstream cares
which one is injected.
"""

from __future__ import annotations

from app.core.models import Fact
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embeddings
from app.storage.base import MemoryStore, ScoredFact


class InMemoryStore(MemoryStore):
    """Dict-backed graph + brute-force vector search."""

    def __init__(self, embeddings: EmbeddingProvider | None = None) -> None:
        self._embeddings = embeddings or get_embeddings()
        self._facts: dict[str, Fact] = {}          # fact_id -> Fact
        self._vectors: dict[str, list[float]] = {}  # fact_id -> embedding

    # ---- writes ---- #
    def add_fact(self, fact: Fact) -> None:
        fact = fact.ensure_id()
        self._facts[fact.fact_id] = fact
        self._vectors[fact.fact_id] = self._embeddings.embed_one(fact.text())

    def update_fact(self, fact_id: str, new_fact: Fact) -> None:
        # Preserve the original id so references stay stable.
        new_fact = new_fact.model_copy(update={"fact_id": fact_id})
        self._facts[fact_id] = new_fact
        self._vectors[fact_id] = self._embeddings.embed_one(new_fact.text())

    def delete_fact(self, fact_id: str) -> None:
        self._facts.pop(fact_id, None)
        self._vectors.pop(fact_id, None)

    # ---- reads ---- #
    def get_by_key(self, entity: str, attribute: str) -> list[Fact]:
        e, a = entity.strip().lower(), attribute.strip().lower()
        return [
            f
            for f in self._facts.values()
            if f.entity.strip().lower() == e and f.attribute.strip().lower() == a
        ]

    def get_entity_facts(self, entity: str) -> list[Fact]:
        e = entity.strip().lower()
        return [f for f in self._facts.values() if f.entity.strip().lower() == e]

    def search_similar(self, query: str, *, top_k: int = 5) -> list[ScoredFact]:
        if not self._vectors:
            return []
        qv = self._embeddings.embed_one(query)
        scored = [
            ScoredFact(fact=self._facts[fid], score=_dot(qv, vec))
            for fid, vec in self._vectors.items()
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def all_facts(self) -> list[Fact]:
        return list(self._facts.values())

    def clear(self) -> None:
        self._facts.clear()
        self._vectors.clear()


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product; equals cosine similarity because vectors are normalized."""
    return sum(x * y for x, y in zip(a, b))
