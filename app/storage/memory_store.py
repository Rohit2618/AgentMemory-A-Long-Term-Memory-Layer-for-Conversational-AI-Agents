"""Composite MemoryStore: graph + vector behind one interface.

This is the production store. It writes every fact to BOTH the graph (for exact
lookups + traversal) and the vector index (for semantic search), and satisfies
the ``MemoryStore`` protocol the consolidation engine and retrieval depend on.
Keeping the two backends in sync lives here, in one place.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.core.models import Fact
from app.storage.base import MemoryStore, ScoredFact
from app.storage.graph_store import GraphStore
from app.storage.vector_store import ChromaVectorStore

logger = get_logger(__name__)


class GraphVectorMemoryStore(MemoryStore):
    """Fact store combining a GraphStore and a ChromaVectorStore."""

    def __init__(self, graph: GraphStore, vectors: ChromaVectorStore) -> None:
        self._graph = graph
        self._vectors = vectors

    # ---- writes (keep both backends in sync) ---- #
    def add_fact(self, fact: Fact) -> None:
        fact = fact.ensure_id()
        self._graph.upsert_fact(fact)
        self._vectors.upsert(fact)

    def update_fact(self, fact_id: str, new_fact: Fact) -> None:
        # Preserve the id so graph/vector references stay aligned.
        new_fact = new_fact.model_copy(update={"fact_id": fact_id})
        self._graph.delete_fact(fact_id)
        self._graph.upsert_fact(new_fact)
        self._vectors.upsert(new_fact)

    def delete_fact(self, fact_id: str) -> None:
        self._graph.delete_fact(fact_id)
        self._vectors.delete(fact_id)

    # ---- reads ---- #
    def get_by_key(self, entity: str, attribute: str) -> list[Fact]:
        return self._graph.get_by_key(entity, attribute)

    def get_entity_facts(self, entity: str) -> list[Fact]:
        return self._graph.get_entity_facts(entity)

    def search_similar(self, query: str, *, top_k: int = 5) -> list[ScoredFact]:
        return self._vectors.search(query, top_k=top_k)

    def related_entities(self, entity: str, *, max_hops: int = 1) -> list[str]:
        """Graph-only helper used by hybrid retrieval (not part of the protocol)."""
        return self._graph.related_entities(entity, max_hops=max_hops)

    def all_facts(self) -> list[Fact]:
        return self._graph.all_facts()

    def clear(self) -> None:
        self._graph.clear()
        self._vectors.clear()
