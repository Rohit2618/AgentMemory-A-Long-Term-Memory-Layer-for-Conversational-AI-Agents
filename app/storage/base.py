"""Storage abstraction shared by consolidation and retrieval.

The consolidation engine (Task 5) and retrieval (Task 7) depend only on the
``MemoryStore`` protocol defined here — never on Neo4j or Chroma directly. The
concrete graph + vector implementation is built in Task 6 and injected in.

A ``MemoryStore`` unifies two backends behind one interface:
  * a **graph** of (entity)-[attribute]->(value) facts, for exact entity/
    attribute lookups and relationship traversal, and
  * a **vector index** of fact embeddings, for fuzzy semantic search.

Keeping this as a Protocol means tests can supply a trivial in-memory fake, and
the engine logic can be validated without any external services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.core.models import Fact


@dataclass
class ScoredFact:
    """A fact returned from a similarity search, with its similarity score."""

    fact: Fact
    score: float


@runtime_checkable
class MemoryStore(Protocol):
    """Unified graph + vector persistence for facts."""

    # ---- writes ---- #
    def add_fact(self, fact: Fact) -> None:
        """Insert a new fact into both the graph and the vector index."""
        ...

    def update_fact(self, fact_id: str, new_fact: Fact) -> None:
        """Replace the value/embedding of an existing fact (same fact_id)."""
        ...

    def delete_fact(self, fact_id: str) -> None:
        """Remove a fact from both backends."""
        ...

    # ---- reads ---- #
    def get_by_key(self, entity: str, attribute: str) -> list[Fact]:
        """Exact graph lookup: all facts for an (entity, attribute) pair."""
        ...

    def get_entity_facts(self, entity: str) -> list[Fact]:
        """All facts about a given entity (graph traversal starting point)."""
        ...

    def search_similar(self, query: str, *, top_k: int = 5) -> list[ScoredFact]:
        """Vector similarity search over fact text. Highest score first."""
        ...

    def all_facts(self) -> list[Fact]:
        """Return every stored fact (used by eval + debugging)."""
        ...

    def clear(self) -> None:
        """Wipe all facts (used by eval to reset between runs)."""
        ...
