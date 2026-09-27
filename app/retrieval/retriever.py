"""Hybrid retrieval: vector similarity + graph traversal.

Given a user query, we want the memories most likely to help answer it. A pure
vector search is good at fuzzy matching but misses facts that are *related by
entity* rather than by wording. So we combine two signals:

  1. **Vector search** — semantic similarity of the query to each fact's text.
  2. **Graph traversal** — starting from the entities surfaced by the vector
     hits, pull in facts about those entities and their graph neighbours. This
     brings in contextually linked memories a vector search alone would rank low.

Results are merged (a fact found by both signals keeps its best score, tagged
"hybrid"), filtered by a minimum-score floor, ranked, and capped at a top-k
budget so we never bloat the LLM context window.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.models import RetrievedMemory
from app.storage.base import MemoryStore

logger = get_logger(__name__)

# Graph-derived facts get a slightly damped score since they were pulled in by
# association rather than direct semantic match to the query.
_GRAPH_SCORE_WEIGHT = 0.7


class HybridRetriever:
    """Retrieves and ranks memories using vector + graph signals."""

    def __init__(self, store: MemoryStore, settings: Settings | None = None) -> None:
        self._store = store
        s = settings or get_settings()
        self._top_k = s.retrieval_top_k
        self._min_score = s.retrieval_min_score

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedMemory]:
        """Return up to ``top_k`` relevant memories, best first.

        Args:
            query: the user query to find memories for.
            top_k: optional override of the configured retrieval budget.
        """
        budget = top_k or self._top_k
        # Pull a wider candidate set than the budget so ranking has room to work.
        candidate_k = max(budget * 3, budget)

        # Best score seen per fact_id, plus which signal(s) found it.
        best: dict[str, RetrievedMemory] = {}

        # --- Signal 1: vector similarity ---------------------------------
        vector_hits = self._store.search_similar(query, top_k=candidate_k)
        for hit in vector_hits:
            fid = hit.fact.ensure_id().fact_id
            best[fid] = RetrievedMemory(fact=hit.fact, score=hit.score, source="vector")

        # --- Signal 2: graph traversal from surfaced entities ------------
        # Only if the store supports graph traversal (composite store does;
        # the vector-only InMemoryStore does not -> we simply skip it).
        related_fn = getattr(self._store, "related_entities", None)
        if callable(related_fn):
            seed_entities = {h.fact.entity for h in vector_hits}
            graph_entities: set[str] = set(seed_entities)
            for ent in seed_entities:
                try:
                    graph_entities.update(related_fn(ent, max_hops=1))
                except TypeError:
                    # Signature without max_hops kwarg.
                    graph_entities.update(related_fn(ent))

            for ent in graph_entities:
                for fact in self._store.get_entity_facts(ent):
                    fid = fact.ensure_id().fact_id
                    graph_score = _GRAPH_SCORE_WEIGHT * fact.confidence
                    if fid in best:
                        # Found by both signals: keep the higher score, mark hybrid.
                        existing = best[fid]
                        merged_score = max(existing.score, graph_score)
                        best[fid] = RetrievedMemory(
                            fact=fact, score=merged_score, source="hybrid"
                        )
                    else:
                        best[fid] = RetrievedMemory(
                            fact=fact, score=graph_score, source="graph"
                        )

        # --- Rank, filter by min-score floor, apply budget ---------------
        ranked = sorted(best.values(), key=lambda m: m.score, reverse=True)
        filtered = [m for m in ranked if m.score >= self._min_score]
        result = filtered[:budget]

        logger.info(
            "Retrieval: query=%r -> %d candidate(s), %d after floor(%.2f), returning %d",
            query,
            len(best),
            len(filtered),
            self._min_score,
            len(result),
        )
        return result
