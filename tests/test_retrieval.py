"""Unit tests for hybrid retrieval: ranking, budget, floor, and fallbacks."""

from __future__ import annotations

import tempfile

from app.core.config import Settings
from app.core.models import Fact
from app.retrieval.retriever import HybridRetriever


def _seed(store, triples):
    for e, a, v in triples:
        store.add_fact(Fact(entity=e, attribute=a, value=v, confidence=0.9))


class TestRetrievalRanking:
    def test_returns_results_sorted_by_score_desc(self, store):
        _seed(store, [("user", "lives_in", "Bangalore"), ("user", "likes", "hiking")])
        r = HybridRetriever(store, Settings(retrieval_top_k=5, retrieval_min_score=0.0))
        res = r.retrieve("where does the user live")
        scores = [m.score for m in res]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_budget_is_enforced(self, store):
        _seed(
            store,
            [("user", f"attr_{i}", f"val_{i}") for i in range(10)],
        )
        r = HybridRetriever(store, Settings(retrieval_top_k=3, retrieval_min_score=0.0))
        assert len(r.retrieve("anything")) <= 3

    def test_min_score_floor_filters(self, store):
        _seed(store, [("user", "lives_in", "Bangalore")])
        r = HybridRetriever(store, Settings(retrieval_top_k=5, retrieval_min_score=0.99))
        # Almost nothing clears a 0.99 floor for an unrelated query.
        res = r.retrieve("totally unrelated zzzz query")
        assert all(m.score >= 0.99 for m in res)

    def test_empty_store_returns_nothing(self, store):
        r = HybridRetriever(store, Settings(retrieval_top_k=5, retrieval_min_score=0.0))
        assert r.retrieve("anything") == []

    def test_vector_only_fallback_for_inmemory_store(self, store):
        # InMemoryStore has no related_entities -> all sources are "vector".
        _seed(store, [("user", "lives_in", "Bangalore"), ("alice", "lives_in", "Bangalore")])
        r = HybridRetriever(store, Settings(retrieval_top_k=5, retrieval_min_score=0.0))
        res = r.retrieve("where user lives")
        assert res and all(m.source == "vector" for m in res)


class TestHybridGraphTraversal:
    def test_graph_traversal_pulls_related_entities(self, mock_embeddings):
        """With a composite store, retrieval reaches entities sharing a value."""
        from app.storage.graph_store import NetworkXGraphStore
        from app.storage.memory_store import GraphVectorMemoryStore
        from app.storage.vector_store import ChromaVectorStore

        with tempfile.TemporaryDirectory() as d:
            composite = GraphVectorMemoryStore(
                graph=NetworkXGraphStore(),
                vectors=ChromaVectorStore(embeddings=mock_embeddings, persist_dir=d),
            )
            _seed(
                composite,
                [
                    ("user", "lives_in", "Bangalore"),
                    ("alice", "lives_in", "Bangalore"),  # shares value -> related
                ],
            )
            r = HybridRetriever(composite, Settings(retrieval_top_k=10, retrieval_min_score=0.0))
            res = r.retrieve("where does the user live")
            entities = {m.fact.entity for m in res}
            assert "alice" in entities  # brought in via graph traversal
            assert any(m.source in ("graph", "hybrid") for m in res)
