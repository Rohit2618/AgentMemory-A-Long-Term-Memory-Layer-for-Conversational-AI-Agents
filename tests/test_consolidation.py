"""Unit tests for the consolidation engine — the four decision paths."""

from __future__ import annotations

import pytest

from app.consolidation.engine import ConsolidationEngine
from app.core.models import ConsolidationAction, Fact


def fact(attr: str, value: str, *, entity: str = "user", conf: float = 0.9) -> Fact:
    return Fact(entity=entity, attribute=attr, value=value, confidence=conf)


@pytest.fixture
def engine(store):
    return ConsolidationEngine(store)


class TestConsolidationDecisions:
    def test_new_fact_is_added(self, engine):
        d = engine.consolidate(fact("lives_in", "Delhi"))
        assert d.action == ConsolidationAction.ADDED

    def test_exact_duplicate_is_rejected(self, engine):
        engine.consolidate(fact("lives_in", "Delhi"))
        d = engine.consolidate(fact("lives_in", "Delhi"))
        assert d.action == ConsolidationAction.REJECTED
        assert d.related_fact_id  # points at the existing fact

    def test_changed_value_updates_in_place(self, engine, store):
        engine.consolidate(fact("lives_in", "Delhi"))
        d = engine.consolidate(fact("lives_in", "Bangalore"))
        assert d.action == ConsolidationAction.UPDATED
        # Crucially: only ONE lives_in fact remains, with the new value.
        lives = store.get_by_key("user", "lives_in")
        assert len(lives) == 1
        assert lives[0].value == "Bangalore"

    def test_higher_confidence_duplicate_refreshes(self, engine):
        engine.consolidate(fact("works_as", "engineer", conf=0.7))
        d = engine.consolidate(fact("works_as", "engineer", conf=1.0))
        assert d.action == ConsolidationAction.UPDATED

    def test_semantic_duplicate_is_merged(self, store):
        # Lower the merge threshold so the mock embeddings can trigger a merge.
        engine = ConsolidationEngine(store, merge_threshold=0.5)
        engine.consolidate(fact("likes", "hiking mountains"))
        d = engine.consolidate(fact("enjoys", "hiking mountains"))
        assert d.action == ConsolidationAction.MERGED
        assert d.similarity is not None and d.similarity >= 0.5

    def test_different_entities_not_merged(self, store):
        engine = ConsolidationEngine(store, merge_threshold=0.1)  # very permissive
        engine.consolidate(fact("lives_in", "Bangalore", entity="user"))
        d = engine.consolidate(fact("lives_in", "Bangalore", entity="alice"))
        # Same value/attr but DIFFERENT entity -> must be a separate ADD.
        assert d.action == ConsolidationAction.ADDED

    def test_no_duplicates_accumulate_over_moves(self, engine, store):
        for city in ["Delhi", "Mumbai", "Bangalore", "Bengaluru"]:
            engine.consolidate(fact("lives_in", city))
        # After a chain of updates, still exactly one lives_in slot.
        assert len(store.get_by_key("user", "lives_in")) == 1
