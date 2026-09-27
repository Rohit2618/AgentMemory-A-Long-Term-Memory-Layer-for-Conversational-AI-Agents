"""Unit tests for fact extraction: the parser and the mock extractor."""

from __future__ import annotations

from app.core.models import ConversationTurn
from app.extraction.extractor import FactExtractor
from app.extraction.parser import parse_facts


class TestParser:
    def test_parses_clean_json_array(self):
        raw = '[{"entity":"user","attribute":"lives_in","value":"Delhi","confidence":0.9}]'
        facts = parse_facts(raw)
        assert len(facts) == 1
        assert facts[0].entity == "user"
        assert facts[0].attribute == "lives_in"
        assert facts[0].value == "Delhi"

    def test_strips_code_fences_and_prose(self):
        raw = (
            "Sure, here you go:\n```json\n"
            '[{"entity":"user","attribute":"likes","value":"tea","confidence":0.8}]\n'
            "```\nHope this helps!"
        )
        facts = parse_facts(raw)
        assert [(f.attribute, f.value) for f in facts] == [("likes", "tea")]

    def test_accepts_single_object(self):
        facts = parse_facts('{"entity":"user","attribute":"age","value":"27"}')
        assert len(facts) == 1
        assert facts[0].value == "27"
        assert facts[0].confidence == 1.0  # default when missing

    def test_snake_cases_attribute(self):
        facts = parse_facts('[{"entity":"user","attribute":"Lives In","value":"Pune"}]')
        assert facts[0].attribute == "lives_in"

    def test_clamps_confidence(self):
        facts = parse_facts('[{"entity":"user","attribute":"likes","value":"x","confidence":5}]')
        assert facts[0].confidence == 1.0

    def test_skips_items_missing_required_fields(self):
        raw = '[{"entity":"","attribute":"a","value":"b"},{"entity":"user","attribute":"name","value":"Rob"}]'
        facts = parse_facts(raw)
        assert len(facts) == 1
        assert facts[0].value == "Rob"

    def test_garbage_returns_empty(self):
        assert parse_facts("I couldn't find any facts.") == []
        assert parse_facts("") == []

    def test_deduplicates_identical_triples(self):
        raw = (
            '[{"entity":"user","attribute":"likes","value":"tea"},'
            '{"entity":"user","attribute":"likes","value":"tea"}]'
        )
        assert len(parse_facts(raw)) == 1

    def test_assigns_stable_fact_id_and_session(self):
        facts = parse_facts(
            '[{"entity":"user","attribute":"name","value":"Rohit"}]', session_id="s1"
        )
        assert facts[0].fact_id  # non-empty
        assert facts[0].session_id == "s1"


class TestMockExtractor:
    def test_extracts_multiple_facts(self, mock_llm):
        turn = ConversationTurn(
            session_id="s1",
            user_message="My name is Rohit and I moved to Bangalore. I love hiking.",
        )
        facts = FactExtractor(mock_llm).extract(turn)
        attrs = {f.attribute: f.value for f in facts}
        assert attrs.get("name") == "Rohit"
        assert attrs.get("lives_in") == "Bangalore"
        assert attrs.get("likes") == "hiking"

    def test_update_phrasing_shares_key(self, mock_llm):
        """'now work as' must map to works_as so consolidation can update it."""
        turn = ConversationTurn(
            session_id="s1", user_message="I now work as a machine learning engineer."
        )
        facts = FactExtractor(mock_llm).extract(turn)
        assert any(
            f.attribute == "works_as" and "machine learning engineer" in f.value
            for f in facts
        )

    def test_no_facts_from_smalltalk(self, mock_llm):
        turn = ConversationTurn(session_id="s1", user_message="Hello, how are you today?")
        facts = FactExtractor(mock_llm).extract(turn)
        assert facts == []
