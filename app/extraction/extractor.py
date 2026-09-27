"""Fact extraction service.

Ties the prompt + LLM call + parser together into a single function the agent
uses after each turn. Uses the higher-quality model tier because extraction is
the reasoning-heavy step where accuracy matters most.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.core.models import ConversationTurn, Fact
from app.extraction.parser import parse_facts
from app.extraction.prompts import SYSTEM, build_extraction_prompt
from app.llm.base import LLMProvider
from app.llm.factory import get_llm

logger = get_logger(__name__)


class FactExtractor:
    """Extracts candidate facts from a conversation turn via an LLM."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        # Allow injection for testing; default to the configured provider.
        self._llm = llm or get_llm()

    def extract(self, turn: ConversationTurn) -> list[Fact]:
        """Return candidate facts extracted from a single turn."""
        prompt = build_extraction_prompt(turn)
        response = self._llm.complete(
            prompt,
            system=SYSTEM,
            quality=True,       # extraction warrants the stronger model
            temperature=0.0,    # deterministic structured output
            max_tokens=800,
        )
        facts = parse_facts(response.text, session_id=turn.session_id)
        logger.info(
            "Extraction: %d fact(s) from turn (session=%s) via %s",
            len(facts),
            turn.session_id,
            response.provider,
        )
        return facts
