"""MemoryAgent — the end-to-end chat agent that uses the memory system.

This ties every stage together into the loop the whole project exists to enable:

    1. RETRIEVE   relevant memories for the incoming message (hybrid retrieval).
    2. RESPOND    call the LLM with those memories injected as context.
    3. INGEST     persist the raw turn (user + agent) to the relational store.
    4. EXTRACT    pull candidate facts from the turn.
    5. CONSOLIDATE reconcile each fact against memory (add/update/merge/reject)
                  and log every decision for evaluation.

Steps 1-2 happen *before* replying (memory informs the answer); steps 3-5 happen
*after* (the exchange updates memory for next time).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.agent.prompts import SYSTEM, build_chat_prompt
from app.consolidation.engine import ConsolidationEngine
from app.core.logging import get_logger
from app.core.models import ConsolidationDecision, ConversationTurn, RetrievedMemory
from app.extraction.extractor import FactExtractor
from app.ingestion import repository, service
from app.llm.base import LLMProvider
from app.llm.factory import get_llm
from app.retrieval.retriever import HybridRetriever
from app.storage.base import MemoryStore

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Everything the agent produced for one turn."""

    response: str
    memories_used: list[RetrievedMemory] = field(default_factory=list)
    decisions: list[ConsolidationDecision] = field(default_factory=list)


class MemoryAgent:
    """Orchestrates retrieval, response, and memory update for one chat turn."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        llm: LLMProvider | None = None,
        extractor: FactExtractor | None = None,
        engine: ConsolidationEngine | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self._store = store
        self._llm = llm or get_llm()
        self._extractor = extractor or FactExtractor(self._llm)
        self._engine = engine or ConsolidationEngine(store)
        self._retriever = retriever or HybridRetriever(store)

    def chat(self, db: Session, *, session_id: str, message: str) -> AgentResult:
        """Handle one user message end-to-end. ``db`` is a live DB session."""
        if not message or not message.strip():
            raise ValueError("message is required")
        message = message.strip()

        # 1. RETRIEVE ------------------------------------------------------
        memories = self._retriever.retrieve(message)

        # 2. RESPOND -------------------------------------------------------
        prompt = build_chat_prompt(message, memories)
        llm_out = self._llm.complete(prompt, system=SYSTEM, quality=False, max_tokens=512)
        response = llm_out.text.strip()

        # 3. INGEST (persist the raw turn) ---------------------------------
        service.ingest_turn(
            db,
            session_id=session_id,
            user_message=message,
            agent_response=response,
        )

        # 4. EXTRACT candidate facts from this turn ------------------------
        turn = ConversationTurn(
            session_id=session_id, user_message=message, agent_response=response
        )
        candidates = self._extractor.extract(turn)

        # 5. CONSOLIDATE each fact + log the decision ----------------------
        decisions: list[ConsolidationDecision] = []
        for cand in candidates:
            cand.session_id = session_id
            decision = self._engine.consolidate(cand)
            repository.save_decision(db, decision)
            decisions.append(decision)

        logger.info(
            "Agent turn: session=%s used %d memory(ies), processed %d fact(s)",
            session_id,
            len(memories),
            len(decisions),
        )
        return AgentResult(response=response, memories_used=memories, decisions=decisions)
