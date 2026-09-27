"""The three memory configurations being compared.

Each config exposes the same tiny interface so the harness can treat them
uniformly:

    build_memory(utterances) -> (construction_time_s, duplicate_count)
    answer(question)         -> (answer_text, retrieval_latency_s)

  a) NoMemoryConfig      — stateless. Never stores or retrieves anything.
  b) NaiveMemoryConfig   — flat vector store, appends EVERY extracted fact with
                           NO consolidation (duplicates + stale values pile up).
  c) AgentMemoryConfig   — full system: graph+vector store + consolidation.

"Duplicate count" = facts stored beyond the number of distinct (entity,
attribute) slots — i.e. how much redundant/conflicting memory accumulated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.consolidation.engine import ConsolidationEngine
from app.core.models import ConversationTurn
from app.embeddings.base import EmbeddingProvider
from app.extraction.extractor import FactExtractor
from app.llm.base import LLMProvider
from app.retrieval.retriever import HybridRetriever
from app.storage.in_memory import InMemoryStore


@dataclass
class MemoryBuildStats:
    construction_time_s: float
    facts_stored: int
    duplicate_count: int


def _count_duplicates(store: InMemoryStore) -> int:
    """Facts stored beyond the number of distinct (entity, attribute) slots."""
    facts = store.all_facts()
    slots = {f.normalized_key() for f in facts}
    return max(0, len(facts) - len(slots))


class NoMemoryConfig:
    """Stateless baseline: no storage, no retrieval."""

    name = "No memory"

    def __init__(self, llm: LLMProvider, **_: object) -> None:
        self._llm = llm

    def build_memory(self, utterances: list[str]) -> MemoryBuildStats:
        # Nothing to build.
        return MemoryBuildStats(construction_time_s=0.0, facts_stored=0, duplicate_count=0)

    def answer(self, question: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        # No memory injected at all.
        prompt = f"User: {question}\nAssistant:"
        out = self._llm.complete(prompt, quality=False, max_tokens=256)
        return out.text.strip(), time.perf_counter() - t0


class NaiveMemoryConfig:
    """Flat vector store; append every fact, NO consolidation."""

    name = "Naive vector"

    def __init__(self, llm: LLMProvider, embeddings: EmbeddingProvider) -> None:
        self._llm = llm
        self._store = InMemoryStore(embeddings=embeddings)
        self._extractor = FactExtractor(llm)
        self._retriever = HybridRetriever(self._store)

    def build_memory(self, utterances: list[str]) -> MemoryBuildStats:
        t0 = time.perf_counter()
        for utt in utterances:
            turn = ConversationTurn(session_id="eval", user_message=utt)
            for fact in self._extractor.extract(turn):
                # NAIVE: unconditionally add — no update/merge/reject. Every
                # restatement or changed value becomes a new stored fact.
                self._store.add_fact(fact)
        elapsed = time.perf_counter() - t0
        return MemoryBuildStats(
            construction_time_s=elapsed,
            facts_stored=len(self._store.all_facts()),
            duplicate_count=_count_duplicates(self._store),
        )

    def answer(self, question: str) -> tuple[str, float]:
        from app.agent.prompts import SYSTEM, build_chat_prompt

        t0 = time.perf_counter()
        memories = self._retriever.retrieve(question)
        latency = time.perf_counter() - t0
        prompt = build_chat_prompt(question, memories)
        out = self._llm.complete(prompt, system=SYSTEM, quality=False, max_tokens=256)
        return out.text.strip(), latency


class AgentMemoryConfig:
    """Full AgentMemory: graph+vector store + consolidation."""

    name = "AgentMemory (full)"

    def __init__(self, llm: LLMProvider, embeddings: EmbeddingProvider) -> None:
        self._llm = llm
        # Use the in-memory store for a fast, dependency-free eval run, but WITH
        # the consolidation engine — that's the variable under test. (The same
        # engine runs unchanged against the Neo4j+Chroma composite store.)
        self._store = InMemoryStore(embeddings=embeddings)
        self._extractor = FactExtractor(llm)
        self._engine = ConsolidationEngine(self._store)
        self._retriever = HybridRetriever(self._store)

    def build_memory(self, utterances: list[str]) -> MemoryBuildStats:
        t0 = time.perf_counter()
        for utt in utterances:
            turn = ConversationTurn(session_id="eval", user_message=utt)
            for fact in self._extractor.extract(turn):
                # FULL: reconcile against existing memory (add/update/merge/reject).
                self._engine.consolidate(fact)
        elapsed = time.perf_counter() - t0
        return MemoryBuildStats(
            construction_time_s=elapsed,
            facts_stored=len(self._store.all_facts()),
            duplicate_count=_count_duplicates(self._store),
        )

    def answer(self, question: str) -> tuple[str, float]:
        from app.agent.prompts import SYSTEM, build_chat_prompt

        t0 = time.perf_counter()
        memories = self._retriever.retrieve(question)
        latency = time.perf_counter() - t0
        prompt = build_chat_prompt(question, memories)
        out = self._llm.complete(prompt, system=SYSTEM, quality=False, max_tokens=256)
        return out.text.strip(), latency
