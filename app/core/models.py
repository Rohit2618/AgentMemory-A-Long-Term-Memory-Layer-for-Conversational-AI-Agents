"""Domain models shared across the whole pipeline.

These are plain pydantic models (not DB rows). They form the common vocabulary
that ingestion, extraction, consolidation, storage and retrieval all speak.
Keeping them in one place makes the data flow easy to follow in interviews.
"""

from __future__ import annotations

import enum
import hashlib
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationTurn(BaseModel):
    """One exchange: what the user said and how the agent replied."""

    session_id: str
    user_message: str
    agent_response: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class Fact(BaseModel):
    """A single atomic memory extracted from conversation.

    Modeled as an (entity, attribute, value) triple — the natural shape for a
    graph — plus a confidence score the extractor assigns.

    Example: entity="user", attribute="lives_in", value="Bangalore".
    """

    entity: str
    attribute: str
    value: str
    confidence: float = 1.0

    # Provenance / bookkeeping (filled in as the fact moves through the pipeline).
    fact_id: str = ""
    session_id: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def normalized_key(self) -> str:
        """A stable key for (entity, attribute), used to detect updates.

        Two facts with the same entity+attribute are candidates for an *update*
        (e.g. the user's city changing), regardless of value.
        """
        return f"{self.entity.strip().lower()}::{self.attribute.strip().lower()}"

    def text(self) -> str:
        """Human/embedding-friendly rendering of the triple."""
        return f"{self.entity} {self.attribute.replace('_', ' ')} {self.value}".strip()

    def ensure_id(self) -> "Fact":
        """Assign a deterministic id if one isn't set yet, then return self."""
        if not self.fact_id:
            raw = f"{self.normalized_key()}::{self.value}::{self.created_at.isoformat()}"
            self.fact_id = hashlib.sha1(raw.encode()).hexdigest()[:16]
        return self


class ConsolidationAction(str, enum.Enum):
    """The four possible outcomes when consolidating a candidate fact."""

    ADDED = "added"       # brand-new fact, no related memory existed
    UPDATED = "updated"   # replaced the value of an existing fact
    MERGED = "merged"     # combined with a near-duplicate (kept one)
    REJECTED = "rejected" # redundant / lower-confidence duplicate, dropped


class ConsolidationDecision(BaseModel):
    """Audit record of a single consolidation decision.

    Every candidate fact produces exactly one of these, which we persist so the
    evaluation harness can count duplicates/conflicts and so the behaviour is
    debuggable.
    """

    action: ConsolidationAction
    candidate: Fact
    related_fact_id: Optional[str] = None  # the existing fact we acted against
    reason: str = ""
    similarity: Optional[float] = None
    created_at: datetime = Field(default_factory=_utcnow)


class RetrievedMemory(BaseModel):
    """A fact returned by retrieval, with the score that ranked it."""

    fact: Fact
    score: float
    source: str = "vector"  # "vector" | "graph" | "hybrid"
