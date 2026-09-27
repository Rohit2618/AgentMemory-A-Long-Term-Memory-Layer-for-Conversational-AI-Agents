"""SQLAlchemy ORM models for the relational store.

Two tables:
  - conversation_turns: the raw log of every user/agent exchange (source of truth).
  - consolidation_decisions: an audit row for every candidate fact the
    consolidation engine processed (added/updated/merged/rejected). The eval
    harness reads this to count duplicates/conflicts.

Facts themselves live in the graph + vector stores, not here — this table is
purely the immutable conversation log + decision audit.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.ingestion.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationTurnRow(Base):
    """A single stored conversation turn."""

    __tablename__ = "conversation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    agent_response: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class ConsolidationDecisionRow(Base):
    """Audit record of a consolidation decision for one candidate fact."""

    __tablename__ = "consolidation_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True, default="")
    action: Mapped[str] = mapped_column(String(16), index=True, nullable=False)

    # Candidate fact fields (denormalized so the log is self-contained).
    entity: Mapped[str] = mapped_column(String(256), default="")
    attribute: Mapped[str] = mapped_column(String(256), default="")
    value: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    related_fact_id: Mapped[str] = mapped_column(String(64), default="")
    similarity: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
