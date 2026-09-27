"""Repository layer over the relational store.

Thin functions that translate between ORM rows and domain models. Keeping this
separate from the ORM classes means callers work with plain domain objects and
the persistence details stay in one place.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.models import ConsolidationDecision, ConversationTurn
from app.ingestion.orm_models import ConsolidationDecisionRow, ConversationTurnRow


# --------------------------------------------------------------------------- #
# Conversation turns
# --------------------------------------------------------------------------- #
def save_turn(session: Session, turn: ConversationTurn) -> int:
    """Persist a conversation turn; return its new row id."""
    row = ConversationTurnRow(
        session_id=turn.session_id,
        user_message=turn.user_message,
        agent_response=turn.agent_response,
        created_at=turn.created_at,
    )
    session.add(row)
    session.flush()  # populate row.id without ending the transaction
    return row.id


def get_turns(session: Session, session_id: str, limit: int = 100) -> list[ConversationTurn]:
    """Return turns for a session, oldest first."""
    stmt = (
        select(ConversationTurnRow)
        .where(ConversationTurnRow.session_id == session_id)
        .order_by(ConversationTurnRow.created_at.asc(), ConversationTurnRow.id.asc())
        .limit(limit)
    )
    rows = session.execute(stmt).scalars().all()
    return [
        ConversationTurn(
            session_id=r.session_id,
            user_message=r.user_message,
            agent_response=r.agent_response,
            created_at=r.created_at,
        )
        for r in rows
    ]


def count_turns(session: Session, session_id: str | None = None) -> int:
    stmt = select(ConversationTurnRow)
    if session_id is not None:
        stmt = stmt.where(ConversationTurnRow.session_id == session_id)
    return len(session.execute(stmt).scalars().all())


# --------------------------------------------------------------------------- #
# Consolidation decision log
# --------------------------------------------------------------------------- #
def save_decision(session: Session, decision: ConsolidationDecision) -> int:
    """Persist a consolidation decision audit row; return its id."""
    c = decision.candidate
    row = ConsolidationDecisionRow(
        session_id=c.session_id,
        action=decision.action.value,
        entity=c.entity,
        attribute=c.attribute,
        value=c.value,
        confidence=c.confidence,
        related_fact_id=decision.related_fact_id or "",
        similarity=decision.similarity or 0.0,
        reason=decision.reason,
        created_at=decision.created_at,
    )
    session.add(row)
    session.flush()
    return row.id


def get_decisions(session: Session, session_id: str | None = None) -> list[ConsolidationDecisionRow]:
    """Return raw decision rows (used by the eval harness for counting)."""
    stmt = select(ConsolidationDecisionRow).order_by(ConsolidationDecisionRow.id.asc())
    if session_id is not None:
        stmt = stmt.where(ConsolidationDecisionRow.session_id == session_id)
    return list(session.execute(stmt).scalars().all())
