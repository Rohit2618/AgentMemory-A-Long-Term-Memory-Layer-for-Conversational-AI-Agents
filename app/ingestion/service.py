"""Ingestion service.

A thin facade the API (and later the agent) uses to record conversation turns.
It exists so callers depend on a stable function rather than the repository/ORM
directly, and so cross-cutting concerns (validation, logging) live in one place.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.core.models import ConversationTurn
from app.ingestion import repository

logger = get_logger(__name__)


def ingest_turn(
    session: Session,
    *,
    session_id: str,
    user_message: str,
    agent_response: str = "",
) -> tuple[int, ConversationTurn]:
    """Validate and persist a conversation turn.

    Returns the new row id and the domain object that was stored.
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id is required")
    if not user_message or not user_message.strip():
        raise ValueError("user_message is required")

    turn = ConversationTurn(
        session_id=session_id.strip(),
        user_message=user_message.strip(),
        agent_response=(agent_response or "").strip(),
    )
    turn_id = repository.save_turn(session, turn)
    logger.info("Ingested turn id=%s session=%s", turn_id, session_id)
    return turn_id, turn
