"""Ingestion HTTP routes: store and read raw conversation turns."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.schemas import IngestRequest, IngestResponse, TurnOut
from app.ingestion import repository, service
from app.ingestion.database import get_db

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("", response_model=IngestResponse)
def ingest(req: IngestRequest, db: Session = Depends(get_db)) -> IngestResponse:
    """Store a single conversation turn (user message + optional agent response)."""
    try:
        turn_id, turn = service.ingest_turn(
            db,
            session_id=req.session_id,
            user_message=req.user_message,
            agent_response=req.agent_response,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return IngestResponse(
        turn_id=turn_id,
        session_id=turn.session_id,
        stored_at=turn.created_at,
    )


@router.get("/{session_id}", response_model=list[TurnOut])
def list_turns(session_id: str, db: Session = Depends(get_db)) -> list[TurnOut]:
    """Return the stored turns for a session (oldest first)."""
    turns = repository.get_turns(db, session_id)
    return [
        TurnOut(
            session_id=t.session_id,
            user_message=t.user_message,
            agent_response=t.agent_response,
            created_at=t.created_at,
        )
        for t in turns
    ]
