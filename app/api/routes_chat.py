"""Chat HTTP route: the end-to-end memory-augmented agent endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agent.factory import get_agent
from app.api.schemas import ChatRequest, ChatResponse, MemoryOut
from app.ingestion.database import get_db

router = APIRouter(tags=["agent"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Send a message; the agent retrieves memories, replies, and updates memory."""
    agent = get_agent()
    try:
        result = agent.chat(db, session_id=req.session_id, message=req.message)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ChatResponse(
        session_id=req.session_id,
        response=result.response,
        memories_used=[
            MemoryOut(
                entity=m.fact.entity,
                attribute=m.fact.attribute,
                value=m.fact.value,
                score=round(m.score, 4),
                source=m.source,
            )
            for m in result.memories_used
        ],
        facts_learned=len(result.decisions),
    )
