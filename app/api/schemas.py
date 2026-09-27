"""Pydantic request/response schemas for the HTTP API.

Kept separate from the internal domain models (app/core/models) so the wire
contract can evolve independently of internal representations.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---- Ingestion ---- #
class IngestRequest(BaseModel):
    session_id: str = Field(..., examples=["demo"])
    user_message: str = Field(..., examples=["I just moved to Bangalore."])
    agent_response: str = Field("", examples=["Nice, Bangalore is great!"])


class IngestResponse(BaseModel):
    turn_id: int
    session_id: str
    stored_at: datetime


class TurnOut(BaseModel):
    session_id: str
    user_message: str
    agent_response: str
    created_at: datetime


# ---- Chat (wired in Task 8) ---- #
class ChatRequest(BaseModel):
    session_id: str = Field(..., examples=["demo"])
    message: str = Field(..., examples=["Where do I live?"])


class MemoryOut(BaseModel):
    entity: str
    attribute: str
    value: str
    score: float
    source: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    memories_used: list[MemoryOut] = []
    facts_learned: int = 0
