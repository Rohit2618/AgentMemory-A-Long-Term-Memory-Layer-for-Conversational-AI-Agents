"""FastAPI application entrypoint.

At scaffold time this only exposes a health check. Ingestion, agent and memory
routes are wired in during later build steps (tasks 3, 8).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import routes_chat, routes_ingest
from app.core.config import get_settings
from app.ingestion.database import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Ensure database tables exist before serving requests."""
    init_db()
    yield


app = FastAPI(
    title="AgentMemory",
    version=__version__,
    description="A long-term memory layer for conversational AI agents.",
    lifespan=lifespan,
)

app.include_router(routes_ingest.router)
app.include_router(routes_chat.router)


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness probe + a peek at which providers are active."""
    s = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": s.llm_provider,
        "embedding_provider": s.embedding_provider,
        "graph_provider": s.graph_provider,
    }
