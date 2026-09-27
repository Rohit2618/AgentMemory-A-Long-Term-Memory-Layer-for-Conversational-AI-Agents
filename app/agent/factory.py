"""Factory for a process-wide MemoryAgent bound to the configured store."""

from __future__ import annotations

from functools import lru_cache

from app.agent.memory_agent import MemoryAgent
from app.storage.factory import get_memory_store


@lru_cache
def get_agent() -> MemoryAgent:
    """Return a cached MemoryAgent using the configured composite store."""
    return MemoryAgent(get_memory_store())
