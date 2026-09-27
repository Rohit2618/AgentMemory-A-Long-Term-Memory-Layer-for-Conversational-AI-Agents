"""Shared pytest fixtures.

Forces the fully-offline, zero-cost configuration (mock LLM + mock embeddings +
in-memory / SQLite stores) so the whole suite runs with no API key and no
external services.
"""

from __future__ import annotations

import os

import pytest

# Set offline providers BEFORE any app module reads settings. This must run at
# import time (module level), not inside a fixture.
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("GRAPH_PROVIDER", "networkx")


@pytest.fixture
def mock_llm():
    from app.llm.mock_provider import MockLLMProvider

    return MockLLMProvider()


@pytest.fixture
def mock_embeddings():
    from app.embeddings.mock_provider import MockEmbeddingProvider

    return MockEmbeddingProvider()


@pytest.fixture
def store(mock_embeddings):
    """A fresh in-memory MemoryStore with deterministic embeddings."""
    from app.storage.in_memory import InMemoryStore

    return InMemoryStore(embeddings=mock_embeddings)
