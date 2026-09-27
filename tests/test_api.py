"""End-to-end API smoke tests (health, ingest, chat) in full mock mode.

These exercise the wired FastAPI app against SQLite + in-process stores, proving
the whole pipeline responds over HTTP with no external services.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Point every store at throwaway temp locations so tests don't touch .data/.
    tmp = tempfile.mkdtemp()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/test.sqlite3"
    os.environ["CHROMA_PERSIST_DIR"] = f"{tmp}/chroma"

    # Import AFTER env is set and clear cached singletons that read settings.
    from app.core import config
    from app.embeddings import factory as emb_factory
    from app.llm import factory as llm_factory
    from app.storage import factory as store_factory
    from app.agent import factory as agent_factory

    config.get_settings.cache_clear()
    llm_factory.get_llm.cache_clear()
    emb_factory.get_embeddings.cache_clear()
    store_factory.get_memory_store.cache_clear()
    agent_factory.get_agent.cache_clear()

    from app.api.main import app

    with TestClient(app) as c:
        yield c


class TestApi:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["llm_provider"] == "mock"

    def test_ingest_and_list(self, client):
        r = client.post(
            "/ingest",
            json={"session_id": "t", "user_message": "hi", "agent_response": "hello"},
        )
        assert r.status_code == 200
        assert r.json()["turn_id"] >= 1

        lst = client.get("/ingest/t")
        assert lst.status_code == 200
        assert len(lst.json()) == 1

    def test_ingest_validation(self, client):
        r = client.post("/ingest", json={"session_id": "", "user_message": "x"})
        assert r.status_code == 422

    def test_chat_learns_and_recalls(self, client):
        # Teach a fact.
        r1 = client.post("/chat", json={"session_id": "c", "message": "I live in Bangalore."})
        assert r1.status_code == 200
        assert r1.json()["facts_learned"] >= 1

        # Ask about it; memory should be retrieved.
        r2 = client.post("/chat", json={"session_id": "c", "message": "Where do I live?"})
        assert r2.status_code == 200
        used = r2.json()["memories_used"]
        assert any(m["attribute"] == "lives_in" and m["value"] == "Bangalore" for m in used)

    def test_chat_updates_on_conflict(self, client):
        client.post("/chat", json={"session_id": "d", "message": "I live in Delhi."})
        client.post("/chat", json={"session_id": "d", "message": "I moved to Bangalore."})

        from app.storage.factory import get_memory_store

        lives = get_memory_store().get_by_key("user", "lives_in")
        assert len(lives) == 1
        assert lives[0].value.lower() == "bangalore"
