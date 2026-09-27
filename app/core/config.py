"""Central configuration.

All tunables live here and are loaded from environment variables (or a local
``.env`` file) via pydantic-settings. Nothing else in the codebase reads
``os.environ`` directly — that keeps configuration discoverable and testable.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings.

    Field names map to UPPER_CASE env vars (case-insensitive). See
    ``.env.example`` for the full documented list.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM provider ----
    llm_provider: str = "mock"  # "groq" | "mock"
    groq_api_key: str = ""
    groq_fast_model: str = "llama-3.1-8b-instant"
    groq_quality_model: str = "llama-3.3-70b-versatile"

    # ---- Embeddings ----
    embedding_provider: str = "local"  # "local" | "mock"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ---- Relational store ----
    # Empty -> fall back to local SQLite file (see ingestion.database).
    database_url: str = ""

    # ---- Graph store ----
    graph_provider: str = "networkx"  # "neo4j" | "networkx"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "agentmemory"

    # ---- Vector store ----
    chroma_persist_dir: str = "./.data/chroma"

    # ---- Retrieval tuning ----
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.25

    @property
    def effective_database_url(self) -> str:
        """Return the DB URL, defaulting to a local SQLite file when unset."""
        if self.database_url:
            return self.database_url
        return "sqlite:///./.data/agentmemory.sqlite3"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (single source of truth)."""
    return Settings()
