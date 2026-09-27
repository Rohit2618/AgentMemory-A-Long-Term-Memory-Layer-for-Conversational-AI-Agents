"""Relational database wiring (SQLAlchemy).

Uses Postgres when ``DATABASE_URL`` is set (docker-compose does this), otherwise
falls back to a local SQLite file so the app runs with zero infra. All other
modules obtain sessions via ``session_scope()`` or the FastAPI ``get_db``
dependency — they never build engines themselves.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _make_engine():
    settings = get_settings()
    url = settings.effective_database_url

    connect_args: dict = {}
    if url.startswith("sqlite"):
        # SQLite: allow use across FastAPI's threadpool, and ensure the parent
        # directory for the .data file exists.
        connect_args = {"check_same_thread": False}
        db_path = url.replace("sqlite:///", "")
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Relational store: SQLite (%s)", url)
    else:
        logger.info("Relational store: %s", url.split("@")[-1])

    return create_engine(url, connect_args=connect_args, future=True)


# Engine + session factory are process-wide singletons.
engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create all tables. Safe to call repeatedly (idempotent)."""
    # Import models so they register on Base.metadata before create_all.
    from app.ingestion import orm_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured.")


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session context manager (commit on success, rollback on error)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session (commit/rollback handled here)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
