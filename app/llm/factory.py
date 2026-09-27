"""LLM provider factory — the single place that maps config -> concrete provider.

Call ``get_llm()`` anywhere you need an LLM. To add a new backend (e.g. Gemini),
write a subclass of ``LLMProvider`` and add one branch here. Nothing else changes.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.llm.base import LLMProvider

logger = get_logger(__name__)


def build_llm(settings: Settings) -> LLMProvider:
    """Construct an LLM provider from settings (no caching)."""
    provider = settings.llm_provider.lower().strip()

    if provider == "mock":
        from app.llm.mock_provider import MockLLMProvider

        logger.info("LLM provider: mock (offline, no API cost)")
        return MockLLMProvider()

    if provider == "groq":
        from app.llm.groq_provider import GroqProvider

        logger.info("LLM provider: groq (%s / %s)", settings.groq_fast_model, settings.groq_quality_model)
        return GroqProvider(settings)

    raise ValueError(
        f"Unknown LLM_PROVIDER={settings.llm_provider!r}. Expected 'groq' or 'mock'."
    )


@lru_cache
def get_llm() -> LLMProvider:
    """Return a cached LLM provider built from the active settings."""
    return build_llm(get_settings())
