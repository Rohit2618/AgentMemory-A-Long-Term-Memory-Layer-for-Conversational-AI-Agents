"""Groq-backed LLM provider (free tier, Llama 3.1 / 3.3).

Uses ``langchain-groq`` so we benefit from LangChain's message abstractions and
can later reuse chains/prompt templates. The API key is read from settings
(never hardcoded). Import of the SDK is deferred so the package stays importable
in mock mode without ``langchain-groq`` installed.
"""

from __future__ import annotations

from app.core.config import Settings
from app.core.logging import get_logger
from app.llm.base import LLMProvider, LLMResponse

logger = get_logger(__name__)


class GroqProvider(LLMProvider):
    """LLM provider backed by Groq's OpenAI-compatible chat API."""

    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is empty. Set it in .env, or use LLM_PROVIDER=mock "
                "to run without an API key."
            )
        self._settings = settings
        self._fast_model = settings.groq_fast_model
        self._quality_model = settings.groq_quality_model
        # Lazy import so mock-mode users don't need the dependency installed.
        from langchain_groq import ChatGroq  # noqa: WPS433 (intentional local import)

        self._ChatGroq = ChatGroq
        self._api_key = settings.groq_api_key
        # Cache one client per model to avoid re-instantiation per call.
        self._clients: dict[str, object] = {}

    def _client(self, model: str, temperature: float, max_tokens: int):
        key = f"{model}:{temperature}:{max_tokens}"
        if key not in self._clients:
            self._clients[key] = self._ChatGroq(
                api_key=self._api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._clients[key]

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        quality: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        model = self._quality_model if quality else self._fast_model
        client = self._client(model, temperature, max_tokens)

        messages = []
        if system:
            messages.append(("system", system))
        messages.append(("human", prompt))

        result = client.invoke(messages)
        text = getattr(result, "content", str(result))
        return LLMResponse(text=text, model=model, provider=self.name())

    def name(self) -> str:
        return "groq"
