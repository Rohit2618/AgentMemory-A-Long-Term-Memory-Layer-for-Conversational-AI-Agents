"""LLM provider interface.

Everything in the codebase that needs an LLM depends on this small abstract
class — never on Groq or LangChain directly. Swapping to Gemini or any other
provider later means writing one new subclass and registering it in
``factory.py``; no other module changes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """A normalized completion result, independent of the backend used."""

    text: str
    model: str
    provider: str


class LLMProvider(abc.ABC):
    """Abstract chat-completion provider.

    Two "quality tiers" are exposed so callers can pick a cheap/fast model for
    simple tasks and a stronger model for extraction/consolidation reasoning,
    without knowing the concrete model names.
    """

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        quality: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Return a completion for ``prompt``.

        Args:
            prompt: The user-role content.
            system: Optional system-role instruction.
            quality: If True, use the higher-quality (slower) model tier.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Upper bound on generated tokens.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable provider identifier (for logs / health)."""
        raise NotImplementedError
