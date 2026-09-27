"""Chat prompt construction for the memory-augmented agent.

The retrieved memories are injected as a "Known facts about the user:" block.
This exact label matters twice:
  * a real LLM (Groq) reads it as grounding context, and
  * the mock provider looks for this label to echo the memories, so the demo
    visibly uses memory even without an API key.
"""

from __future__ import annotations

from app.core.models import RetrievedMemory

SYSTEM = (
    "You are a helpful assistant with long-term memory of the user. "
    "Use the known facts below when they are relevant to answer accurately and "
    "personally. If the facts don't cover the question, answer normally and do "
    "not invent details about the user."
)


def build_chat_prompt(message: str, memories: list[RetrievedMemory]) -> str:
    """Assemble the chat prompt, injecting retrieved memories as context."""
    lines: list[str] = []
    if memories:
        lines.append("Known facts about the user:")
        for m in memories:
            # e.g. "- lives in Bangalore"
            lines.append(f"- {m.fact.attribute.replace('_', ' ')} {m.fact.value}")
        lines.append("")  # blank line separates context from the question

    lines.append(f"User: {message}")
    lines.append("Assistant:")
    return "\n".join(lines)
