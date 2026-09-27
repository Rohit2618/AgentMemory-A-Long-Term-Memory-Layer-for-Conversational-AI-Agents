"""LLM-as-judge scoring.

Asks the LLM whether an answer matches the expected value. Uses the JUDGE_MARKER
convention so the mock provider can score offline (token-overlap heuristic); a
real LLM (Groq) reads the natural-language instruction and returns 1/0.
"""

from __future__ import annotations

from app.llm.base import LLMProvider
from app.llm.mock_provider import JUDGE_MARKER

_JUDGE_SYSTEM = (
    "You are a strict grader. Decide whether the ANSWER correctly contains the "
    "information in EXPECTED. Reply with exactly '1' if correct, '0' if not."
)


def judge_answer(llm: LLMProvider, *, question: str, expected: str, answer: str) -> int:
    """Return 1 if ``answer`` satisfies ``expected`` for ``question``, else 0."""
    prompt = (
        f"{JUDGE_MARKER}\n"
        f"QUESTION: {question}\n"
        f"EXPECTED: {expected}\n"
        f"ANSWER: {answer}\n"
        "Is the answer correct? Reply 1 or 0."
    )
    out = llm.complete(prompt, system=_JUDGE_SYSTEM, quality=False, temperature=0.0, max_tokens=4)
    text = out.text.strip()
    # Be lenient parsing the verdict: look for a leading 1.
    return 1 if text.startswith("1") else 0
