"""Deterministic, offline mock LLM provider.

This is what makes the whole project **free and testable**. It never calls a
network API. Instead it recognizes the two prompt "shapes" the app uses and
returns plausible, deterministic output:

  1. Extraction prompts  -> returns a JSON array of {entity, attribute, value,
     confidence} facts, parsed heuristically from the conversation text.
  2. Judge prompts        -> returns a 0/1 correctness score (LLM-as-judge).
  3. Anything else (agent chat) -> a canned reply that echoes any injected
     memories, so the agent endpoint is demonstrable without a real LLM.

The heuristics are intentionally simple and well-commented — the *real* work is
meant to be done by Groq in production. The mock only needs to exercise the
pipeline plumbing end-to-end.
"""

from __future__ import annotations

import json
import re

from app.llm.base import LLMProvider, LLMResponse

# Sentinel markers the prompt builders embed so the mock can route reliably
# (see app/extraction and app/eval). Real providers ignore these.
EXTRACTION_MARKER = "[[TASK:EXTRACT]]"
JUDGE_MARKER = "[[TASK:JUDGE]]"


# A few hand-written patterns covering the kinds of facts the synthetic eval
# dataset contains. Each maps a regex to an (attribute, value-group) rule.
# Group 1 of the regex is taken as the value unless a literal value is given.
_PATTERNS: list[tuple[str, str]] = [
    # location (the "(?:just )?" and "now" handle updates across sessions)
    (r"\bi (?:now )?(?:live|living|reside|am based) in ([A-Z][\w .'-]+)", "lives_in"),
    (r"\bi (?:just |recently )?moved to ([A-Z][\w .'-]+)", "lives_in"),
    (r"\bi'?m from ([A-Z][\w .'-]+)", "from"),
    # work / role ("now work as" is an update)
    (r"\bi (?:now )?work (?:as|at) (?:an? )?([\w .'-]+)", "works_as"),
    (r"\bi'?m (?:now )?an? ([\w .'-]*(?:engineer|developer|designer|manager|scientist|doctor|teacher))", "works_as"),
    # preferences
    (r"\bmy favorite (\w+) is (?:now )?([\w .'-]+)", "favorite"),  # special: 2 groups
    (r"\bi (?:love|like|prefer|enjoy) ([\w .'-]+)", "likes"),
    (r"\bi (?:hate|dislike|can'?t stand) ([\w .'-]+)", "dislikes"),
    (r"\bi became (?:a |an )?([\w .'-]+)", "became"),
    # possessions (drive/own/sold->new)
    (r"\bi (?:now )?drive an? ([\w .'-]+)", "drives"),
    (r"\bnow i drive an? ([\w .'-]+)", "drives"),
    (r"\bi (?:own|have) an? ([\w .'-]*(?:car|dog|cat|bike|house|tesla|honda|toyota))", "owns"),
    # identity
    (r"\bmy name is ([A-Z][\w .'-]+)", "name"),
    (r"\bi(?:'m| am) (\d{1,3}) years old", "age"),
    (r"\bi have an? ([\w .'-]+)", "has"),
]


class MockLLMProvider(LLMProvider):
    """Offline provider that fakes just enough intelligence to run the pipeline."""

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        quality: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        text: str
        if EXTRACTION_MARKER in prompt:
            text = self._fake_extraction(prompt)
        elif JUDGE_MARKER in prompt:
            text = self._fake_judge(prompt)
        else:
            text = self._fake_chat(prompt)
        return LLMResponse(text=text, model="mock", provider=self.name())

    # ------------------------------------------------------------------ #
    # Task 1: extraction -> JSON array of facts
    # ------------------------------------------------------------------ #
    def _fake_extraction(self, prompt: str) -> str:
        # The extraction prompt puts the raw conversation after the marker.
        convo = prompt.split(EXTRACTION_MARKER, 1)[-1]
        facts: list[dict] = []
        seen: set[tuple[str, str, str]] = set()

        for line in convo.splitlines():
            lowered = " " + line.strip()
            for pattern, attribute in _PATTERNS:
                for m in re.finditer(pattern, lowered, flags=re.IGNORECASE):
                    if attribute == "favorite" and m.lastindex == 2:
                        attr = f"favorite_{m.group(1).strip().lower()}"
                        value = m.group(2)
                    else:
                        attr = attribute
                        value = m.group(m.lastindex or 1)
                    value = _clean(value)
                    if not value:
                        continue
                    key = ("user", attr, value.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    facts.append(
                        {
                            "entity": "user",
                            "attribute": attr,
                            "value": value,
                            "confidence": 0.9,
                        }
                    )
        return json.dumps(facts)

    # ------------------------------------------------------------------ #
    # Task 2: LLM-as-judge -> "1" (correct) or "0" (incorrect)
    # ------------------------------------------------------------------ #
    def _fake_judge(self, prompt: str) -> str:
        # Convention in the judge prompt: it contains "EXPECTED: <x>" and
        # "ANSWER: <y>". We mark correct if the expected value's tokens appear
        # in the answer (case-insensitive) — a crude but deterministic proxy.
        expected = _grab(prompt, "EXPECTED:")
        answer = _grab(prompt, "ANSWER:")
        if not expected:
            return "0"
        exp_tokens = {t for t in re.split(r"\W+", expected.lower()) if len(t) > 2}
        ans_lower = answer.lower()
        hit = any(tok in ans_lower for tok in exp_tokens) if exp_tokens else False
        return "1" if hit else "0"

    # ------------------------------------------------------------------ #
    # Task 3: plain chat -> echo any injected memory context
    # ------------------------------------------------------------------ #
    def _fake_chat(self, prompt: str) -> str:
        # The agent injects a "Known facts about the user:" block. If present,
        # surface it so responses visibly use memory (great for demos/eval).
        mem = ""
        if "Known facts about the user:" in prompt:
            block = prompt.split("Known facts about the user:", 1)[-1]
            block = block.split("User:", 1)[0].strip()
            if block:
                mem = f" Based on what I remember ({block.replace(chr(10), '; ').strip()}),"
        question = _grab(prompt, "User:") or prompt.strip()[-200:]
        return f"[mock reply]{mem} regarding: {question.strip()}"

    def name(self) -> str:
        return "mock"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clean(value: str) -> str:
    """Trim a captured value down to a single clean phrase.

    Regexes can over-capture across sentence and clause boundaries, so we:
      1. Cut at the first sentence terminator (. ! ? ; ,) or clause word.
      2. Strip surrounding punctuation/whitespace.
      3. Drop trailing filler words.
    """
    # 1. Stop at the first sentence/clause boundary.
    value = re.split(r"[.!?;,]", value, maxsplit=1)[0]
    # Also stop at conjunctions / prepositions that start a new clause or a
    # purpose/time phrase (e.g. "Bangalore for a new job", "hiking on weekends").
    value = re.split(
        r"\band\b|\bbut\b|\bbecause\b|\bfor\b|\bon\b|\bso\b|\bwith\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    # 2. Strip punctuation/whitespace.
    value = value.strip().strip(".,!?;:").strip()
    # 3. Drop trailing filler / time expressions.
    value = re.sub(
        r"\s*\b(now|today|recently|a lot|really|too|last (?:month|week|year|night)|yesterday|this (?:month|week|year))\b\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip()


def _grab(text: str, label: str) -> str:
    """Return the remainder of the line following ``label``, if present."""
    idx = text.find(label)
    if idx == -1:
        return ""
    rest = text[idx + len(label):]
    return rest.splitlines()[0].strip() if rest else ""
