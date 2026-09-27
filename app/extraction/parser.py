"""Robust parsing of LLM output into a list of facts.

Real LLMs are messy: they wrap JSON in ```json fences, add a sentence before or
after, or emit a single object instead of an array. This module extracts the
JSON payload defensively and validates each item into a ``Fact``. Anything it
can't parse is skipped (logged) rather than crashing the pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.logging import get_logger
from app.core.models import Fact

logger = get_logger(__name__)

# Matches a ```json ... ``` or ``` ... ``` fenced block.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_facts(raw: str, *, session_id: str = "") -> list[Fact]:
    """Parse raw LLM text into validated, de-duplicated ``Fact`` objects."""
    payload = _extract_json(raw)
    if payload is None:
        logger.warning("Extraction: no JSON found in LLM output; returning no facts.")
        return []

    # Accept either a list of objects or a single object.
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        logger.warning("Extraction: JSON is not a list/object; skipping.")
        return []

    facts: list[Fact] = []
    seen: set[str] = set()
    for item in payload:
        fact = _coerce_fact(item, session_id=session_id)
        if fact is None:
            continue
        # De-duplicate identical (entity, attribute, value) triples.
        dedupe_key = f"{fact.normalized_key()}::{fact.value.strip().lower()}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        facts.append(fact)
    return facts


def _extract_json(raw: str) -> Any | None:
    """Best-effort extraction of a JSON value from arbitrary LLM text."""
    if not raw or not raw.strip():
        return None

    candidates: list[str] = []

    # 1. Fenced code block(s).
    candidates.extend(m.group(1).strip() for m in _FENCE_RE.finditer(raw))
    # 2. The whole string.
    candidates.append(raw.strip())
    # 3. The substring from the first bracket to the last matching one.
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = raw.find(open_ch)
        end = raw.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidates.append(raw[start : end + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _coerce_fact(item: Any, *, session_id: str) -> Fact | None:
    """Validate one JSON item into a Fact, or return None if invalid."""
    if not isinstance(item, dict):
        return None

    entity = str(item.get("entity", "")).strip()
    attribute = str(item.get("attribute", "")).strip()
    value = str(item.get("value", "")).strip()

    # entity/attribute/value are all required and non-empty.
    if not entity or not attribute or not value:
        return None

    # Confidence: clamp to [0, 1], default 1.0 when missing/unparseable.
    try:
        confidence = float(item.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    # Normalize attribute to snake_case for stable graph keys.
    attribute = re.sub(r"\s+", "_", attribute.lower()).strip("_")

    fact = Fact(
        entity=entity.lower(),
        attribute=attribute,
        value=value,
        confidence=confidence,
        session_id=session_id,
    )
    return fact.ensure_id()
