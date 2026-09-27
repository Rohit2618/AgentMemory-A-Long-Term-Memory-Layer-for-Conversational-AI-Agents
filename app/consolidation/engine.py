"""Consolidation engine — the core novel piece of AgentMemory.

Naive memory appends every extracted fact, accumulating duplicates and
contradictions. This engine instead *reconciles* each candidate fact against
what's already stored, producing exactly one decision per candidate:

    ADDED    - genuinely new information; store it.
    UPDATED  - same (entity, attribute) as an existing fact but a DIFFERENT
               value -> the fact evolved (e.g. moved city). Replace in place.
    MERGED   - semantically the same fact as an existing one, just phrased/
               spelled differently (e.g. "Bangalore" vs "Bengaluru"). Keep one
               canonical fact, don't duplicate.
    REJECTED - redundant restatement of an existing fact with equal-or-lower
               confidence. Drop it.

Detection uses BOTH signals the brief asks for:
  1. **Graph / entity matching** — an exact (entity, attribute) lookup finds
     facts about the same thing regardless of how the value is worded.
  2. **Vector similarity** — a semantic search finds near-duplicate facts even
     when the attribute wording differs.

Every decision is returned as a ``ConsolidationDecision`` (with the related
fact id, similarity and a human-readable reason) so it can be logged to the
relational store for evaluation and debugging.
"""

from __future__ import annotations

import re

from app.core.logging import get_logger
from app.core.models import (
    ConsolidationAction,
    ConsolidationDecision,
    Fact,
)
from app.storage.base import MemoryStore

logger = get_logger(__name__)


class ConsolidationEngine:
    """Decides how each candidate fact should affect stored memory."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        merge_threshold: float = 0.82,
        relate_threshold: float = 0.60,
    ) -> None:
        """
        Args:
            store: the graph + vector memory backend.
            merge_threshold: cosine similarity at/above which two facts about
                the same entity are treated as the SAME fact (merge/reject).
            relate_threshold: similarity above which facts are considered
                "related" — used only for logging context, not decisions.
        """
        self._store = store
        self._merge_threshold = merge_threshold
        self._relate_threshold = relate_threshold

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def consolidate(self, candidate: Fact) -> ConsolidationDecision:
        """Reconcile one candidate fact against stored memory and apply it.

        Returns the decision (already applied to the store). The caller is
        responsible for persisting the decision to the audit log.
        """
        candidate = candidate.ensure_id()

        # --- Signal 1: exact (entity, attribute) match in the graph --------
        same_key = self._store.get_by_key(candidate.entity, candidate.attribute)
        if same_key:
            return self._decide_against_same_key(candidate, same_key)

        # --- Signal 2: semantic near-duplicate for the SAME entity ---------
        # No fact shares this exact key, but the user may have expressed the
        # same idea with a different attribute wording. Look for a very similar
        # existing fact about the same entity.
        semantic_match, sim = self._find_semantic_duplicate(candidate)
        if semantic_match is not None:
            reason = (
                f"Semantically equivalent to existing fact "
                f"'{semantic_match.text()}' (sim={sim:.2f}); merged to avoid duplication."
            )
            # Keep whichever has higher confidence; otherwise keep the existing.
            if candidate.confidence > semantic_match.confidence + 1e-9:
                self._store.update_fact(semantic_match.fact_id, candidate)
            decision = ConsolidationDecision(
                action=ConsolidationAction.MERGED,
                candidate=candidate,
                related_fact_id=semantic_match.fact_id,
                similarity=sim,
                reason=reason,
            )
            logger.info("Consolidation: MERGED %s (sim=%.2f)", candidate.text(), sim)
            return decision

        # --- Nothing related: brand-new fact -------------------------------
        self._store.add_fact(candidate)
        decision = ConsolidationDecision(
            action=ConsolidationAction.ADDED,
            candidate=candidate,
            reason="No existing fact matched by key or semantics; added as new.",
        )
        logger.info("Consolidation: ADDED %s", candidate.text())
        return decision

    # ------------------------------------------------------------------ #
    # Decision helpers
    # ------------------------------------------------------------------ #
    def _decide_against_same_key(
        self, candidate: Fact, existing_facts: list[Fact]
    ) -> ConsolidationDecision:
        """Handle the case where facts with the same (entity, attribute) exist.

        Same key means "same slot" (e.g. user.lives_in). Compare values:
          * equivalent value  -> REJECT (pure duplicate) unless higher conf.
          * different value    -> UPDATE (the fact changed).
        We compare against the best-matching existing fact for the slot.
        """
        # Pick the existing fact whose value is most similar to the candidate,
        # so "Bengaluru" updates the "Bangalore" entry rather than adding noise.
        best = max(
            existing_facts,
            key=lambda f: _value_similarity(f.value, candidate.value),
        )
        val_sim = _value_similarity(best.value, candidate.value)

        if _values_equivalent(best.value, candidate.value):
            # Same slot, same value -> duplicate.
            if candidate.confidence > best.confidence + 1e-9:
                # Higher-confidence restatement: refresh confidence via update.
                self._store.update_fact(best.fact_id, candidate)
                action, reason = (
                    ConsolidationAction.UPDATED,
                    f"Same value as '{best.text()}' but higher confidence; refreshed.",
                )
            else:
                action, reason = (
                    ConsolidationAction.REJECTED,
                    f"Duplicate of existing fact '{best.text()}'; dropped.",
                )
                logger.info("Consolidation: REJECTED %s", candidate.text())
                return ConsolidationDecision(
                    action=action,
                    candidate=candidate,
                    related_fact_id=best.fact_id,
                    similarity=val_sim,
                    reason=reason,
                )
        else:
            # Same slot, different value -> the fact evolved / was corrected.
            self._store.update_fact(best.fact_id, candidate)
            action, reason = (
                ConsolidationAction.UPDATED,
                f"Updated '{best.value}' -> '{candidate.value}' for "
                f"{candidate.entity}.{candidate.attribute} (conflict resolved).",
            )

        logger.info("Consolidation: %s %s", action.value.upper(), candidate.text())
        return ConsolidationDecision(
            action=action,
            candidate=candidate,
            related_fact_id=best.fact_id,
            similarity=val_sim,
            reason=reason,
        )

    def _find_semantic_duplicate(self, candidate: Fact) -> tuple[Fact | None, float]:
        """Return the most similar existing fact for the same entity, if any.

        Only facts about the SAME entity qualify as duplicates — a high textual
        similarity between facts about different entities is not a duplicate.
        """
        results = self._store.search_similar(candidate.text(), top_k=5)
        for scored in results:
            other = scored.fact
            if other.fact_id == candidate.fact_id:
                continue
            if other.entity.strip().lower() != candidate.entity.strip().lower():
                continue
            if scored.score >= self._merge_threshold:
                return other, scored.score
        return None, 0.0


# --------------------------------------------------------------------------- #
# Value comparison helpers
# --------------------------------------------------------------------------- #
def _normalize_value(value: str) -> str:
    """Lowercase, strip punctuation/whitespace for lenient value comparison."""
    return re.sub(r"[^\w\s]", "", value.strip().lower()).strip()


def _values_equivalent(a: str, b: str) -> bool:
    """True if two values mean the same thing for exact-key comparison.

    Uses normalized exact match plus a high token-overlap ratio so trivial
    variations ("New York" vs "new york.") count as equal, while genuinely
    different values ("Delhi" vs "Bangalore") do not.
    """
    na, nb = _normalize_value(a), _normalize_value(b)
    if na == nb:
        return True
    return _value_similarity(a, b) >= 0.9


def _value_similarity(a: str, b: str) -> float:
    """Cheap Jaccard token similarity in [0, 1] (no embeddings needed).

    Used to pick the best existing fact within a slot and to judge value
    equivalence. Semantic (embedding) similarity is handled separately by the
    vector store for the cross-attribute duplicate check.
    """
    ta = set(_normalize_value(a).split())
    tb = set(_normalize_value(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0
