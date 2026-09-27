"""Synthetic multi-session dataset for evaluation.

LoCoMo and similar long-term-memory benchmarks aren't reliably downloadable
without friction, so we ship a small, self-contained dataset that deliberately
exercises the behaviours AgentMemory targets:

  * **Updates / conflicts** — the user's city changes across sessions; the
    correct answer is the *latest* value, not the first.
  * **Corrections** — the user corrects an earlier statement.
  * **Preferences & attributes** — stable facts that should be recalled.

Each item is a multi-session conversation (list of user utterances, in order)
plus QA probes asked *after* all sessions, whose expected answers depend on the
consolidated final state. This is exactly where naive append-only memory
degrades (it stores stale + new values) while consolidation stays correct.

The loader is structured so a real LoCoMo JSON file could be dropped in later
behind the same ``EvalCase`` shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QAProbe:
    """A question asked after the conversation, with its expected answer.

    ``stale`` lists values that were true *earlier* but have since been
    superseded (e.g. the old city). A correct answer must contain ``expected``
    and must NOT surface a ``stale`` value — surfacing a stale value is the
    exact failure mode of append-only memory that consolidation prevents.
    """

    question: str
    expected: str
    stale: list[str] = field(default_factory=list)


@dataclass
class EvalCase:
    """One user's multi-session history plus the questions to probe afterwards."""

    case_id: str
    # Ordered user utterances across sessions (session boundaries don't matter
    # for scoring; order does, because later facts supersede earlier ones).
    utterances: list[str]
    probes: list[QAProbe] = field(default_factory=list)


def load_dataset() -> list[EvalCase]:
    """Return the built-in synthetic evaluation cases."""
    return [
        EvalCase(
            case_id="relocation",
            utterances=[
                # Session 1
                "Hi, my name is Rohit and I live in Delhi.",
                "I work as a software engineer.",
                # Session 2 (later): the move — a conflict/update.
                "Big news: I just moved to Bangalore for a new job.",
                "I now work as a machine learning engineer.",
            ],
            probes=[
                QAProbe("Where does the user currently live?", "Bangalore", stale=["Delhi"]),
                QAProbe(
                    "What is the user's current job?",
                    "machine learning engineer",
                    stale=["software engineer"],
                ),
                QAProbe("What is the user's name?", "Rohit"),
            ],
        ),
        EvalCase(
            case_id="preferences",
            utterances=[
                "I love hiking on weekends.",
                "My favorite food is biryani.",
                # Correction: dietary change supersedes the earlier favorite.
                "Actually I became vegetarian, so my favorite food is now paneer.",
            ],
            probes=[
                QAProbe("What is the user's favorite food now?", "paneer", stale=["biryani"]),
                QAProbe("What hobby does the user enjoy?", "hiking"),
            ],
        ),
        EvalCase(
            case_id="pet_and_car",
            utterances=[
                "I have a dog named Bruno.",
                "I drive a Honda.",
                # Update: sold the Honda, bought a Tesla.
                "I sold my Honda and now I drive a Tesla.",
            ],
            probes=[
                QAProbe("What car does the user drive now?", "Tesla", stale=["Honda"]),
                QAProbe("What is the user's pet?", "dog"),
            ],
        ),
    ]


def total_probes(cases: list[EvalCase]) -> int:
    return sum(len(c.probes) for c in cases)
