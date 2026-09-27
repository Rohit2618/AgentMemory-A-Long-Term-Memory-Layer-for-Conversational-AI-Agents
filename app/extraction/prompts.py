"""Prompt construction for fact extraction.

The prompt does double duty:
  * For a real LLM (Groq), the instructions + few-shot example steer it to emit
    a clean JSON array of facts.
  * For the mock provider, the EXTRACTION_MARKER lets it recognize the task and
    run its regex-based extractor on the conversation text that follows.
"""

from __future__ import annotations

from app.core.models import ConversationTurn
from app.llm.mock_provider import EXTRACTION_MARKER

SYSTEM = (
    "You are a memory-extraction component for a conversational AI. "
    "Your job is to read a conversation turn and extract durable facts worth "
    "remembering long-term: user preferences, decisions, personal attributes, "
    "entities, relationships, and corrections. Ignore small talk and transient "
    "context. Return ONLY facts that would still matter in a future session."
)

INSTRUCTIONS = """\
Extract facts as a JSON array. Each fact is an object with exactly these keys:
  - "entity":     who/what the fact is about (use "user" for the person talking)
  - "attribute":  a short snake_case relation (e.g. lives_in, works_as, likes,
                  dislikes, favorite_food, name, age, owns)
  - "value":      the value of the attribute (a short phrase)
  - "confidence": a number from 0.0 to 1.0

Rules:
  - Output ONLY the JSON array. No prose, no markdown fences.
  - One fact per distinct piece of information.
  - Normalize attributes to snake_case.
  - If nothing is worth remembering, return [].

Example conversation:
  User: My name is Rohit and I moved to Bangalore. I love hiking.
Example output:
  [{"entity":"user","attribute":"name","value":"Rohit","confidence":0.95},
   {"entity":"user","attribute":"lives_in","value":"Bangalore","confidence":0.9},
   {"entity":"user","attribute":"likes","value":"hiking","confidence":0.85}]
"""


def build_extraction_prompt(turn: ConversationTurn) -> str:
    """Assemble the user-role prompt for extracting facts from one turn.

    The conversation text is placed AFTER the EXTRACTION_MARKER so the mock
    provider can locate and parse it; a real LLM simply treats the marker as
    part of the instructions and returns JSON.
    """
    convo_lines = [f"User: {turn.user_message}"]
    if turn.agent_response:
        convo_lines.append(f"Agent: {turn.agent_response}")
    conversation = "\n".join(convo_lines)

    return (
        f"{INSTRUCTIONS}\n\n"
        f"Now extract facts from this conversation.\n"
        f"{EXTRACTION_MARKER}\n"
        f"{conversation}\n"
    )
