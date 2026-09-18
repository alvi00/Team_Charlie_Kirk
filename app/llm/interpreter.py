"""Operator-note interpretation.

P0 STUB. Every note is returned as ``no_op`` so the API contract is testable end
to end before the model is wired in. The real Groq-backed interpreter (system
prompt, JSON mode, repair retry, fallback model, cache) replaces
``interpret_notes`` in P2; the signature below is the contract it must keep.

The LLM is mandatory in this path - see PROJECT.md section 1, rule 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

NO_OP_EXPLANATION = "This note does not affect today's 24-hour energy schedule."


@dataclass
class Interpretation:
    """What the interpretation layer hands to the guardrails and the optimizer."""

    #: one raw entry per note, in note_index order
    entries: list[dict[str, Any]]
    #: "stub" in P0, later "llm" | "llm_repair" | "deterministic_fallback"
    source: str = "stub"


def interpret_notes(notes: Sequence[str], battery: Any) -> Interpretation:
    """Return one interpretation entry per note, in ``note_index`` order.

    P0: always ``no_op``. This keeps the response schema-valid and the optimizer
    unconstrained, which is exactly what makes the contract testable now.
    """
    entries = [
        {
            "note_index": index,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": NO_OP_EXPLANATION,
        }
        for index, _ in enumerate(notes)
    ]
    return Interpretation(entries=entries, source="stub")
