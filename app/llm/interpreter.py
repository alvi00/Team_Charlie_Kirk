"""Operator-note interpretation: the mandatory LLM path (PROJECT.md 7).

One call handles all 1-3 notes. The escalation ladder from section 7.5:

    primary model (N attempts)
      -> one repair call carrying the validator's complaint
      -> fallback model, one attempt
      -> rule-based deterministic interpreter

Every branch ends in a valid 200 response. The LLM is the primary interpreter -
the deterministic interpreter is a safety net for a provider outage, never the
main path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import Settings, get_settings
from ..guardrails.fallback import interpret_rule_based
from ..guardrails.validate import normalize_interpretation
from .client import GroqClient, LLMError, LRUCache, cache_key, parse_json_object, scrub
from .prompt import build_messages, build_repair_messages

log = logging.getLogger(__name__)


@dataclass
class Interpretation:
    """What the interpretation layer hands to the optimizer."""

    #: one normalized entry per note, in note_index order
    entries: list[dict[str, Any]]
    #: "llm" | "llm_repair" | "llm_fallback_model" | "deterministic_fallback" | "cache"
    source: str
    model: str | None = None
    #: guardrail corrections, for server-side logging only
    repairs: list[str] = field(default_factory=list)


_cache = LRUCache()
_client: GroqClient | None = None


def get_client(settings: Settings | None = None) -> GroqClient:
    """Process-wide pooled client, built lazily so imports stay cheap."""
    global _client
    if _client is None:
        _client = GroqClient(settings or get_settings())
    return _client


def reset_client() -> None:
    """Drop the pooled client (shutdown, and tests that swap the settings)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def _shape_error(payload: dict[str, Any], note_count: int) -> str | None:
    """Cheap structural check on the raw model payload, before normalization.

    Only used to decide whether a repair call is worth making - the guardrails do
    the real work and can recover most of what this flags.
    """
    directives = payload.get("directives")
    if not isinstance(directives, list):
        return 'missing a "directives" array'
    if len(directives) != note_count:
        return f"expected {note_count} entries, got {len(directives)}"
    seen = set()
    for entry in directives:
        if not isinstance(entry, dict):
            return "every entry must be an object"
        index = entry.get("note_index")
        if not isinstance(index, int) or not 0 <= index < note_count:
            return f"invalid note_index {index!r}"
        if index in seen:
            return f"duplicate note_index {index}"
        seen.add(index)
        if not entry.get("directive_type"):
            return f"entry {index} has no directive_type"
    return None


def interpret_notes(
    notes: Sequence[str], battery: Any, settings: Settings | None = None
) -> Interpretation:
    """Return one normalized interpretation entry per note, in note_index order."""
    settings = settings or get_settings()
    capacity = float(battery.capacity_kwh)
    key = cache_key(notes, capacity)

    cached = _cache.get(key)
    if cached is not None:
        return Interpretation(
            entries=[dict(entry) for entry in cached.entries],
            source="cache",
            model=cached.model,
        )

    result = _interpret_uncached(notes, battery, capacity, settings)
    if result.source != "deterministic_fallback":
        # Never cache a degraded result: the next request should retry the model.
        _cache.put(key, result)
    return result


def _interpret_uncached(
    notes: Sequence[str], battery: Any, capacity: float, settings: Settings
) -> Interpretation:
    client = get_client(settings)
    messages = build_messages(notes, battery)
    last_error = "not attempted"
    last_content = ""

    if client.configured:
        attempts: list[tuple[str, str]] = [
            *((settings.groq_model, "llm") for _ in range(max(1, settings.llm_max_retries))),
            (settings.groq_fallback_model, "llm_fallback_model"),
        ]
        repaired = False
        exhausted: set[str] = set()

        for model, source in attempts:
            if model in exhausted:
                continue
            try:
                reply = client.chat_json(messages, model)
                last_content = reply["content"]
                payload = parse_json_object(last_content)
                shape_error = _shape_error(payload, len(notes))
                if shape_error:
                    raise LLMError(shape_error)
                return _finalize(payload, notes, capacity, source, model)

            except LLMError as exc:
                last_error = scrub(str(exc))
                log.warning("llm attempt failed (model=%s): %s", model, last_error)

                if exc.rate_limited:
                    # Retrying the same model inside the same minute just burns
                    # latency; move straight on to the next one.
                    exhausted.add(model)
                    continue

                # one repair call, carrying the complaint back to the model
                if not repaired and last_content:
                    repaired = True
                    try:
                        reply = client.chat_json(
                            build_repair_messages(notes, battery, last_content, last_error),
                            model,
                        )
                        payload = parse_json_object(reply["content"])
                        shape_error = _shape_error(payload, len(notes))
                        if shape_error:
                            raise LLMError(shape_error)
                        return _finalize(payload, notes, capacity, "llm_repair", model)
                    except LLMError as repair_exc:
                        last_error = scrub(str(repair_exc))
                        log.warning("llm repair call failed: %s", last_error)

    log.error("all llm attempts failed (%s); using deterministic fallback", last_error)
    entries = interpret_rule_based(notes, capacity)
    normalized = normalize_interpretation(entries, notes, capacity)
    return Interpretation(
        entries=normalized.entries,
        source="deterministic_fallback",
        model=None,
        repairs=normalized.repairs,
    )


def _finalize(
    payload: dict[str, Any],
    notes: Sequence[str],
    capacity: float,
    source: str,
    model: str,
) -> Interpretation:
    """Run the raw model output through the deterministic guardrails."""
    normalized = normalize_interpretation(payload.get("directives"), notes, capacity)
    if normalized.repairs:
        log.info("guardrails repaired %d item(s): %s", len(normalized.repairs), normalized.repairs)
    return Interpretation(
        entries=normalized.entries,
        source=source,
        model=model,
        repairs=normalized.repairs,
    )
