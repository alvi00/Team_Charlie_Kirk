"""Operator-note interpretation: the mandatory LLM path (PROJECT.md section 7).

One call handles all 1-3 notes. The escalation ladder from section 7.5, widened
to cross a provider boundary so a single vendor outage cannot take interpretation
down:

    primary model  (N attempts, same provider)
      -> one repair call carrying the validator's complaint
      -> secondary model at the same provider
      -> failover provider, if one is configured
      -> rule-based deterministic interpreter

A wall-clock budget caps the whole stage so a hanging provider can never push a
request past the judge's 30s ceiling. Every branch ends in a valid 200 response.

The LLM is the primary interpreter - the deterministic interpreter is a safety
net for a provider outage, never the main path.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from ..config import Settings, get_settings
from ..guardrails.fallback import interpret_rule_based
from ..guardrails.validate import normalize_interpretation
from .client import (
    ChatClient,
    Endpoint,
    LLMError,
    LRUCache,
    cache_key,
    parse_json_object,
    scrub,
)
from .prompt import RESPONSE_SCHEMA, build_messages, build_repair_messages

log = logging.getLogger(__name__)


@dataclass
class Interpretation:
    """What the interpretation layer hands to the optimizer."""

    #: one normalized entry per note, in note_index order
    entries: list[dict[str, Any]]
    #: "llm" | "llm_repair" | "llm_fallback_model" | "llm_failover_provider"
    #: | "deterministic_fallback" | "cache"
    source: str
    model: str | None = None
    provider: str | None = None
    #: guardrail corrections, for server-side logging only
    repairs: list[str] = field(default_factory=list)


_cache = LRUCache()
_client: ChatClient | None = None


def get_client(settings: Settings | None = None) -> ChatClient:
    """Process-wide pooled client, built lazily so imports stay cheap."""
    global _client
    if _client is None:
        settings = settings or get_settings()
        _client = ChatClient(
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=settings.llm_reasoning_effort,
        )
    return _client


def reset_client() -> None:
    """Drop the pooled client (shutdown, and tests that swap the settings)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def warm_up(settings: Settings | None = None) -> None:
    """Open the provider connection and learn the model's parameter profile.

    The first call to any worker otherwise pays DNS, the TLS handshake and
    capability discovery, which is what puts the slowest requests near the p95
    budget. Runs in a background thread at startup so readiness is never blocked,
    and swallows every error - a failed warm-up must not affect serving.

    This is deliberately NOT part of ``/health``: the readiness probe stays
    synchronous and provider-independent (PROJECT.md 5.1).
    """
    settings = settings or get_settings()
    if not settings.llm_configured:
        return

    def _run() -> None:
        try:
            client = get_client(settings)
            endpoint = Endpoint(
                "openai", settings.openai_base_url, settings.openai_api_key
            )
            if not endpoint.configured:
                return
            client.chat_json(
                [
                    {"role": "system", "content": "Reply with JSON only."},
                    {"role": "user", "content": 'Reply exactly {"ok":true}'},
                ],
                settings.openai_model,
                endpoint,
                None,
            )
            log.info(
                "llm warm-up complete: %s accepts %s",
                settings.openai_model,
                client.profile_for(endpoint, settings.openai_model).describe(),
            )
        except Exception as exc:  # never let warm-up affect serving
            log.warning("llm warm-up skipped: %s", scrub(str(exc), 160))

    threading.Thread(target=_run, name="llm-warmup", daemon=True).start()


def _attempts(settings: Settings) -> Iterator[tuple[Endpoint, str, str]]:
    """The ordered (endpoint, model, source-label) ladder."""
    primary = Endpoint("openai", settings.openai_base_url, settings.openai_api_key)
    if primary.configured:
        for _ in range(max(1, settings.llm_max_retries)):
            yield primary, settings.openai_model, "llm"
        if settings.openai_fallback_model:
            yield primary, settings.openai_fallback_model, "llm_fallback_model"

    if settings.failover_configured:
        failover = Endpoint(
            "failover",
            settings.fallback_provider_base_url,
            settings.fallback_provider_api_key,
        )
        yield failover, settings.fallback_provider_model, "llm_failover_provider"


def _shape_error(payload: dict[str, Any], note_count: int) -> str | None:
    """Cheap structural check on the raw model payload, before normalization.

    Only used to decide whether a repair call is worth making - the guardrails do
    the real work and can recover most of what this flags.
    """
    directives = payload.get("directives")
    if not isinstance(directives, list):
        return 'the response had no "directives" array'
    if len(directives) != note_count:
        return f"expected exactly {note_count} entries, got {len(directives)}"
    seen: set[int] = set()
    for entry in directives:
        if not isinstance(entry, dict):
            return "every entry must be a JSON object"
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
            provider=cached.provider,
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
    repaired = False
    exhausted: set[tuple[str, str]] = set()

    # Wall-clock budget for the whole stage. Each individual call has its own
    # timeout, but several in series could otherwise push a single request past
    # the judge's 30s ceiling.
    deadline = time.monotonic() + settings.llm_total_budget_seconds

    def budget_left() -> bool:
        if time.monotonic() < deadline:
            return True
        log.warning("interpretation budget spent; falling back deterministically")
        return False

    for endpoint, model, source in _attempts(settings):
        marker = (endpoint.base_url, model)
        if marker in exhausted:
            continue
        if not budget_left():
            break
        try:
            last_content = client.chat_json(messages, model, endpoint, RESPONSE_SCHEMA)
            payload = parse_json_object(last_content)
            shape_error = _shape_error(payload, len(notes))
            if shape_error:
                raise LLMError(shape_error)
            return _finalize(payload, notes, capacity, source, model, endpoint.label)

        except LLMError as exc:
            last_error = scrub(str(exc))
            log.warning("llm attempt failed (%s/%s): %s", endpoint.label, model, last_error)

            if exc.rate_limited:
                # Retrying the same model inside the same window just burns
                # latency; move straight on to the next rung.
                exhausted.add(marker)
                continue

            # one repair call, carrying the complaint back to the model
            if not repaired and last_content and budget_left():
                repaired = True
                try:
                    repair_messages = build_repair_messages(
                        notes, battery, last_content, last_error
                    )
                    content = client.chat_json(
                        repair_messages, model, endpoint, RESPONSE_SCHEMA
                    )
                    payload = parse_json_object(content)
                    shape_error = _shape_error(payload, len(notes))
                    if shape_error:
                        raise LLMError(shape_error)
                    return _finalize(
                        payload, notes, capacity, "llm_repair", model, endpoint.label
                    )
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
        provider=None,
        repairs=normalized.repairs,
    )


def _finalize(
    payload: dict[str, Any],
    notes: Sequence[str],
    capacity: float,
    source: str,
    model: str,
    provider: str,
) -> Interpretation:
    """Run the raw model output through the deterministic guardrails."""
    normalized = normalize_interpretation(payload.get("directives"), notes, capacity)
    if normalized.repairs:
        log.info(
            "guardrails repaired %d item(s): %s",
            len(normalized.repairs),
            normalized.repairs,
        )
    return Interpretation(
        entries=normalized.entries,
        source=source,
        model=model,
        provider=provider,
        repairs=normalized.repairs,
    )
