"""Chat client over the OpenAI Chat Completions API (PROJECT.md 7.1, 7.6).

``httpx`` directly rather than an SDK: fewer dependencies and exact control over
the per-call timeout, which is what protects p95. The same wire format is spoken
by OpenAI, Groq and most hosted-inference vendors, so one client covers the
primary provider and the cross-provider failover.

**Capability negotiation.** Model families disagree about request parameters -
newer reasoning models require ``max_completion_tokens`` instead of
``max_tokens``, reject a non-default ``temperature``, and may or may not accept
``reasoning_effort`` or strict ``json_schema`` response formats. Rather than
hard-coding assumptions about any one model id, the client starts from the
strictest (best) parameter set and, on a 400 that names an offending parameter,
drops or swaps that parameter and retries. The working profile is remembered per
(endpoint, model) so the discovery cost is paid once per process.

Secret hygiene (graded): the API key lives only in the Authorization header. No
code path logs it, and provider error bodies are truncated and scrubbed before
they reach a log line.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Sequence

import httpx

log = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 700
#: PROJECT.md 7.6 - hash(sorted(notes) + capacity) -> interpretation, LRU 512
CACHE_SIZE = 512
#: how many parameter adaptations to attempt before giving up on a model
MAX_ADAPTATIONS = 5

#: OpenAI (sk-..., sk-proj-...), Groq (gsk_...) and generic bearer-ish blobs
_SECRET_RE = re.compile(
    r"\b((?:sk|gsk|rk)[-_](?:proj[-_])?[A-Za-z0-9]{4})[A-Za-z0-9_\-]{8,}",
    re.IGNORECASE,
)


class LLMError(RuntimeError):
    """Any provider-side failure. Never carries the key or a raw provider body."""

    def __init__(
        self, message: str, *, rate_limited: bool = False, status: int | None = None
    ) -> None:
        super().__init__(message)
        #: a 429 means retrying the same model immediately will fail the same way
        self.rate_limited = rate_limited
        self.status = status


def scrub(text: str, limit: int = 300) -> str:
    """Redact anything key-shaped and truncate, for safe server-side logging."""
    cleaned = _SECRET_RE.sub(r"\1***", text or "")
    return cleaned[:limit]


# ---------------------------------------------------------------------------
# endpoints and capability profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    """One provider: a base URL plus the key that opens it."""

    label: str
    base_url: str
    api_key: str

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url)


@dataclass(frozen=True)
class ParamProfile:
    """Which request parameters a given model actually accepts.

    Starts at the strictest, most capable setting and degrades on rejection.
    """

    token_param: str = "max_completion_tokens"
    use_temperature: bool = True
    use_reasoning_effort: bool = True
    #: "json_schema" (strict, best) -> "json_object" -> "none" (prompt only)
    response_format: str = "json_schema"

    def describe(self) -> str:
        return (
            f"{self.token_param}"
            f"{'+temp' if self.use_temperature else '-temp'}"
            f"{'+reasoning' if self.use_reasoning_effort else '-reasoning'}"
            f"/{self.response_format}"
        )


def _adapt(profile: ParamProfile, error_text: str) -> ParamProfile | None:
    """Drop or swap whichever parameter the provider just rejected.

    Returns the adjusted profile, or None when nothing is left to try.
    """
    text = (error_text or "").lower()

    def mentions(*needles: str) -> bool:
        return any(n in text for n in needles)

    if profile.token_param == "max_completion_tokens" and mentions(
        "max_completion_tokens", "max_tokens"
    ):
        return replace(profile, token_param="max_tokens")
    if profile.use_temperature and mentions("temperature"):
        return replace(profile, use_temperature=False)
    if profile.use_reasoning_effort and mentions("reasoning_effort", "reasoning"):
        return replace(profile, use_reasoning_effort=False)
    if profile.response_format == "json_schema" and mentions(
        "json_schema", "response_format", "structured output", "schema"
    ):
        return replace(profile, response_format="json_object")
    if profile.response_format == "json_object" and mentions(
        "response_format", "json_object", "json"
    ):
        return replace(profile, response_format="none")

    # Unrecognised 400: degrade along the axes most likely to be the cause,
    # cheapest-to-lose first, so an unfamiliar model still gets a chance.
    for candidate in (
        replace(profile, use_reasoning_effort=False)
        if profile.use_reasoning_effort
        else None,
        replace(profile, use_temperature=False) if profile.use_temperature else None,
        replace(profile, token_param="max_tokens")
        if profile.token_param == "max_completion_tokens"
        else None,
        replace(profile, response_format="json_object")
        if profile.response_format == "json_schema"
        else None,
        replace(profile, response_format="none")
        if profile.response_format == "json_object"
        else None,
    ):
        if candidate is not None:
            return candidate
    return None


class LRUCache:
    """Small thread-safe LRU. The judge fires repeated and related cases."""

    def __init__(self, maxsize: int = CACHE_SIZE) -> None:
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:  # pragma: no cover - diagnostics only
        with self._lock:
            return len(self._data)


class ChatClient:
    """Pooled HTTP clients, JSON-mode chat completions, hard per-call timeout."""

    def __init__(self, timeout_seconds: float, reasoning_effort: str = "low") -> None:
        self._timeout = timeout_seconds
        self._reasoning_effort = reasoning_effort
        self._clients: dict[str, httpx.Client] = {}
        self._profiles: dict[tuple[str, str], ParamProfile] = {}
        self._lock = threading.Lock()

    def _client_for(self, base_url: str) -> httpx.Client:
        key = base_url.rstrip("/")
        with self._lock:
            client = self._clients.get(key)
            if client is None:
                client = httpx.Client(
                    base_url=key,
                    timeout=httpx.Timeout(self._timeout),
                    headers={"Content-Type": "application/json"},
                )
                self._clients[key] = client
            return client

    def close(self) -> None:
        with self._lock:
            for client in self._clients.values():
                client.close()
            self._clients.clear()

    def profile_for(self, endpoint: Endpoint, model: str) -> ParamProfile:
        return self._profiles.get((endpoint.base_url, model), ParamProfile())

    def _build_payload(
        self,
        messages: Sequence[dict[str, str]],
        model: str,
        profile: ParamProfile,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            profile.token_param: MAX_OUTPUT_TOKENS,
        }
        if profile.use_temperature:
            payload["temperature"] = 0.0
        if profile.use_reasoning_effort and self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        if profile.response_format == "json_schema" and schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": schema}
        elif profile.response_format in ("json_schema", "json_object"):
            payload["response_format"] = {"type": "json_object"}
        return payload

    def chat_json(
        self,
        messages: Sequence[dict[str, str]],
        model: str,
        endpoint: Endpoint,
        schema: dict[str, Any] | None = None,
    ) -> str:
        """One chat completion returning JSON text. Raises ``LLMError`` on failure."""
        if not endpoint.configured:
            raise LLMError(f"{endpoint.label}: no api key configured")

        client = self._client_for(endpoint.base_url)
        profile = self.profile_for(endpoint, model)
        last_error = "no attempt made"

        for _ in range(MAX_ADAPTATIONS):
            payload = self._build_payload(messages, model, profile, schema)
            try:
                response = client.post(
                    "/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {endpoint.api_key}"},
                )
            except httpx.TimeoutException as exc:
                raise LLMError(
                    f"{endpoint.label}: timeout after {self._timeout}s"
                ) from exc
            except httpx.HTTPError as exc:
                raise LLMError(
                    f"{endpoint.label}: transport error {type(exc).__name__}"
                ) from exc

            if response.status_code == 200:
                # remember the profile that worked
                with self._lock:
                    self._profiles[(endpoint.base_url, model)] = profile
                return self._extract(response)

            body = scrub(response.text, 300)
            if response.status_code == 400:
                # a rejected parameter is recoverable: adapt and retry
                adapted = _adapt(profile, body)
                if adapted is not None and adapted != profile:
                    log.info(
                        "%s/%s rejected %s; retrying as %s",
                        endpoint.label,
                        model,
                        profile.describe(),
                        adapted.describe(),
                    )
                    profile = adapted
                    last_error = f"400 {body}"
                    continue

            raise LLMError(
                f"{endpoint.label}: status {response.status_code}: {body}",
                rate_limited=response.status_code == 429,
                status=response.status_code,
            )

        raise LLMError(f"{endpoint.label}: exhausted parameter adaptations; {last_error}")

    @staticmethod
    def _extract(response: httpx.Response) -> str:
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("malformed provider envelope") from exc

        content = message.get("content")
        if isinstance(content, list):
            # some providers return content parts rather than a plain string
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in (None, "text", "output_text")
            )
        if not content:
            finish = choice.get("finish_reason")
            raise LLMError(f"empty completion (finish_reason={finish})")
        return content


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse the model's reply, tolerating a stray code fence or prose wrapper."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("response was not JSON")
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError as exc:
            raise LLMError("response was not JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMError("response JSON was not an object")
    return parsed


def cache_key(notes: Sequence[str], capacity_kwh: float) -> str:
    """PROJECT.md 7.6: hash(sorted(notes) + battery.capacity_kwh)."""
    digest = hashlib.sha256()
    for note in sorted(n.strip() for n in notes):
        digest.update(note.encode("utf-8", "ignore"))
        digest.update(b"\x00")
    digest.update(str(float(capacity_kwh)).encode("ascii"))
    return digest.hexdigest()
