"""Groq chat client over the OpenAI-compatible endpoint (PROJECT.md 7.1, 7.6).

``httpx`` directly rather than an SDK: fewer dependencies and exact control over
the per-call timeout, which is what protects p95.

Secret hygiene (graded): the API key lives only in the Authorization header. No
code path logs it, and provider error bodies are truncated and scrubbed before
they reach a log line.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from typing import Any, Sequence

import httpx

from ..config import Settings

log = logging.getLogger(__name__)

#: Three notes serialise to roughly 250 completion tokens; 700 leaves headroom
#: without letting a runaway response eat the per-minute token budget.
MAX_TOKENS = 700
TEMPERATURE = 0.0
#: gpt-oss models emit reasoning tokens. This task needs none - the rules are all
#: in the prompt - and "low" roughly halves the completion cost.
REASONING_EFFORT = "low"
#: PROJECT.md 7.6 - hash(sorted(notes) + capacity) -> interpretation, LRU 512
CACHE_SIZE = 512

_SECRET_RE = re.compile(r"(gsk_[A-Za-z0-9]{4})[A-Za-z0-9_\-]{8,}", re.IGNORECASE)


class LLMError(RuntimeError):
    """Any provider-side failure. Never carries the key or a raw provider body."""

    def __init__(self, message: str, *, rate_limited: bool = False) -> None:
        super().__init__(message)
        #: a 429 means retrying the same model immediately will fail the same way
        self.rate_limited = rate_limited


def scrub(text: str, limit: int = 300) -> str:
    """Redact anything key-shaped and truncate, for safe server-side logging."""
    cleaned = _SECRET_RE.sub(r"\1***", text or "")
    return cleaned[:limit]


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

    def __len__(self) -> int:  # pragma: no cover - diagnostics only
        with self._lock:
            return len(self._data)


class GroqClient:
    """One pooled HTTP client, JSON mode, hard per-call timeout."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.groq_base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.llm_timeout_seconds),
            headers={"Content-Type": "application/json"},
        )

    @property
    def configured(self) -> bool:
        return self._settings.llm_configured

    def close(self) -> None:
        self._client.close()

    def chat_json(self, messages: Sequence[dict[str, str]], model: str) -> dict[str, Any]:
        """One chat completion in JSON mode. Raises ``LLMError`` on any failure."""
        if not self.configured:
            raise LLMError("no api key configured")

        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "reasoning_effort": REASONING_EFFORT,
        }
        try:
            response = self._client.post(
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._settings.groq_api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise LLMError(f"timeout after {self._settings.llm_timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"transport error: {type(exc).__name__}") from exc

        if response.status_code != 200:
            # Truncated and scrubbed: a provider body must never reach a log raw.
            raise LLMError(
                f"provider status {response.status_code}: {scrub(response.text, 200)}",
                rate_limited=response.status_code == 429,
            )

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("malformed provider envelope") from exc

        return {"content": content or "", "model": model}


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
    import hashlib

    digest = hashlib.sha256()
    for note in sorted(n.strip() for n in notes):
        digest.update(note.encode("utf-8", "ignore"))
        digest.update(b"\x00")
    digest.update(str(float(capacity_kwh)).encode("ascii"))
    return digest.hexdigest()
