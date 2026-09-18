"""Verify the configured LLM model before you rely on it.

PROJECT.md section 18 calls a wrong model id "a silent killer": the request 404s
and interpretation quietly degrades to the deterministic fallback. Run this after
setting the key, and again after changing any model id.

    python tools/verify_model.py

It checks, for the primary provider and (if configured) the failover provider:

  1. the key authenticates
  2. the model id exists and answers
  3. which request parameters the model actually accepts
  4. a real operator note is interpreted correctly end to end

Nothing here prints the API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.llm.client import ChatClient, Endpoint, LLMError, parse_json_object  # noqa: E402
from app.llm.prompt import RESPONSE_SCHEMA, build_messages  # noqa: E402
from app.schemas import BatteryInput  # noqa: E402

PROBE_NOTE = "Do not charge the battery from 2 AM until 5 AM for maintenance."
PROBE_BATTERY = BatteryInput(
    capacity_kwh=200,
    initial_energy_kwh=100,
    minimum_energy_kwh=40,
    max_charge_kwh_per_hour=50,
    max_discharge_kwh_per_hour=50,
)
EXPECTED_TYPE = "no_charge_window"
EXPECTED_HOURS = [2, 3, 4]


def check(client: ChatClient, endpoint: Endpoint, model: str, label: str) -> bool:
    print(f"\n--- {label}: {model} @ {endpoint.base_url}")
    if not endpoint.configured:
        print("    SKIP  no API key configured")
        return True

    messages = build_messages([PROBE_NOTE], PROBE_BATTERY)
    try:
        content = client.chat_json(messages, model, endpoint, RESPONSE_SCHEMA)
    except LLMError as exc:
        text = str(exc)
        print(f"    FAIL  {text}")
        if "404" in text or "does not exist" in text or "model_not_found" in text:
            print("          -> the model id is wrong or not available to this account")
        elif "401" in text or "invalid_api_key" in text:
            print("          -> the API key is missing, wrong, or lacks access")
        elif "429" in text:
            print("          -> rate limited / no quota on this account")
        return False

    profile = client.profile_for(endpoint, model)
    print(f"    reachable, accepted parameters: {profile.describe()}")

    try:
        entries = parse_json_object(content).get("directives") or []
        first = entries[0]
        got_type = first.get("directive_type")
        got_hours = (first.get("structured_adjustment") or {}).get("hours")
    except Exception:
        print(f"    FAIL  could not parse the reply: {content[:160]}")
        return False

    ok = got_type == EXPECTED_TYPE and list(got_hours or []) == EXPECTED_HOURS
    print(f"    probe note -> {got_type} hours={got_hours}")
    print(f"    {'PASS  interpretation correct' if ok else 'WARN  unexpected interpretation'}")
    if not ok:
        print(f"          expected {EXPECTED_TYPE} hours={EXPECTED_HOURS}")
    return ok


def main() -> int:
    settings = get_settings()
    print("GridWise model verification")
    print(f"  primary model : {settings.openai_model}")
    print(f"  same-provider fallback: {settings.openai_fallback_model}")
    print(f"  failover provider     : "
          f"{settings.fallback_provider_model if settings.failover_configured else '(disabled)'}")

    if not settings.llm_configured:
        print("\nNo API key configured. Set OPENAI_API_KEY in .env and re-run.")
        return 1

    client = ChatClient(
        timeout_seconds=max(settings.llm_timeout_seconds, 20.0),
        reasoning_effort=settings.llm_reasoning_effort,
    )
    primary = Endpoint("openai", settings.openai_base_url, settings.openai_api_key)
    failover = Endpoint(
        "failover",
        settings.fallback_provider_base_url,
        settings.fallback_provider_api_key,
    )

    results = [
        check(client, primary, settings.openai_model, "primary"),
        check(client, primary, settings.openai_fallback_model, "same-provider fallback"),
        check(client, failover, settings.fallback_provider_model, "failover provider"),
    ]
    client.close()

    print()
    if all(results):
        print("All configured models verified.")
        return 0
    print("At least one model failed. Fix the ids/keys above before deploying -")
    print("a bad model id degrades silently to the deterministic interpreter.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
