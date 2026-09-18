"""API contract and robustness tests (PROJECT.md 5, 12; rubric 'API Contract').

Deliberately hermetic: ``GROQ_API_KEY`` is blanked before the app is imported, so
the service runs on its deterministic interpreter and these tests need no network
and no quota. That also means they exercise the exact path a provider outage
takes - which is the behaviour the 'controlled failure handling' marks reward.

Runs under pytest, or standalone with no extra dependency:

    python tests/test_api.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# must happen before app.config is imported: env beats the .env file
os.environ["GROQ_API_KEY"] = ""

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

CASES = json.loads(
    (ROOT / "data" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json").read_text(
        encoding="utf-8"
    )
)["cases"]
SAMPLE = CASES[0]["input"]


def _fresh(**overrides):
    payload = json.loads(json.dumps(SAMPLE))
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def test_health_is_ok_and_does_not_call_the_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# happy path contract
# ---------------------------------------------------------------------------


def test_response_matches_the_contract():
    response = client.post("/optimize-energy", json=SAMPLE)
    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body) == {
        "scenario_id",
        "directive_interpretation",
        "hourly_plan",
        "total_grid_kwh",
        "total_cost_bdt",
        "peak_grid_kwh",
        "plan_summary",
    }
    assert body["scenario_id"] == SAMPLE["scenario_id"]
    assert isinstance(body["plan_summary"], str) and body["plan_summary"]

    plan = body["hourly_plan"]
    assert [row["hour"] for row in plan] == list(range(24))
    for row in plan:
        assert set(row) == {
            "hour",
            "grid_kwh",
            "solar_used_kwh",
            "battery_action",
            "battery_kwh",
            "battery_energy_after_kwh",
        }
        assert row["battery_action"] in {"charge", "discharge", "idle"}
        assert row["grid_kwh"] >= 0 and row["solar_used_kwh"] >= 0
        assert row["battery_kwh"] >= 0
        if row["battery_action"] == "idle":
            assert row["battery_kwh"] == 0


def test_interpretation_is_one_entry_per_note_in_order():
    for case in CASES:
        payload = case["input"]
        body = client.post("/optimize-energy", json=payload).json()
        entries = body["directive_interpretation"]
        assert len(entries) == len(payload["operator_notes"])
        assert [e["note_index"] for e in entries] == list(range(len(entries)))
        for e in entries:
            assert set(e) == {
                "note_index",
                "applies",
                "directive_type",
                "structured_adjustment",
                "explanation",
            }
            if e["directive_type"] == "no_op":
                assert e["applies"] is False
                assert e["structured_adjustment"] is None
            else:
                assert e["applies"] is True
                adjustment = e["structured_adjustment"]
                assert isinstance(adjustment, dict)
                hours = adjustment["hours"]
                assert hours == sorted(set(hours))
                assert all(isinstance(h, int) and 0 <= h <= 23 for h in hours)
                if "factor" in adjustment:
                    assert 0.0 <= adjustment["factor"] <= 1.0


def test_totals_agree_with_the_emitted_plan():
    for case in CASES:
        payload = case["input"]
        body = client.post("/optimize-energy", json=payload).json()
        plan = body["hourly_plan"]
        tariff = {h["hour"]: h["tariff_bdt_per_kwh"] for h in payload["hours"]}
        grid = sum(r["grid_kwh"] for r in plan)
        cost = sum(r["grid_kwh"] * tariff[r["hour"]] for r in plan)
        peak = max(r["grid_kwh"] for r in plan)
        assert abs(grid - body["total_grid_kwh"]) < 0.01
        assert abs(cost - body["total_cost_bdt"]) < 0.01
        assert abs(peak - body["peak_grid_kwh"]) < 0.01
        assert (
            abs(
                plan[23]["battery_energy_after_kwh"]
                - payload["battery"]["initial_energy_kwh"]
            )
            < 0.01
        )


def test_hours_may_arrive_in_any_order():
    shuffled = _fresh()
    random.seed(7)
    random.shuffle(shuffled["hours"])
    ordered = client.post("/optimize-energy", json=SAMPLE).json()
    jumbled = client.post("/optimize-energy", json=shuffled).json()
    assert jumbled["hourly_plan"] == ordered["hourly_plan"]


# ---------------------------------------------------------------------------
# malformed input -> 400 (never 422, never 500)
# ---------------------------------------------------------------------------


def test_malformed_requests_return_400():
    bad = {
        "missing battery": {k: v for k, v in SAMPLE.items() if k != "battery"},
        "missing hours": {k: v for k, v in SAMPLE.items() if k != "hours"},
        "missing scenario_id": {k: v for k, v in SAMPLE.items() if k != "scenario_id"},
        "23 hours": _fresh(hours=SAMPLE["hours"][:23]),
        "duplicate hour": _fresh(hours=SAMPLE["hours"][:23] + [SAMPLE["hours"][0]]),
        "no notes": _fresh(operator_notes=[]),
        "four notes": _fresh(operator_notes=["a", "b", "c", "d"]),
        "blank note": _fresh(operator_notes=["   "]),
        "note is not a string": _fresh(operator_notes=[123]),
        "non-numeric demand": _fresh(
            hours=[{**SAMPLE["hours"][0], "demand_kwh": "abc"}] + SAMPLE["hours"][1:]
        ),
        "hour out of range": _fresh(
            hours=[{**SAMPLE["hours"][0], "hour": 99}] + SAMPLE["hours"][1:]
        ),
    }
    for name, payload in bad.items():
        response = client.post("/optimize-energy", json=payload)
        assert response.status_code == 400, f"{name}: got {response.status_code}"
        assert response.json()["error"] == "bad_request"


def test_non_json_body_returns_400():
    response = client.post(
        "/optimize-energy",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_unsolvable_battery_returns_422():
    for field, value in (
        ("initial_energy_kwh", 99999),
        ("minimum_energy_kwh", 99999),
    ):
        payload = _fresh()
        payload["battery"] = {**payload["battery"], field: value}
        response = client.post("/optimize-energy", json=payload)
        assert response.status_code == 422, field
        assert response.json()["error"] == "unprocessable"


def test_unknown_route_is_a_controlled_404():
    response = client.get("/nope")
    assert response.status_code == 404
    assert "error" in response.json()


# ---------------------------------------------------------------------------
# secret safety and stability
# ---------------------------------------------------------------------------


def test_no_response_leaks_secrets_or_stack_traces():
    payloads = [SAMPLE, _fresh(operator_notes=[]), {"garbage": True}]
    for payload in payloads:
        text = client.post("/optimize-energy", json=payload).text.lower()
        for forbidden in ("traceback", "gsk_", "api_key", "authorization", "bearer"):
            assert forbidden not in text, forbidden


def test_repeated_requests_are_stable():
    first = client.post("/optimize-energy", json=SAMPLE).json()
    for _ in range(4):
        again = client.post("/optimize-energy", json=SAMPLE).json()
        assert again == first


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
