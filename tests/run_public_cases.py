"""Public sample-case scoreboard (PROJECT.md section 11).

Two modes:

  offline (default)  Feed each case's ground-truth directives straight into the
                     optimizer. Isolates the optimizer and the replay validator,
                     needs no server and no API key.

  api                POST each case to a running service and score it the way
                     the judge does: interpretation match against ground truth,
                     validity replayed against the GROUND-TRUTH directives (not
                     ours), cost ratio, and p50/p95 latency.

    python tests/run_public_cases.py
    python tests/run_public_cases.py --mode api --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.optimizer.model import Scenario, compile_directives  # noqa: E402
from app.optimizer.solve import InfeasibleError, solve  # noqa: E402
from app.validator.replay import (  # noqa: E402
    TOL,
    replay,
    safe_fallback_plan,
    validate_interpretation,
)

DEFAULT_CASES = ROOT / "data" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
NUMERIC_FIELDS = ("factor", "minimum_energy_kwh", "max_grid_kwh")


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------


def interpretation_matches(got: Any, expected: dict[str, Any]) -> bool:
    """Judge-equivalent comparison. Explanation wording is not compared."""
    if not isinstance(got, dict):
        return False
    if got.get("applies") != expected["applies"]:
        return False
    if got.get("directive_type") != expected["directive_type"]:
        return False

    want, have = expected["structured_adjustment"], got.get("structured_adjustment")
    if want is None:
        return have is None
    if not isinstance(have, dict):
        return False
    if set(have.get("hours") or []) != set(want.get("hours") or []):
        return False
    for field in NUMERIC_FIELDS:
        if field in want:
            try:
                if abs(float(have.get(field)) - float(want[field])) > TOL:
                    return False
            except (TypeError, ValueError):
                return False
    return True


def score_plan(
    case: dict[str, Any], plan: dict[str, Any], rung: str, elapsed_ms: float
) -> dict[str, Any]:
    """Replay a plan against the case's GROUND-TRUTH directives, as the judge does."""
    payload = case["input"]
    expected = case["expected_output"]
    ground_truth = expected["directive_interpretation"]
    scenario = Scenario.from_dict(payload)

    totals = {k: plan[k] for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh")}
    check = replay(plan["hourly_plan"], scenario, ground_truth, totals)

    expected_cost = float(expected["total_cost_bdt"])
    our_cost = float(plan["total_cost_bdt"])
    ratio = 1.0 if our_cost <= TOL else min(1.0, expected_cost / our_cost)

    return {
        "id": case.get("id", "?"),
        "cost": our_cost,
        "expected_cost": expected_cost,
        "delta": our_cost - expected_cost,
        "valid": check.ok,
        "failures": check.failures,
        "ratio": ratio,
        "rung": rung,
        "ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# offline mode
# ---------------------------------------------------------------------------


def run_case_offline(case: dict[str, Any]) -> dict[str, Any]:
    payload = case["input"]
    ground_truth = case["expected_output"]["directive_interpretation"]
    scenario = Scenario.from_dict(payload)
    cons = compile_directives(ground_truth, scenario)

    started = time.perf_counter()
    try:
        result = solve(scenario, cons)
        plan = {
            "hourly_plan": result.hourly_plan,
            "total_grid_kwh": result.total_grid_kwh,
            "total_cost_bdt": result.total_cost_bdt,
            "peak_grid_kwh": result.peak_grid_kwh,
        }
        rung = result.rung
    except InfeasibleError:
        plan = safe_fallback_plan(scenario, ground_truth)
        rung = "safe_fallback"
    elapsed_ms = (time.perf_counter() - started) * 1000

    row = score_plan(case, plan, rung, elapsed_ms)
    row["interp_failures"] = validate_interpretation(
        ground_truth, len(payload["operator_notes"]), scenario.capacity_kwh
    )
    # offline mode bypasses the interpreter, so interpretation is ground truth
    row["interp_matched"] = len(ground_truth)
    row["interp_total"] = len(ground_truth)
    return row


# ---------------------------------------------------------------------------
# api mode
# ---------------------------------------------------------------------------


def run_case_api(case: dict[str, Any], base_url: str, timeout: float) -> dict[str, Any]:
    import httpx

    payload = case["input"]
    expected = case["expected_output"]["directive_interpretation"]

    started = time.perf_counter()
    response = httpx.post(
        f"{base_url.rstrip('/')}/optimize-energy", json=payload, timeout=timeout
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    if response.status_code != 200:
        return {
            "id": case.get("id", "?"),
            "cost": float("inf"),
            "expected_cost": float(case["expected_output"]["total_cost_bdt"]),
            "delta": float("inf"),
            "valid": False,
            "failures": [f"HTTP {response.status_code}: {response.text[:200]}"],
            "ratio": 0.0,
            "rung": "http_error",
            "ms": elapsed_ms,
            "interp_failures": [],
            "interp_matched": 0,
            "interp_total": len(expected),
        }

    body = response.json()
    row = score_plan(case, body, "api", elapsed_ms)

    got = body.get("directive_interpretation") or []
    row["interp_matched"] = sum(
        1
        for index, want in enumerate(expected)
        if index < len(got) and interpretation_matches(got[index], want)
    )
    row["interp_total"] = len(expected)
    row["interp_failures"] = validate_interpretation(
        got, len(payload["operator_notes"]), payload["battery"]["capacity_kwh"]
    )
    if body.get("scenario_id") != payload["scenario_id"]:
        row["interp_failures"].append("scenario_id was not echoed")
    row["mismatches"] = [
        f"note {i}: got {got[i].get('directive_type') if i < len(got) else None}"
        f" {got[i].get('structured_adjustment') if i < len(got) else None}"
        f" | want {want['directive_type']} {want['structured_adjustment']}"
        for i, want in enumerate(expected)
        if not (i < len(got) and interpretation_matches(got[i], want))
    ]
    return row


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--mode", choices=("offline", "api"), default="offline")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help=(
            "seconds to wait between api-mode cases. Use it when the provider "
            "account has a low tokens-per-minute ceiling: firing 10 cases in 10 "
            "seconds trips the limit and measures the fallback path, not the model."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="print every failure in full")
    args = parser.parse_args()

    pack = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = pack["cases"]

    if args.mode == "api":
        print(f"mode=api  base_url={args.base_url}  delay={args.delay}s\n")
        rows = []
        for index, case in enumerate(cases):
            if index and args.delay:
                time.sleep(args.delay)
            rows.append(run_case_api(case, args.base_url, args.timeout))
    else:
        print("mode=offline  (ground-truth directives -> optimizer, no server)\n")
        rows = [run_case_offline(case) for case in cases]

    header = (
        f"{'case':<12}{'cost':>12}{'expected':>12}{'delta':>10}"
        f"{'valid':>8}{'ratio':>8}{'interp':>9}{'ms':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        interp = f"{r['interp_matched']}/{r['interp_total']}"
        cost = "n/a" if r["cost"] == float("inf") else f"{r['cost']:.2f}"
        delta = "n/a" if r["delta"] == float("inf") else f"{r['delta']:.2f}"
        print(
            f"{r['id']:<12}{cost:>12}{r['expected_cost']:>12.2f}{delta:>10}"
            f"{('PASS' if r['valid'] else 'FAIL'):>8}{r['ratio']:>8.3f}"
            f"{interp:>9}{r['ms']:>9.1f}"
        )
    print("-" * len(header))

    valid_count = sum(1 for r in rows if r["valid"])
    exact_count = sum(1 for r in rows if abs(r["delta"]) <= TOL)
    interp_ok = sum(1 for r in rows if r["interp_matched"] == r["interp_total"])
    matched = sum(r["interp_matched"] for r in rows)
    entries = sum(r["interp_total"] for r in rows)
    mean_ratio = sum(r["ratio"] for r in rows) / len(rows)
    latencies = [r["ms"] for r in rows]

    print(f"interpretation    : {interp_ok}/{len(rows)} cases ({matched}/{entries} entries)")
    print(f"replay-valid      : {valid_count}/{len(rows)}")
    print(f"cost matches exact: {exact_count}/{len(rows)}")
    print(f"mean cost ratio   : {mean_ratio:.4f}")
    print(f"latency p50 / p95 : {percentile(latencies, 0.5):.0f} ms / {percentile(latencies, 0.95):.0f} ms")

    for r in rows:
        problems = r["failures"] + r.get("interp_failures", []) + r.get("mismatches", [])
        if problems:
            print(f"\n{r['id']} problems:")
            shown = problems if args.verbose else problems[:5]
            for item in shown:
                print(f"  {item}")
            if len(problems) > len(shown):
                print(f"  ... {len(problems) - len(shown)} more (use --verbose)")

    ok = valid_count == len(rows) and exact_count == len(rows) and interp_ok == len(rows)
    print("\nACCEPTANCE GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
