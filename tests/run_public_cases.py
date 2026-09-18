"""Public sample-case scoreboard (PROJECT.md section 11).

P1 runs OFFLINE: each case's ``expected_output.directive_interpretation`` is fed
straight into the optimizer, bypassing the API and the P0 stub interpreter. That
isolates the optimizer and the replay validator, which is what this phase is
being graded on.

The API mode (POST each case to a running service and also score interpretation
accuracy against ground truth) arrives in P2, once the LLM interpreter exists.

    python tests/run_public_cases.py
    python tests/run_public_cases.py --cases data/..._Sample_Cases.json
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


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    """Optimize one case against its ground-truth directives and replay it."""
    payload = case["input"]
    expected = case["expected_output"]
    ground_truth = expected["directive_interpretation"]

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

    # The judge replays against its own ground-truth directives, so we do too.
    check = replay(
        plan["hourly_plan"],
        scenario,
        ground_truth,
        {
            "total_grid_kwh": plan["total_grid_kwh"],
            "total_cost_bdt": plan["total_cost_bdt"],
            "peak_grid_kwh": plan["peak_grid_kwh"],
        },
    )
    interp_failures = validate_interpretation(
        ground_truth, len(payload["operator_notes"]), scenario.capacity_kwh
    )

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
        "interp_failures": interp_failures,
        "ratio": ratio,
        "rung": rung,
        "ms": elapsed_ms,
        "total_grid_kwh": plan["total_grid_kwh"],
        "expected_total_grid_kwh": float(expected["total_grid_kwh"]),
        "peak_grid_kwh": plan["peak_grid_kwh"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--verbose", action="store_true", help="print every replay failure in full"
    )
    args = parser.parse_args()

    pack = json.loads(args.cases.read_text(encoding="utf-8"))
    rows = [run_case(case) for case in pack["cases"]]

    header = (
        f"{'case':<12}{'cost':>12}{'expected':>12}{'delta':>10}"
        f"{'valid':>8}{'ratio':>8}{'rung':>16}{'ms':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['id']:<12}{r['cost']:>12.2f}{r['expected_cost']:>12.2f}"
            f"{r['delta']:>10.2f}{('PASS' if r['valid'] else 'FAIL'):>8}"
            f"{r['ratio']:>8.3f}{r['rung']:>16}{r['ms']:>8.1f}"
        )
    print("-" * len(header))

    valid_count = sum(1 for r in rows if r["valid"])
    exact_count = sum(1 for r in rows if abs(r["delta"]) <= TOL)
    mean_ratio = sum(r["ratio"] for r in rows) / len(rows)
    print(f"replay-valid      : {valid_count}/{len(rows)}")
    print(f"cost matches exact: {exact_count}/{len(rows)}")
    print(f"mean cost ratio   : {mean_ratio:.4f}")

    for r in rows:
        if r["failures"] or r["interp_failures"]:
            print(f"\n{r['id']} problems:")
            for f in r["interp_failures"]:
                print(f"  interpretation: {f}")
            shown = r["failures"] if args.verbose else r["failures"][:5]
            for f in shown:
                print(f"  replay: {f}")
            hidden = len(r["failures"]) - len(shown)
            if hidden > 0:
                print(f"  replay: ... {hidden} more (use --verbose)")

    ok = valid_count == len(rows) and exact_count == len(rows)
    print("\nACCEPTANCE GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
