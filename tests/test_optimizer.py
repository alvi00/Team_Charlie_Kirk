"""Unit tests for the optimizer and the replay validator (PROJECT.md 6, 9, 10).

Runs under pytest, or standalone with no extra dependency:

    python tests/test_optimizer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.optimizer.model import Scenario, compile_directives  # noqa: E402
from app.optimizer.solve import solve  # noqa: E402
from app.validator.replay import replay, safe_fallback_plan  # noqa: E402

HOURS = 24


def make_scenario(**overrides) -> Scenario:
    """A small, deliberately lumpy day: cheap nights, expensive evening peak."""
    demand = [100.0] * HOURS
    solar = [0.0] * 6 + [40.0] * 12 + [0.0] * 6
    tariff = [5.0] * 6 + [8.0] * 11 + [14.0] * 5 + [8.0] * 2
    base = {
        "scenario_id": "TEST",
        "demand": tuple(demand),
        "solar": tuple(solar),
        "tariff": tuple(tariff),
        "capacity_kwh": 200.0,
        "initial_energy_kwh": 100.0,
        "minimum_energy_kwh": 40.0,
        "max_charge_kwh_per_hour": 50.0,
        "max_discharge_kwh_per_hour": 50.0,
    }
    base.update(overrides)
    return Scenario(**base)


def directive(dtype: str, hours: list[int], **extra) -> dict:
    adjustment = None if dtype == "no_op" else {"hours": hours, **extra}
    return {
        "note_index": 0,
        "applies": dtype != "no_op",
        "directive_type": dtype,
        "structured_adjustment": adjustment,
        "explanation": "test directive",
    }


def solve_with(directives: list[dict], scenario: Scenario | None = None):
    scenario = scenario or make_scenario()
    cons = compile_directives(directives, scenario)
    result = solve(scenario, cons)
    check = replay(
        result.hourly_plan,
        scenario,
        directives,
        {
            "total_grid_kwh": result.total_grid_kwh,
            "total_cost_bdt": result.total_cost_bdt,
            "peak_grid_kwh": result.peak_grid_kwh,
        },
    )
    assert check.ok, check.failures
    return scenario, result


# ---------------------------------------------------------------------------
# section 9.5 post-solve conditioning
# ---------------------------------------------------------------------------


def test_plan_shape_and_action_consistency():
    _, result = solve_with([])
    assert [row["hour"] for row in result.hourly_plan] == list(range(HOURS))
    for row in result.hourly_plan:
        assert row["battery_action"] in {"charge", "discharge", "idle"}
        assert row["battery_kwh"] >= 0
        assert row["grid_kwh"] >= 0
        assert row["solar_used_kwh"] >= 0
        # netting guarantees an hour is never both charging and discharging
        if row["battery_action"] == "idle":
            assert row["battery_kwh"] == 0


def test_end_of_day_neutrality():
    scenario, result = solve_with([])
    assert abs(
        result.hourly_plan[-1]["battery_energy_after_kwh"] - scenario.initial_energy_kwh
    ) < 0.01


def test_battery_energy_is_forward_simulated_from_rounded_actions():
    scenario, result = solve_with([])
    energy = scenario.initial_energy_kwh
    for row in result.hourly_plan:
        if row["battery_action"] == "charge":
            energy += row["battery_kwh"]
        elif row["battery_action"] == "discharge":
            energy -= row["battery_kwh"]
        assert abs(energy - row["battery_energy_after_kwh"]) < 1e-6


def test_totals_recomputed_from_rounded_plan():
    scenario, result = solve_with([])
    grid = sum(row["grid_kwh"] for row in result.hourly_plan)
    cost = sum(row["grid_kwh"] * scenario.tariff[row["hour"]] for row in result.hourly_plan)
    peak = max(row["grid_kwh"] for row in result.hourly_plan)
    assert abs(grid - result.total_grid_kwh) < 0.01
    assert abs(cost - result.total_cost_bdt) < 0.01
    assert abs(peak - result.peak_grid_kwh) < 0.01


# ---------------------------------------------------------------------------
# section 6 directive semantics, enforced end to end
# ---------------------------------------------------------------------------


def test_solar_reduction_caps_usable_solar():
    scenario, result = solve_with([directive("solar_reduction", [10, 11], factor=0.25)])
    for h in (10, 11):
        assert result.hourly_plan[h]["solar_used_kwh"] <= scenario.solar[h] * 0.25 + 0.01


def test_no_charge_window_is_respected():
    _, result = solve_with([directive("no_charge_window", [2, 3, 4])])
    for h in (2, 3, 4):
        assert result.hourly_plan[h]["battery_action"] != "charge"


def test_no_discharge_window_is_respected():
    _, result = solve_with([directive("no_discharge_window", [18, 19])])
    for h in (18, 19):
        assert result.hourly_plan[h]["battery_action"] != "discharge"


def test_minimum_battery_reserve_raises_the_floor():
    _, result = solve_with(
        [directive("minimum_battery_reserve", [18, 19, 20], minimum_energy_kwh=150)]
    )
    for h in (18, 19, 20):
        assert result.hourly_plan[h]["battery_energy_after_kwh"] >= 150 - 0.01


def test_max_grid_window_caps_import():
    _, result = solve_with([directive("max_grid_window", [17, 18, 19], max_grid_kwh=60)])
    for h in (17, 18, 19):
        assert result.hourly_plan[h]["grid_kwh"] <= 60.01


def test_no_op_changes_nothing():
    _, plain = solve_with([])
    _, with_noop = solve_with([directive("no_op", [])])
    assert abs(plain.total_cost_bdt - with_noop.total_cost_bdt) < 0.01


# ---------------------------------------------------------------------------
# section 8.4 conflict resolution
# ---------------------------------------------------------------------------


def test_conflicting_directives_combine_conservatively():
    scenario = make_scenario()
    cons = compile_directives(
        [
            directive("minimum_battery_reserve", [18], minimum_energy_kwh=90),
            directive("minimum_battery_reserve", [18], minimum_energy_kwh=120),
            directive("max_grid_window", [18], max_grid_kwh=150),
            directive("max_grid_window", [18], max_grid_kwh=120),
            directive("solar_reduction", [12], factor=0.5),
            directive("solar_reduction", [12], factor=0.2),
            directive("no_charge_window", [2]),
            directive("no_charge_window", [3]),
        ],
        scenario,
    )
    assert cons.active_min[18] == 120      # reserves take the max
    assert cons.max_grid[18] == 120        # grid caps take the min
    assert cons.solar_factor[12] == 0.2    # solar factors take the min
    assert not cons.charge_allowed[2] and not cons.charge_allowed[3]  # windows union
    assert cons.active_min[0] == scenario.minimum_energy_kwh


# ---------------------------------------------------------------------------
# section 10 the validator must actually catch violations
# ---------------------------------------------------------------------------


def test_replay_rejects_a_tampered_plan():
    scenario = make_scenario()
    result = solve(scenario, compile_directives([], scenario))

    broken = [dict(row) for row in result.hourly_plan]
    broken[5]["grid_kwh"] += 25  # breaks the balance equation
    assert not replay(broken, scenario, []).ok

    broken = [dict(row) for row in result.hourly_plan]
    broken[5]["battery_action"] = "idle"
    broken[5]["battery_kwh"] = 17.0  # idle must carry 0
    assert not replay(broken, scenario, []).ok

    broken = [dict(row) for row in result.hourly_plan]
    broken[-1]["battery_energy_after_kwh"] += 30  # breaks neutrality
    assert not replay(broken, scenario, []).ok


def test_replay_rejects_a_directive_violation():
    scenario = make_scenario()
    unconstrained = [directive("no_op", [])]
    result = solve(scenario, compile_directives(unconstrained, scenario))
    # The same plan judged against a directive it was not built for must fail.
    strict = [directive("minimum_battery_reserve", list(range(24)), minimum_energy_kwh=199)]
    assert not replay(result.hourly_plan, scenario, strict).ok


# ---------------------------------------------------------------------------
# section 10.3 safe fallback plan
# ---------------------------------------------------------------------------


def test_safe_fallback_is_valid_without_directives():
    scenario = make_scenario()
    plan = safe_fallback_plan(scenario, [])
    check = replay(
        plan["hourly_plan"],
        scenario,
        [],
        {k: plan[k] for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh")},
    )
    assert check.ok, check.failures


def test_safe_fallback_discharges_under_a_grid_cap():
    scenario = make_scenario()
    directives = [directive("max_grid_window", [18, 19, 20], max_grid_kwh=70)]
    plan = safe_fallback_plan(scenario, directives)
    check = replay(
        plan["hourly_plan"],
        scenario,
        directives,
        {k: plan[k] for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh")},
    )
    assert check.ok, check.failures
    for h in (18, 19, 20):
        assert plan["hourly_plan"][h]["grid_kwh"] <= 70.01


def test_safe_fallback_is_valid_but_not_cheaper_than_the_lp():
    scenario = make_scenario()
    fallback = safe_fallback_plan(scenario, [])
    optimal = solve(scenario, compile_directives([], scenario))
    assert fallback["total_cost_bdt"] >= optimal.total_cost_bdt - 0.01


# ---------------------------------------------------------------------------
# section 9.4 infeasibility ladder
# ---------------------------------------------------------------------------


def test_impossible_grid_cap_still_returns_a_plan():
    # A cap of 0 for the whole day with far too little solar cannot be met.
    scenario = make_scenario()
    directives = [directive("max_grid_window", list(range(24)), max_grid_kwh=0)]
    cons = compile_directives(directives, scenario)
    result = solve(scenario, cons)
    assert result.rung != "exact"
    assert len(result.hourly_plan) == 24


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
