"""Judge-equivalent replay validator (PROJECT.md section 10).

This is our own private clone of the judge. It deliberately re-derives effective
solar, active reserves, windows and grid caps from the directives itself rather
than reusing ``optimizer.model``, so an error in the constraint compiler shows up
as a validation failure instead of being silently replayed back at us.

It runs on every response before it leaves the process. An invalid schedule zeroes
both the 25-point application category and all optimization credit for that case,
so this module is the last thing to cut.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

HOURS = 24
#: the judge's stated absolute tolerance (Problem Statement 11.5)
TOL = 0.01

ALLOWED_TYPES = frozenset(
    {
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    }
)
ALLOWED_ACTIONS = frozenset({"charge", "discharge", "idle"})

REQUIRED_ADJUSTMENT_KEYS: dict[str, frozenset[str]] = {
    "solar_reduction": frozenset({"hours", "factor"}),
    "minimum_battery_reserve": frozenset({"hours", "minimum_energy_kwh"}),
    "no_charge_window": frozenset({"hours"}),
    "no_discharge_window": frozenset({"hours"}),
    "max_grid_window": frozenset({"hours", "max_grid_kwh"}),
}


@dataclass
class ReplayResult:
    ok: bool
    failures: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_plain(obj: Any) -> Any:
    """Normalise pydantic models to plain dicts; leave dicts alone."""
    if obj is None or isinstance(obj, Mapping):
        return obj
    dump = getattr(obj, "model_dump", None)
    return dump() if callable(dump) else obj


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _replay_constraints(
    interpretation: Iterable[Any], scenario: Any
) -> tuple[list[float], list[float], set[int], set[int], list[float | None]]:
    """Re-derive the per-hour constraint arrays from the directives."""
    solar = [float(s) for s in scenario.solar]
    factor = [1.0] * HOURS
    active_min = [float(scenario.minimum_energy_kwh)] * HOURS
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    max_grid: list[float | None] = [None] * HOURS

    for entry in interpretation:
        dtype = _get(entry, "directive_type")
        adj = _as_plain(_get(entry, "structured_adjustment"))
        if dtype == "no_op" or not adj:
            continue
        hours = [int(h) for h in (adj.get("hours") or []) if 0 <= int(h) < HOURS]
        if dtype == "solar_reduction":
            f = min(1.0, max(0.0, float(adj.get("factor", 1.0))))
            for h in hours:
                factor[h] = min(factor[h], f)
        elif dtype == "minimum_battery_reserve":
            reserve = float(adj.get("minimum_energy_kwh", 0.0))
            for h in hours:
                active_min[h] = max(active_min[h], reserve)
        elif dtype == "no_charge_window":
            no_charge.update(hours)
        elif dtype == "no_discharge_window":
            no_discharge.update(hours)
        elif dtype == "max_grid_window":
            cap = float(adj.get("max_grid_kwh", 0.0))
            for h in hours:
                max_grid[h] = cap if max_grid[h] is None else min(max_grid[h], cap)

    effective_solar = [solar[h] * factor[h] for h in range(HOURS)]
    return effective_solar, active_min, no_charge, no_discharge, max_grid


# ---------------------------------------------------------------------------
# section 10.1 checks
# ---------------------------------------------------------------------------


def validate_interpretation(
    interpretation: Sequence[Any], note_count: int, capacity_kwh: float | None = None
) -> list[str]:
    """Check the directive_interpretation block against sections 5.3 and 10.1."""
    failures: list[str] = []
    if len(interpretation) != note_count:
        failures.append(
            f"interpretation has {len(interpretation)} entries for {note_count} notes"
        )

    seen: set[int] = set()
    for position, entry in enumerate(interpretation):
        idx = _get(entry, "note_index")
        if not isinstance(idx, int) or isinstance(idx, bool):
            failures.append(f"entry {position}: note_index is not an integer")
            continue
        if idx != position:
            failures.append(f"entry {position}: note_index {idx} is out of order")
        if idx in seen:
            failures.append(f"entry {position}: duplicate note_index {idx}")
        seen.add(idx)

        dtype = _get(entry, "directive_type")
        if dtype not in ALLOWED_TYPES:
            failures.append(f"note {idx}: unsupported directive_type {dtype!r}")
            continue

        applies = _get(entry, "applies")
        adj = _as_plain(_get(entry, "structured_adjustment"))

        if dtype == "no_op":
            if applies is not False:
                failures.append(f"note {idx}: no_op must have applies=false")
            if adj is not None:
                failures.append(f"note {idx}: no_op must have a null structured_adjustment")
            continue

        if applies is not True:
            failures.append(f"note {idx}: {dtype} must have applies=true")
        if not isinstance(adj, Mapping):
            failures.append(f"note {idx}: {dtype} needs a structured_adjustment object")
            continue

        required = REQUIRED_ADJUSTMENT_KEYS[dtype]
        if set(adj.keys()) != required:
            failures.append(
                f"note {idx}: structured_adjustment keys {sorted(adj)} != {sorted(required)}"
            )

        raw_hours = adj.get("hours")
        if not isinstance(raw_hours, (list, tuple)) or not raw_hours:
            failures.append(f"note {idx}: hours must be a non-empty array")
        else:
            bad = [
                h
                for h in raw_hours
                if not isinstance(h, int) or isinstance(h, bool) or not 0 <= h <= 23
            ]
            if bad:
                failures.append(f"note {idx}: hours contain non-integers or out-of-range {bad}")
            elif list(raw_hours) != sorted(set(raw_hours)):
                failures.append(f"note {idx}: hours must be unique and ascending")

        if dtype == "solar_reduction":
            f = _num(adj.get("factor"))
            if f is None or not 0.0 <= f <= 1.0:
                failures.append(f"note {idx}: factor {adj.get('factor')!r} is not in [0,1]")
        elif dtype == "minimum_battery_reserve":
            r = _num(adj.get("minimum_energy_kwh"))
            if r is None or r < 0:
                failures.append(f"note {idx}: minimum_energy_kwh must be finite and >= 0")
            elif capacity_kwh is not None and r > capacity_kwh + TOL:
                failures.append(f"note {idx}: minimum_energy_kwh {r} exceeds capacity")
        elif dtype == "max_grid_window":
            g = _num(adj.get("max_grid_kwh"))
            if g is None or g < 0:
                failures.append(f"note {idx}: max_grid_kwh must be finite and >= 0")

    return failures


def replay(
    hourly_plan: Sequence[Any],
    scenario: Any,
    interpretation: Sequence[Any],
    totals: Mapping[str, float] | None = None,
) -> ReplayResult:
    """Replay a schedule exactly the way the judge does.

    ``interpretation`` should be the directive set the plan is judged against -
    the organizer's ground truth when scoring offline, our own validated
    interpretation at request time.
    """
    failures: list[str] = []

    if len(hourly_plan) != HOURS:
        return ReplayResult(False, [f"hourly_plan has {len(hourly_plan)} entries, need 24"])
    hours = [_get(row, "hour") for row in hourly_plan]
    if hours != list(range(HOURS)):
        return ReplayResult(False, ["hourly_plan hours must be 0..23 unique and ascending"])

    eff_solar, active_min, no_charge, no_discharge, max_grid = _replay_constraints(
        interpretation, scenario
    )

    energy = float(scenario.initial_energy_kwh)
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for h, row in enumerate(hourly_plan):
        grid = _num(_get(row, "grid_kwh"))
        solar_used = _num(_get(row, "solar_used_kwh"))
        action = _get(row, "battery_action")
        magnitude = _num(_get(row, "battery_kwh"))
        reported_after = _num(_get(row, "battery_energy_after_kwh"))

        if grid is None or solar_used is None or magnitude is None or reported_after is None:
            failures.append(f"hour {h}: non-finite or missing numeric value")
            continue
        if grid < -TOL or solar_used < -TOL or magnitude < -TOL or reported_after < -TOL:
            failures.append(f"hour {h}: negative value in plan")
        if action not in ALLOWED_ACTIONS:
            failures.append(f"hour {h}: battery_action {action!r} is not a valid enum value")
            continue
        if action == "idle" and abs(magnitude) > 0:
            failures.append(f"hour {h}: idle hour must have battery_kwh = 0")

        if solar_used > eff_solar[h] + TOL:
            failures.append(
                f"hour {h}: solar_used {solar_used} exceeds effective solar {eff_solar[h]:.3f}"
            )

        charge = magnitude if action == "charge" else 0.0
        discharge = magnitude if action == "discharge" else 0.0

        if charge > scenario.max_charge_kwh_per_hour + TOL:
            failures.append(f"hour {h}: charge {charge} exceeds the hourly charge limit")
        if discharge > scenario.max_discharge_kwh_per_hour + TOL:
            failures.append(f"hour {h}: discharge {discharge} exceeds the hourly discharge limit")
        if h in no_charge and charge > TOL:
            failures.append(f"hour {h}: charging inside a no_charge_window")
        if h in no_discharge and discharge > TOL:
            failures.append(f"hour {h}: discharging inside a no_discharge_window")
        if max_grid[h] is not None and grid > max_grid[h] + TOL:
            failures.append(f"hour {h}: grid {grid} exceeds the cap {max_grid[h]}")

        lhs = grid + solar_used + discharge
        rhs = scenario.demand[h] + charge
        if abs(lhs - rhs) > TOL:
            failures.append(f"hour {h}: energy balance off by {lhs - rhs:.4f}")

        energy = energy + charge - discharge
        if energy < active_min[h] - TOL:
            failures.append(
                f"hour {h}: battery energy {energy:.3f} below the active minimum {active_min[h]}"
            )
        if energy > scenario.capacity_kwh + TOL:
            failures.append(f"hour {h}: battery energy {energy:.3f} above capacity")
        if abs(energy - reported_after) > TOL:
            failures.append(
                f"hour {h}: battery_energy_after_kwh {reported_after} != replayed {energy:.3f}"
            )

        total_grid += grid
        total_cost += grid * scenario.tariff[h]
        peak_grid = max(peak_grid, grid)

    if abs(energy - scenario.initial_energy_kwh) > TOL:
        failures.append(
            f"end-of-day battery energy {energy:.3f} != initial {scenario.initial_energy_kwh}"
        )

    if totals is not None:
        for key, recomputed in (
            ("total_grid_kwh", total_grid),
            ("total_cost_bdt", total_cost),
            ("peak_grid_kwh", peak_grid),
        ):
            reported = _num(totals.get(key))
            if reported is None or abs(reported - recomputed) > TOL:
                failures.append(f"{key} {totals.get(key)!r} != recomputed {recomputed:.3f}")

    return ReplayResult(not failures, failures)


# ---------------------------------------------------------------------------
# section 10.3 safe fallback plan
# ---------------------------------------------------------------------------


def safe_fallback_plan(scenario: Any, interpretation: Sequence[Any]) -> dict[str, Any]:
    """Expensive but valid plan, used when every solver rung has failed.

    The section 10.3 shape: battery idle, solar used up to demand, grid covers the
    rest. Section 10.3 then adds one exception - discharge down to the active
    minimum when a ``max_grid_window`` cap would be breached - and names one gap:
    the idle plan only satisfies a reserve "if initial >= reserve".

    Two deterministic passes bracket the idle plan so both hold: a pre-charge
    before the first constrained hour, covering whatever the cap will borrow plus
    any reserve that sits above the starting energy, and a settle-up afterwards
    (recharge or shed) so end-of-day neutrality still lands on ``initial``.
    """
    eff_solar, active_min, no_charge, no_discharge, max_grid = _replay_constraints(
        interpretation, scenario
    )

    solar_used = [
        min(max(0.0, eff_solar[h]), float(scenario.demand[h])) for h in range(HOURS)
    ]
    base_grid = [float(scenario.demand[h]) - solar_used[h] for h in range(HOURS)]

    # how much each capped hour needs the battery to cover
    need: dict[int, float] = {}
    for h in range(HOURS):
        cap = max_grid[h]
        if cap is None or h in no_discharge:
            continue
        deficit = base_grid[h] - cap
        if deficit > 1e-9:
            need[h] = min(deficit, float(scenario.max_discharge_kwh_per_hour))

    # hours whose reserve floor sits above the starting energy - the idle plan
    # cannot satisfy those without charging up first
    raised = [h for h in range(HOURS) if active_min[h] > scenario.initial_energy_kwh]

    charge = [0.0] * HOURS
    discharge = [0.0] * HOURS

    if need or raised:
        _precharge(
            need, raised, charge, base_grid, active_min, scenario, no_charge, max_grid
        )

    # forward pass: apply the charges, then the discharges, clamped to the bounds
    energy = float(scenario.initial_energy_kwh)
    for h in range(HOURS):
        if charge[h] > 0:
            charge[h] = min(charge[h], max(0.0, scenario.capacity_kwh - energy))
            energy += charge[h]
        want = need.get(h, 0.0)
        if want > 0:
            discharge[h] = min(want, max(0.0, energy - active_min[h]))
            energy -= discharge[h]

    # Settle back on the starting energy, or neutrality fails. A deficit means the
    # cap borrowed energy; a surplus means a pre-charge could not be spent because
    # the cap turned out to be unreachable.
    delta = scenario.initial_energy_kwh - energy
    if delta > 1e-9:
        _recharge_for_neutrality(
            delta, charge, discharge, base_grid, scenario, no_charge, max_grid
        )
    elif delta < -1e-9:
        _shed_surplus(
            -delta, charge, discharge, base_grid, raised, scenario, no_discharge
        )

    rows = _emit_rows(charge, discharge, base_grid, solar_used, scenario, active_min)
    total_grid = round(sum(r["grid_kwh"] for r in rows), 3)
    total_cost = round(sum(r["grid_kwh"] * scenario.tariff[r["hour"]] for r in rows), 3)
    peak_grid = round(max(r["grid_kwh"] for r in rows), 3)
    return {
        "hourly_plan": rows,
        "total_grid_kwh": total_grid,
        "total_cost_bdt": total_cost,
        "peak_grid_kwh": peak_grid,
    }


def _charge_headroom(
    h: int, base_grid: list[float], scenario: Any, max_grid: list[float | None]
) -> float:
    """How much this hour may charge without breaching its own grid cap."""
    headroom = float(scenario.max_charge_kwh_per_hour)
    cap = max_grid[h]
    if cap is not None:
        headroom = min(headroom, max(0.0, cap - base_grid[h]))
    return headroom


def _precharge(
    need: dict[int, float],
    raised: list[int],
    charge: list[float],
    base_grid: list[float],
    active_min: list[float],
    scenario: Any,
    no_charge: set[int],
    max_grid: list[float | None],
) -> None:
    """Top the battery up before the first hour a directive constrains it.

    Covers two independent requirements, added together because the same stored
    kWh cannot both be held as reserve and spent under a grid cap:

    * what the capped hours will draw, beyond what is already spendable above
      the reserve floor
    * how far the highest reserve floor sits above the starting energy
    """
    initial = float(scenario.initial_energy_kwh)
    constrained = sorted(set(need) | set(raised))
    first = constrained[0]

    # The highest floor anywhere in the day, not just the floor local to each
    # window: a capped hour that drains the battery early would otherwise be
    # counted as free, then leave a later reserve window short.
    day_floor = max(active_min)

    cap_shortfall = 0.0
    if need:
        # energy spendable under the caps without undercutting any reserve
        usable = max(0.0, initial - day_floor)
        cap_shortfall = max(0.0, sum(need.values()) - usable)

    reserve_shortfall = 0.0
    if raised:
        reserve_shortfall = max(active_min[h] for h in raised) - initial

    room = scenario.capacity_kwh - initial
    remaining = min(cap_shortfall + max(0.0, reserve_shortfall), max(0.0, room))
    if remaining <= 1e-9:
        return

    # ``first`` itself is in range: charging during an hour raises the energy the
    # reserve floor is checked against. A capped hour self-limits to zero headroom.
    for h in sorted(
        (h for h in range(first + 1) if h not in no_charge),
        key=lambda h: scenario.tariff[h],
    ):
        if remaining <= 1e-9:
            break
        add = min(remaining, _charge_headroom(h, base_grid, scenario, max_grid))
        if add <= 1e-9:
            continue
        charge[h] = add
        remaining -= add


def _recharge_for_neutrality(
    deficit: float,
    charge: list[float],
    discharge: list[float],
    base_grid: list[float],
    scenario: Any,
    no_charge: set[int],
    max_grid: list[float | None],
) -> None:
    """Refill ``deficit`` kWh in the cheapest hours after the last discharge.

    Charging only after the battery has been drawn down keeps the running energy
    at or below where it started, so the capacity ceiling can never be breached.
    """
    last_discharge = max((h for h in range(HOURS) if discharge[h] > 0), default=-1)
    candidates = [
        h
        for h in range(last_discharge + 1, HOURS)
        if h not in no_charge and charge[h] == 0 and discharge[h] == 0
    ]
    for h in sorted(candidates, key=lambda h: scenario.tariff[h]):
        if deficit <= 1e-9:
            break
        add = min(deficit, _charge_headroom(h, base_grid, scenario, max_grid))
        if add <= 1e-9:
            continue
        charge[h] = add
        deficit -= add
    # A leftover deficit means no hour could take the charge without breaking a
    # hard constraint. We let the caller's replay flag it rather than emit a
    # silent violation.


def _shed_surplus(
    surplus: float,
    charge: list[float],
    discharge: list[float],
    base_grid: list[float],
    raised: list[int],
    scenario: Any,
    no_discharge: set[int],
) -> None:
    """Spend leftover pre-charge in the priciest hours, once nothing still needs it.

    Discharging only lowers grid import, so it can never breach a grid cap, and
    running after both the last charge and the last raised reserve floor keeps the
    energy state falling monotonically back towards where it started.
    """
    last_charge = max((h for h in range(HOURS) if charge[h] > 0), default=-1)
    start = max(last_charge, max(raised, default=-1)) + 1
    candidates = [
        h
        for h in range(start, HOURS)
        if h not in no_discharge and charge[h] == 0 and discharge[h] == 0
    ]
    for h in sorted(candidates, key=lambda h: -scenario.tariff[h]):
        if surplus <= 1e-9:
            break
        take = min(surplus, float(scenario.max_discharge_kwh_per_hour), base_grid[h])
        if take <= 1e-9:
            continue
        discharge[h] = take
        surplus -= take


def _emit_rows(
    charge: list[float],
    discharge: list[float],
    base_grid: list[float],
    solar_used: list[float],
    scenario: Any,
    active_min: list[float],
) -> list[dict[str, Any]]:
    """Round, forward-simulate the energy state, and build the plan rows."""
    rows: list[dict[str, Any]] = []
    energy = float(scenario.initial_energy_kwh)
    for h in range(HOURS):
        c = round(max(0.0, charge[h]), 3)
        d = round(max(0.0, discharge[h]), 3)
        if c > 0 and d > 0:  # never both in one hour
            c, d = (c - d, 0.0) if c > d else (0.0, d - c)
        c = min(c, max(0.0, scenario.capacity_kwh - energy))
        energy += c
        d = min(d, max(0.0, energy - active_min[h]))
        energy = round(energy - d, 6)

        if c > 0:
            action, magnitude = "charge", c
        elif d > 0:
            action, magnitude = "discharge", d
        else:
            action, magnitude = "idle", 0.0

        rows.append(
            {
                "hour": h,
                "grid_kwh": round(max(0.0, base_grid[h] + c - d), 3),
                "solar_used_kwh": round(max(0.0, solar_used[h]), 3),
                "battery_action": action,
                "battery_kwh": round(magnitude, 3),
                "battery_energy_after_kwh": round(energy, 3),
            }
        )
    return rows
