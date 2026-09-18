"""Directive -> constraint compilation (PROJECT.md sections 6, 8.4 and 9.3).

This module turns a validated ``directive_interpretation`` list plus the raw
scenario into the flat per-hour arrays the LP needs:

    effective_solar[h]   solar_kwh[h] scaled by every solar_reduction on h
    active_min[h]        max(base minimum_energy_kwh, any reserve directive on h)
    charge_allowed[h]    False inside a no_charge_window
    discharge_allowed[h] False inside a no_discharge_window
    max_grid[h]          per-hour import cap, or None

Duplicate directives of the same type are combined conservatively per section
8.4: reserves take the max, grid caps the min, windows union, solar factors the
min (most restrictive).  Because solar factors are merged to one factor per hour
*before* they are applied, section 9.3's "product of factors applying to h"
reduces to that single merged factor - the two spellings agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

HOURS = 24


@dataclass(frozen=True)
class Scenario:
    """The numeric half of an optimize request, indexed by hour 0..23."""

    scenario_id: str
    demand: tuple[float, ...]
    solar: tuple[float, ...]
    tariff: tuple[float, ...]
    capacity_kwh: float
    initial_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kwh_per_hour: float
    max_discharge_kwh_per_hour: float

    @classmethod
    def from_request(cls, request: Any) -> "Scenario":
        """Build from an ``OptimizeRequest`` (hours already sorted by hour)."""
        rows = sorted(request.hours, key=lambda r: r.hour)
        b = request.battery
        return cls(
            scenario_id=request.scenario_id,
            demand=tuple(float(r.demand_kwh) for r in rows),
            solar=tuple(float(r.solar_kwh) for r in rows),
            tariff=tuple(float(r.tariff_bdt_per_kwh) for r in rows),
            capacity_kwh=float(b.capacity_kwh),
            initial_energy_kwh=float(b.initial_energy_kwh),
            minimum_energy_kwh=float(b.minimum_energy_kwh),
            max_charge_kwh_per_hour=float(b.max_charge_kwh_per_hour),
            max_discharge_kwh_per_hour=float(b.max_discharge_kwh_per_hour),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Scenario":
        """Build straight from a raw request dict (used by the offline harness)."""
        rows = sorted(payload["hours"], key=lambda r: int(r["hour"]))
        b = payload["battery"]
        return cls(
            scenario_id=str(payload.get("scenario_id", "")),
            demand=tuple(float(r["demand_kwh"]) for r in rows),
            solar=tuple(float(r["solar_kwh"]) for r in rows),
            tariff=tuple(float(r["tariff_bdt_per_kwh"]) for r in rows),
            capacity_kwh=float(b["capacity_kwh"]),
            initial_energy_kwh=float(b["initial_energy_kwh"]),
            minimum_energy_kwh=float(b["minimum_energy_kwh"]),
            max_charge_kwh_per_hour=float(b["max_charge_kwh_per_hour"]),
            max_discharge_kwh_per_hour=float(b["max_discharge_kwh_per_hour"]),
        )


@dataclass
class CompiledConstraints:
    """Per-hour arrays the LP and the replay validator both consume."""

    effective_solar: list[float]
    active_min: list[float]
    charge_allowed: list[bool]
    discharge_allowed: list[bool]
    max_grid: list[float | None]
    #: merged solar factor per hour, kept for the plan summary / diagnostics
    solar_factor: list[float] = field(default_factory=lambda: [1.0] * HOURS)


def _as_entry(entry: Any) -> tuple[str, Mapping[str, Any] | None]:
    """Normalise a DirectiveInterpretation model *or* plain dict to a pair."""
    if isinstance(entry, Mapping):
        dtype = entry.get("directive_type")
        adj = entry.get("structured_adjustment")
    else:
        dtype = getattr(entry, "directive_type", None)
        adj = getattr(entry, "structured_adjustment", None)
    if adj is not None and not isinstance(adj, Mapping):
        # pydantic adjustment model
        adj = adj.model_dump()
    return (str(dtype) if dtype is not None else "no_op", adj)


def _valid_hours(adj: Mapping[str, Any] | None) -> list[int]:
    if not adj:
        return []
    raw = adj.get("hours") or []
    out: list[int] = []
    for h in raw:
        try:
            hi = int(h)
        except (TypeError, ValueError):
            continue
        if 0 <= hi < HOURS:
            out.append(hi)
    return sorted(set(out))


def compile_directives(
    interpretation: Iterable[Any],
    scenario: Scenario,
) -> CompiledConstraints:
    """Compile directives into the per-hour constraint arrays.

    ``interpretation`` accepts either ``DirectiveInterpretation`` models or the
    equivalent plain dicts, so the offline harness can feed the organizer's
    ground-truth directives in directly.
    """
    factor: list[float] = [1.0] * HOURS
    active_min: list[float] = [float(scenario.minimum_energy_kwh)] * HOURS
    charge_allowed: list[bool] = [True] * HOURS
    discharge_allowed: list[bool] = [True] * HOURS
    max_grid: list[float | None] = [None] * HOURS

    for entry in interpretation:
        dtype, adj = _as_entry(entry)
        if dtype == "no_op" or adj is None:
            continue
        hours = _valid_hours(adj)
        if not hours:
            continue

        if dtype == "solar_reduction":
            f = float(adj.get("factor", 1.0))
            f = min(1.0, max(0.0, f))
            for h in hours:
                factor[h] = min(factor[h], f)  # most restrictive wins

        elif dtype == "minimum_battery_reserve":
            reserve = float(adj.get("minimum_energy_kwh", 0.0))
            reserve = min(max(0.0, reserve), float(scenario.capacity_kwh))
            for h in hours:
                active_min[h] = max(active_min[h], reserve)

        elif dtype == "no_charge_window":
            for h in hours:
                charge_allowed[h] = False

        elif dtype == "no_discharge_window":
            for h in hours:
                discharge_allowed[h] = False

        elif dtype == "max_grid_window":
            cap = float(adj.get("max_grid_kwh", 0.0))
            cap = max(0.0, cap)
            for h in hours:
                max_grid[h] = cap if max_grid[h] is None else min(max_grid[h], cap)

    effective_solar = [scenario.solar[h] * factor[h] for h in range(HOURS)]
    return CompiledConstraints(
        effective_solar=effective_solar,
        active_min=active_min,
        charge_allowed=charge_allowed,
        discharge_allowed=discharge_allowed,
        max_grid=max_grid,
        solar_factor=factor,
    )


def directive_hours(interpretation: Sequence[Any], directive_type: str) -> set[int]:
    """Union of the hours carried by every directive of ``directive_type``."""
    hours: set[int] = set()
    for entry in interpretation:
        dtype, adj = _as_entry(entry)
        if dtype == directive_type:
            hours.update(_valid_hours(adj))
    return hours
