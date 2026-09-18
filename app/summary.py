"""Deterministic ``plan_summary`` text (PROJECT.md section 4).

Kept deterministic on purpose: it costs no latency, cannot fail, and the LLM's
mandatory role is note interpretation, not prose.
"""

from __future__ import annotations

from typing import Any, Sequence

_DIRECTIVE_PHRASE = {
    "solar_reduction": "reduced solar availability",
    "minimum_battery_reserve": "a raised battery reserve",
    "no_charge_window": "a no-charging window",
    "no_discharge_window": "a no-discharging window",
    "max_grid_window": "a grid import cap",
}


def build_plan_summary(
    interpretation: Sequence[Any],
    plan: Sequence[dict[str, Any]],
    total_cost_bdt: float,
    peak_grid_kwh: float,
    degraded_note: str | None = None,
) -> str:
    applied = [
        _DIRECTIVE_PHRASE[entry["directive_type"]]
        for entry in (_as_dict(e) for e in interpretation)
        if entry["directive_type"] in _DIRECTIVE_PHRASE
    ]
    no_ops = sum(1 for e in interpretation if _as_dict(e)["directive_type"] == "no_op")

    charge_hours = [r["hour"] for r in plan if r["battery_action"] == "charge"]
    discharge_hours = [r["hour"] for r in plan if r["battery_action"] == "discharge"]

    parts: list[str] = []
    if applied:
        unique = sorted(set(applied))
        parts.append("Applied " + ", ".join(unique) + " as hard constraints.")
    else:
        parts.append("No operator note changed today's schedule.")
    if no_ops:
        parts.append(f"{no_ops} note(s) were irrelevant and treated as no_op.")

    if charge_hours or discharge_hours:
        parts.append(
            "The battery charges in hour(s) "
            + _compact(charge_hours)
            + " and discharges in hour(s) "
            + _compact(discharge_hours)
            + ", ending the day at its starting energy."
        )
    else:
        parts.append("The battery stays idle and ends the day at its starting energy.")

    parts.append(
        f"Solar is used first, then the cheapest grid hours, for a total of "
        f"{total_cost_bdt:g} BDT with a peak import of {peak_grid_kwh:g} kWh."
    )
    if degraded_note:
        parts.append(degraded_note)
    return " ".join(parts)


def _as_dict(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return entry
    return entry.model_dump()


def _compact(hours: Sequence[int]) -> str:
    return ", ".join(str(h) for h in hours) if hours else "none"
