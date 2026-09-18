"""Deterministic validation and normalization of LLM output (PROJECT.md 8).

LLM output is untrusted structured data. Every entry passes through here before
it reaches the optimizer. The guiding rule from section 8.1 is repair, not
reject: a model that gets ``applies`` backwards or wraps the payload in an extra
key has still understood the note, and those points are recoverable
deterministically.

This layer is explicitly permitted by the rules as normalization. It must never
pattern-match a known answer - only normalize what the note and the model say.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .numbers import normalize_factor, normalize_grid_cap, normalize_reserve
from .timeparse import normalize_hours, parse_hour_windows

ALLOWED_TYPES = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
)

#: spellings seen from models, mapped onto the canonical type
_TYPE_ALIASES: dict[str, str] = {
    "solar": "solar_reduction",
    "solarreduction": "solar_reduction",
    "solar_reduction": "solar_reduction",
    "reduce_solar": "solar_reduction",
    "solar_curtailment": "solar_reduction",
    "minimum_battery_reserve": "minimum_battery_reserve",
    "minimumbatteryreserve": "minimum_battery_reserve",
    "battery_reserve": "minimum_battery_reserve",
    "min_battery_reserve": "minimum_battery_reserve",
    "minimum_reserve": "minimum_battery_reserve",
    "reserve": "minimum_battery_reserve",
    "no_charge_window": "no_charge_window",
    "nochargewindow": "no_charge_window",
    "no_charge": "no_charge_window",
    "charge_window": "no_charge_window",
    "no_charging": "no_charge_window",
    "no_discharge_window": "no_discharge_window",
    "nodischargewindow": "no_discharge_window",
    "no_discharge": "no_discharge_window",
    "discharge_window": "no_discharge_window",
    "no_discharging": "no_discharge_window",
    "max_grid_window": "max_grid_window",
    "maxgridwindow": "max_grid_window",
    "max_grid": "max_grid_window",
    "grid_cap": "max_grid_window",
    "grid_limit": "max_grid_window",
    "max_grid_import": "max_grid_window",
    "no_op": "no_op",
    "noop": "no_op",
    "none": "no_op",
    "not_applicable": "no_op",
    "n/a": "no_op",
    "irrelevant": "no_op",
}

_EXPLANATIONS = {
    "solar_reduction": "Usable solar is reduced during the stated hours.",
    "minimum_battery_reserve": "Battery energy must stay at or above the stated level in those hours.",
    "no_charge_window": "Battery charging is unavailable during the stated hours.",
    "no_discharge_window": "Battery discharging is unavailable during the stated hours.",
    "max_grid_window": "Grid import is capped during the stated hours.",
    "no_op": "This note does not affect today's 24-hour energy schedule.",
}

MAX_EXPLANATION = 200


@dataclass
class NormalizedInterpretation:
    entries: list[dict[str, Any]]
    #: what the guardrails had to correct, for server-side logging only
    repairs: list[str] = field(default_factory=list)


def _no_op_entry(index: int) -> dict[str, Any]:
    return {
        "note_index": index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": _EXPLANATIONS["no_op"],
    }


def canonical_type(raw: Any) -> str | None:
    """Case- and separator-insensitive directive type, or None if unknown."""
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if key in ALLOWED_TYPES:
        return key
    return _TYPE_ALIASES.get(key) or _TYPE_ALIASES.get(key.replace("_", ""))


def _clean_explanation(raw: Any, directive_type: str) -> str:
    if isinstance(raw, str) and raw.strip():
        text = " ".join(raw.split())
        if len(text) > MAX_EXPLANATION:
            text = text[: MAX_EXPLANATION - 1].rstrip() + "…"
        return text
    return _EXPLANATIONS[directive_type]


def _index_entries(raw_entries: Any, note_count: int) -> dict[int, Mapping[str, Any]]:
    """Map note_index -> entry, dropping duplicates and out-of-range indexes."""
    indexed: dict[int, Mapping[str, Any]] = {}
    if not isinstance(raw_entries, (list, tuple)):
        return indexed
    for position, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping):
            continue
        raw_index = entry.get("note_index", position)
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = position
        if 0 <= index < note_count and index not in indexed:
            indexed[index] = entry
    return indexed


def normalize_interpretation(
    raw_entries: Any,
    notes: Sequence[str],
    capacity_kwh: float,
) -> NormalizedInterpretation:
    """Turn raw model output into exactly one valid entry per note."""
    indexed = _index_entries(raw_entries, len(notes))
    repairs: list[str] = []
    entries: list[dict[str, Any]] = []

    for index, note in enumerate(notes):
        entry = indexed.get(index)
        if entry is None:
            repairs.append(f"note {index}: missing entry, synthesised no_op")
            entries.append(_no_op_entry(index))
            continue

        directive_type = canonical_type(entry.get("directive_type"))
        if directive_type is None:
            repairs.append(
                f"note {index}: unknown directive_type {entry.get('directive_type')!r}"
            )
            entries.append(_no_op_entry(index))
            continue

        if directive_type == "no_op":
            if entry.get("applies") is True:
                repairs.append(f"note {index}: no_op forced to applies=false")
            entries.append(
                {
                    **_no_op_entry(index),
                    "explanation": _clean_explanation(entry.get("explanation"), "no_op"),
                }
            )
            continue

        adjustment, note_repairs = _normalize_adjustment(
            directive_type, entry.get("structured_adjustment"), note, index, capacity_kwh
        )
        repairs.extend(note_repairs)

        if adjustment is None:
            repairs.append(f"note {index}: {directive_type} unrecoverable, downgraded to no_op")
            entries.append(_no_op_entry(index))
            continue

        if entry.get("applies") is not True:
            repairs.append(f"note {index}: {directive_type} forced to applies=true")

        entries.append(
            {
                "note_index": index,
                "applies": True,
                "directive_type": directive_type,
                "structured_adjustment": adjustment,
                "explanation": _clean_explanation(entry.get("explanation"), directive_type),
            }
        )

    return NormalizedInterpretation(entries=entries, repairs=repairs)


def _normalize_adjustment(
    directive_type: str,
    raw: Any,
    note: str,
    index: int,
    capacity_kwh: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize one structured_adjustment, or return None if unrecoverable."""
    repairs: list[str] = []
    payload: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}

    # --- hours: deterministic arbitration (section 8.2) ---
    llm_hours = normalize_hours(payload.get("hours"))
    parsed = parse_hour_windows(note)
    if parsed.decisive and parsed.hours:
        if parsed.hours != llm_hours:
            repairs.append(
                f"note {index}: hours {llm_hours} -> {parsed.hours} from the note text"
            )
        hours = parsed.hours
    else:
        hours = llm_hours

    if not hours:
        return None, repairs

    adjustment: dict[str, Any] = {"hours": hours}

    # --- type-specific numerics (section 8.3) ---
    if directive_type == "solar_reduction":
        factor = normalize_factor(payload.get("factor"), note)
        if factor is None:
            return None, repairs
        if payload.get("factor") is not None and abs(float(payload["factor"]) - factor) > 1e-6:
            repairs.append(f"note {index}: factor {payload['factor']} -> {factor}")
        adjustment["factor"] = factor

    elif directive_type == "minimum_battery_reserve":
        reserve = normalize_reserve(payload.get("minimum_energy_kwh"), note, capacity_kwh)
        if reserve is None:
            return None, repairs
        adjustment["minimum_energy_kwh"] = reserve

    elif directive_type == "max_grid_window":
        cap = normalize_grid_cap(payload.get("max_grid_kwh"), note)
        if cap is None:
            return None, repairs
        adjustment["max_grid_kwh"] = cap

    # no_charge_window / no_discharge_window carry hours only; any extra key the
    # model added has already been dropped by rebuilding the dict from scratch.
    return adjustment, repairs
