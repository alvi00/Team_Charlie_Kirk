"""Rule-based interpreter used ONLY when every LLM attempt has failed (7.5).

This is a safety net, not the main path. The LLM stays the primary interpreter -
that is the mandatory challenge requirement, and using rules as the *sole*
interpreter is explicitly non-compliant. This module exists so a provider outage
degrades to a valid 200 response instead of a 500.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .numbers import read_factor, read_grid_cap, read_reserve
from .timeparse import parse_hour_windows

#: "charging"/"charger"/"charge point" followed, within a few filler words, by an
#: outage word - covers "the charging circuit will be unavailable", "chargers are
#: locked out", "the battery charger will be isolated".
_NO_CHARGE_RE = re.compile(
    r"(?:\bdo(?:\s+not|n'?t)\s+charg|\b(?:avoid|refrain\s+from|cease|stop|suspend|halt)\s+"
    r"(?:\w+\s+){0,2}?charg|\bno\s+(?:\w+\s+){0,2}?charging\b|\bnot\s+be\s+charged\b|"
    r"charg(?:e|er|ers|ing|e\s+point)\w*(?:\s+\w+){0,4}?\s+(?:un)?(?:available|disabled|offline|"
    r"isolated|suspended|locked|prohibited|blocked|down|out\s+of\s+service))",
    re.IGNORECASE,
)
_NO_DISCHARGE_RE = re.compile(
    r"(?:\bdo(?:\s+not|n'?t)\s+discharg|\b(?:avoid|refrain\s+from|cease|stop|halt)\s+"
    r"(?:\w+\s+){0,2}?discharg|\bno\s+(?:\w+\s+){0,2}?discharging\b|\bmust\s+not\s+discharg|"
    r"\bnot\s+be\s+discharged\b|discharg\w*\s+(?:is\s+|are\s+)?(?:un)?(?:available|disabled|"
    r"prohibited|blocked)|\bnot\s+be\s+drawn\s+down\b)",
    re.IGNORECASE,
)
_RESERVE_RE = re.compile(
    r"(?:\bat\s+least\b|\bno\s+less\s+than\b|\bno\s+lower\s+than\b|\bminimum\b|\bfloor\s+of\b|"
    r"\bkeep\b|\bhold\b|\breserve\b|\bremain\s+in\s+the\s+battery\b|\bstay\s+(?:above|at\s+or\s+above)\b|"
    r"\bnot\s+(?:drop|fall)\s+below\b|\bmaintain\b)",
    re.IGNORECASE,
)
_GRID_RE = re.compile(
    r"(?:grid\s+(?:import|intake|draw|supply)|import\s+from\s+the\s+grid|campus\s+grid|"
    r"\bimport\b|\bintake\b|feeder|transformer|substation)",
    re.IGNORECASE,
)
_SOLAR_RE = re.compile(r"(?:\bsolar\b|\bpv\b|\bphotovoltaic\b|\bpanel|\brooftop\b|\barray\b)", re.IGNORECASE)
_BATTERY_RE = re.compile(r"(?:\bbattery\b|\bpack\b|\bstorage\b|\bbess\b)", re.IGNORECASE)
_KWH_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:kwh|kw)\b", re.IGNORECASE)

_EXPLANATIONS = {
    "solar_reduction": "Usable solar is reduced during the stated hours.",
    "minimum_battery_reserve": "Battery energy must stay at or above the stated level in those hours.",
    "no_charge_window": "Battery charging is unavailable during the stated hours.",
    "no_discharge_window": "Battery discharging is unavailable during the stated hours.",
    "max_grid_window": "Grid import is capped during the stated hours.",
    "no_op": "This note does not affect today's 24-hour energy schedule.",
}


def _no_op(index: int) -> dict[str, Any]:
    return {
        "note_index": index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": _EXPLANATIONS["no_op"],
    }


def _classify(note: str) -> str:
    """Best-effort directive type for one note."""
    if _NO_DISCHARGE_RE.search(note):
        return "no_discharge_window"
    if _NO_CHARGE_RE.search(note):
        return "no_charge_window"
    if _SOLAR_RE.search(note) and read_factor(note).decisive:
        return "solar_reduction"
    if _GRID_RE.search(note) and read_grid_cap(note).decisive:
        return "max_grid_window"
    if _BATTERY_RE.search(note) and _RESERVE_RE.search(note) and _KWH_RE.search(note):
        return "minimum_battery_reserve"
    if _BATTERY_RE.search(note) and _RESERVE_RE.search(note) and "%" in note:
        return "minimum_battery_reserve"
    return "no_op"


def interpret_rule_based(
    notes: Sequence[str], capacity_kwh: float
) -> list[dict[str, Any]]:
    """One best-effort entry per note, in ``note_index`` order."""
    entries: list[dict[str, Any]] = []
    for index, note in enumerate(notes):
        text = note or ""
        directive_type = _classify(text)
        if directive_type == "no_op":
            entries.append(_no_op(index))
            continue

        window = parse_hour_windows(text)
        hours = window.hours if window.hours else []
        if not hours:
            entries.append(_no_op(index))
            continue

        adjustment: dict[str, Any] = {"hours": hours}
        if directive_type == "solar_reduction":
            reading = read_factor(text)
            if reading.value is None:
                entries.append(_no_op(index))
                continue
            adjustment["factor"] = reading.value
        elif directive_type == "minimum_battery_reserve":
            reading = read_reserve(text, capacity_kwh)
            value = reading.value
            if value is None:
                match = _KWH_RE.search(text)
                value = float(re.sub(r"[^\d.]", "", match.group(0))) if match else None
            if value is None:
                entries.append(_no_op(index))
                continue
            adjustment["minimum_energy_kwh"] = min(max(0.0, value), capacity_kwh)
        elif directive_type == "max_grid_window":
            reading = read_grid_cap(text)
            if reading.value is None:
                entries.append(_no_op(index))
                continue
            adjustment["max_grid_kwh"] = max(0.0, reading.value)

        entries.append(
            {
                "note_index": index,
                "applies": True,
                "directive_type": directive_type,
                "structured_adjustment": adjustment,
                "explanation": _EXPLANATIONS[directive_type],
            }
        )
    return entries
