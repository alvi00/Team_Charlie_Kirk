"""Numeric normalization for directive adjustments (PROJECT.md 8.3).

Two readings of a note exist for every number: the LLM's, and a deterministic one
taken straight from the text. Where the deterministic reading is unambiguous it
wins - that is what stops the single most expensive interpretation error on this
task, inverting ``factor`` on "an 80% reduction".

``factor`` is always the FRACTION OF SOLAR THAT REMAINS USABLE.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

#: word fractions, as the fraction the words name (not yet remaining-vs-removed)
_WORD_FRACTIONS: dict[str, float] = {
    "half": 0.5,
    "a half": 0.5,
    "one half": 0.5,
    "one-half": 0.5,
    "a third": 1.0 / 3.0,
    "one third": 1.0 / 3.0,
    "one-third": 1.0 / 3.0,
    "two thirds": 2.0 / 3.0,
    "two-thirds": 2.0 / 3.0,
    "a quarter": 0.25,
    "one quarter": 0.25,
    "one-quarter": 0.25,
    "three quarters": 0.75,
    "three-quarters": 0.75,
    "a fifth": 0.2,
    "one fifth": 0.2,
    "one-fifth": 0.2,
    "a tenth": 0.1,
    "one tenth": 0.1,
    "one-tenth": 0.1,
}

_NUM = r"(\d{1,3}(?:\.\d+)?)"

# "an 80% reduction", "reduced by 80%", "down 80 percent", "80% lower"
_REMOVED_PATTERNS = [
    re.compile(rf"{_NUM}\s*(?:%|percent)\s*(?:\w+\s+){{0,2}}?(?:reduction|drop|decline|decrease|cut|loss)", re.I),
    re.compile(rf"(?:reduc\w*|drop\w*|fall\w*|decreas\w*|declin\w*|down|cut|lower\w*|less)\s+(?:by\s+)?(?:about\s+|roughly\s+|around\s+|approximately\s+|some\s+)?{_NUM}\s*(?:%|percent)", re.I),
    re.compile(rf"{_NUM}\s*(?:%|percent)\s+(?:lower|less|below|off)", re.I),
]

# "drops to about 20%", "20% of the forecast", "operating at 25%"
_REMAINING_PATTERNS = [
    re.compile(rf"(?:to|at)\s+(?:about\s+|roughly\s+|around\s+|approximately\s+|only\s+|just\s+)?{_NUM}\s*(?:%|percent)", re.I),
    re.compile(rf"{_NUM}\s*(?:%|percent)\s+of\b", re.I),
]

#: "reduced by half", and also "cut array output by half" - the verb and the
#: "by <fraction>" are often separated by the thing being reduced.
_WORD_REMOVED_RE = re.compile(
    r"(?:reduc\w*|drop\w*|fall\w*|decreas\w*|declin\w*|down|cut\w*|lower\w*|less|trim\w*)"
    r"(?:\s+\w+){0,3}?\s+by\s+"
    r"(?:about\s+|roughly\s+|around\s+)?(" + "|".join(sorted(_WORD_FRACTIONS, key=len, reverse=True)) + r")\b",
    re.I,
)
_WORD_REMAINING_RE = re.compile(
    r"\b(?:about\s+|roughly\s+|around\s+|only\s+|just\s+)?("
    + "|".join(sorted(_WORD_FRACTIONS, key=len, reverse=True))
    + r")\s+(?:of\s+|the\s+)?(?:of\s+)?"
    r"(?:its\s+|their\s+|the\s+)?(?:normal|usual|forecast|expected|typical|rated|predicted|projected)",
    re.I,
)

# a percentage tied to battery capacity, e.g. "50% of the battery capacity"
_PERCENT_OF_CAPACITY_RE = re.compile(
    rf"{_NUM}\s*(?:%|percent)\s*(?:of\s+)?(?:the\s+|its\s+)?"
    r"(?:battery\s+|pack\s+|usable\s+|total\s+|rated\s+|nominal\s+)*"
    r"(?:capacity|battery|pack|storage|pack\s+capacity)",
    re.I,
)
_ANY_PERCENT_RE = re.compile(rf"{_NUM}\s*(?:%|percent)", re.I)


@dataclass(frozen=True)
class Reading:
    """A deterministic reading of a number from the note text."""

    value: float | None
    #: True when the text supports exactly one interpretation
    decisive: bool
    source: str = "none"


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ---------------------------------------------------------------------------
# solar_reduction factor
# ---------------------------------------------------------------------------


def read_factor(note: str) -> Reading:
    """Read the remaining-usable-solar fraction directly from the note."""
    text = note or ""

    for pattern in _REMOVED_PATTERNS:
        match = pattern.search(text)
        if match:
            removed = float(match.group(1)) / 100.0
            return Reading(_clamp01(1.0 - removed), True, "percent_removed")

    match = _WORD_REMOVED_RE.search(text)
    if match:
        removed = _WORD_FRACTIONS[match.group(1).lower()]
        return Reading(_clamp01(1.0 - removed), True, "word_removed")

    for pattern in _REMAINING_PATTERNS:
        match = pattern.search(text)
        if match:
            return Reading(_clamp01(float(match.group(1)) / 100.0), True, "percent_remaining")

    match = _WORD_REMAINING_RE.search(text)
    if match:
        return Reading(_clamp01(_WORD_FRACTIONS[match.group(1).lower()]), True, "word_remaining")

    return Reading(None, False)


def normalize_factor(raw: object, note: str) -> float | None:
    """Final factor: deterministic reading wins, else the LLM value, cleaned up."""
    deterministic = read_factor(note)
    if deterministic.decisive and deterministic.value is not None:
        return deterministic.value

    value = _finite(raw)
    if value is None:
        return None
    # a model that answered "80" meaning 80% rather than 0.8
    if 1.0 < value <= 100.0:
        value /= 100.0
    return _clamp01(value)


def _clamp01(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 6)


# ---------------------------------------------------------------------------
# minimum_battery_reserve
# ---------------------------------------------------------------------------


def read_reserve(note: str, capacity_kwh: float) -> Reading:
    """Resolve "50% of the battery" against the scenario's capacity."""
    match = _PERCENT_OF_CAPACITY_RE.search(note or "")
    if match:
        return Reading(float(match.group(1)) / 100.0 * capacity_kwh, True, "percent_of_capacity")
    return Reading(None, False)


def normalize_reserve(raw: object, note: str, capacity_kwh: float) -> float | None:
    """Absolute kWh floor, clamped into ``[0, capacity]`` (section 8.3)."""
    deterministic = read_reserve(note, capacity_kwh)
    if deterministic.decisive and deterministic.value is not None:
        return _clamp(deterministic.value, capacity_kwh)

    value = _finite(raw)
    if value is None:
        return None

    # The model returned the bare percent ("50") where the note says "50% of the
    # battery" - convert it rather than accepting a nonsensical 50 kWh floor.
    percent = _ANY_PERCENT_RE.search(note or "")
    if percent and abs(value - float(percent.group(1))) < 1e-6 and value <= 100:
        return _clamp(value / 100.0 * capacity_kwh, capacity_kwh)

    return _clamp(value, capacity_kwh)


def _clamp(value: float, capacity_kwh: float) -> float:
    return round(min(max(0.0, value), max(0.0, capacity_kwh)), 6)


# ---------------------------------------------------------------------------
# max_grid_window
# ---------------------------------------------------------------------------

_GRID_CAP_RE = re.compile(
    r"(?:exceed|above|below|over|under|beyond|limit(?:ed)?\s+(?:to|of|is)|limit\s+is|"
    # "cap at 140", and also "cap campus import at 140"
    r"cap(?:s|ped|ping)?(?:\s+\w+){0,3}?\s+(?:at|to)|"
    r"maximum\s+(?:of|is)|max\s+of|more\s+than|at\s+or\s+below|"
    r"no\s+more\s+than|stay\s+(?:under|below)|ceiling\s+of)"
    r"\s*(?:about\s+|roughly\s+|around\s+)?(\d{1,6}(?:\.\d+)?)\s*(?:kwh|kw)?",
    re.I,
)


def read_grid_cap(note: str) -> Reading:
    match = _GRID_CAP_RE.search(note or "")
    if match:
        return Reading(float(match.group(1)), True, "grid_cap")
    return Reading(None, False)


def normalize_grid_cap(raw: object, note: str) -> float | None:
    """Per-hour import cap in kWh: finite and non-negative (section 8.3)."""
    value = _finite(raw)
    if value is None or value < 0:
        deterministic = read_grid_cap(note)
        if deterministic.decisive and deterministic.value is not None:
            return round(max(0.0, deterministic.value), 6)
        return None
    return round(max(0.0, value), 6)
