"""Deterministic hour-window parser used for arbitration (PROJECT.md 8.2).

This is the single most valuable guardrail: the most common LLM error on this
task is an off-by-one on the end-exclusive window convention, and a regex-level
reading of the note fixes it without ever pattern-matching an *answer*.

Arbitration policy (section 8.2):

* exactly one unambiguous window found -> use it, keep the LLM's directive_type
* nothing found, or more than one candidate -> trust the LLM's hours

Windows are start-inclusive and end-exclusive throughout: "1 PM to 3 PM" is
``[13, 14]``. Windows that wrap past midnight are split and sorted ascending.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HOURS = 24

_WORD_NUMBERS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

# A clock reading: "noon", "midnight", "3", "3 PM", "15:00", "3:30pm".
# Units are excluded so "155 kWh" and "80 %" are never read as times.
_TIME = (
    r"(?:noon|midday|midnight|"
    r"\d{1,2}(?::\d{2})?(?:\s*[ap]\.?\s?m\.?)?(?!\s*(?:kwh|kw|mwh|%|percent|bdt|taka)))"
)

_CONNECTOR = r"(?:-|‐|‑|‒|–|—|\bto\b|\buntil\b|\btil{1,2}\b|\bthrough\b|\bthru\b)"

_RANGE_RE = re.compile(
    rf"\b(?:from\s+|starting\s+(?:at|from)\s+)?({_TIME})\s*{_CONNECTOR}\s*({_TIME})",
    re.IGNORECASE,
)
_BETWEEN_RE = re.compile(
    rf"\bbetween\s+({_TIME})\s+and\s+({_TIME})", re.IGNORECASE
)
_SINGLE_RE = re.compile(
    rf"\b(?:at|during\s+the|for\s+the|in\s+the)\s+({_TIME})(?:\s*hour)?", re.IGNORECASE
)

_COUNT = r"(?:\d{1,2}|" + "|".join(_WORD_NUMBERS) + r")"
_FOR_THEN_START_RE = re.compile(
    rf"\bfor\s+({_COUNT})\s+hours?\s+(?:starting|beginning|from)\s*(?:at|from)?\s*({_TIME})",
    re.IGNORECASE,
)
_START_THEN_FOR_RE = re.compile(
    rf"\b(?:starting|beginning)\s+(?:at|from)\s+({_TIME})\s+for\s+({_COUNT})\s+hours?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WindowParse:
    """What the deterministic reading of a note found."""

    #: the parsed hours, or None when the parser declines to rule
    hours: list[int] | None
    #: True when several conflicting windows were found
    ambiguous: bool
    #: which pattern produced the answer, for logging
    source: str

    @property
    def decisive(self) -> bool:
        """Whether arbitration should override the LLM with this reading."""
        return self.hours is not None and not self.ambiguous


def _resolve(token: str) -> tuple[int | None, bool]:
    """Resolve one clock token to an hour. Returns ``(hour, has_explicit_marker)``.

    ``has_explicit_marker`` records whether the token carried an am/pm, a
    ``:MM``, or the words noon/midnight - a bare number is too weak on its own to
    be trusted as a time.
    """
    text = token.strip().lower()
    if text in {"noon", "midday"}:
        return 12, True
    if text == "midnight":
        return 0, True

    match = re.fullmatch(
        r"(\d{1,2})(?::(\d{2}))?\s*([ap])?\.?\s?(m)?\.?", text, re.IGNORECASE
    )
    if not match:
        return None, False
    hour = int(match.group(1))
    has_minutes = match.group(2) is not None
    meridiem = match.group(3)

    if meridiem:
        if not 1 <= hour <= 12:
            return None, False
        hour = hour % 12 + (12 if meridiem == "p" else 0)
        return hour, True

    if not 0 <= hour <= 23:
        return None, False
    return hour, has_minutes


def _expand(start: int, end: int) -> list[int] | None:
    """Start-inclusive, end-exclusive expansion, wrapping past midnight."""
    if start == end:
        return None  # a zero-length or whole-day window is not a usable reading
    if end > start:
        return list(range(start, end))
    return sorted(list(range(start, HOURS)) + list(range(0, end)))


def _read_pair(raw_start: str, raw_end: str) -> list[int] | None:
    start, start_marked = _resolve(raw_start)
    end, end_marked = _resolve(raw_end)
    if start is None or end is None:
        return None
    # At least one endpoint must look like a real clock reading, so phrases like
    # "between 40 and 60" are never mistaken for a window.
    if not (start_marked or end_marked):
        return None

    # "from 6 until 10 PM": an unmarked start inherits the end's half of the day.
    if not start_marked and end_marked and end >= 12 and start < 12:
        candidate = start + 12
        if candidate < end:
            start = candidate
    # "from 11 AM until 2": an unmarked end that lands before the start is the
    # afternoon reading.
    elif start_marked and not end_marked and end <= start and end < 12:
        end += 12

    return _expand(start, end)


def parse_hour_windows(note: str) -> WindowParse:
    """Read the hour window a note describes, or decline."""
    text = note or ""

    duration = _parse_duration(text)
    if duration is not None:
        return WindowParse(duration, False, "duration")

    candidates: list[list[int]] = []
    for pattern, label in ((_BETWEEN_RE, "between"), (_RANGE_RE, "range")):
        for match in pattern.finditer(text):
            hours = _read_pair(match.group(1), match.group(2))
            if hours and hours not in candidates:
                candidates.append(hours)

    if len(candidates) == 1:
        return WindowParse(candidates[0], False, "range")
    if len(candidates) > 1:
        return WindowParse(None, True, "ambiguous")

    for match in _SINGLE_RE.finditer(text):
        hour, marked = _resolve(match.group(1))
        if hour is not None and marked:
            return WindowParse([hour], False, "single")

    return WindowParse(None, False, "none")


def _parse_duration(text: str) -> list[int] | None:
    """"for N hours starting at X" -> X, X+1, ... X+N-1."""
    for pattern, flipped in ((_FOR_THEN_START_RE, False), (_START_THEN_FOR_RE, True)):
        match = pattern.search(text)
        if not match:
            continue
        raw_count = match.group(2 if flipped else 1)
        raw_start = match.group(1 if flipped else 2)
        count = _WORD_NUMBERS.get(raw_count.lower()) or (
            int(raw_count) if raw_count.isdigit() else None
        )
        start, marked = _resolve(raw_start)
        if count is None or start is None or not marked or not 1 <= count <= HOURS:
            continue
        return sorted({(start + offset) % HOURS for offset in range(count)})
    return None


def normalize_hours(raw: object) -> list[int]:
    """Cast to int, drop out-of-range, dedupe, sort ascending (section 8.2)."""
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: set[int] = set()
    for item in raw:
        if isinstance(item, bool):
            continue
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= value < HOURS:
            out.add(value)
    return sorted(out)
