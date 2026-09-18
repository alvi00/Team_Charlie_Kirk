"""System prompt, response schema, few-shot examples and payload builder (7.2-7.4).

The few-shot wording is deliberately paraphrased rather than lifted from the
public sample pack: hidden notes paraphrase the same directives, so the examples
teach the pattern instead of memorising the cases. Hard-coding public note
wording is explicitly non-compliant.

Examples are chosen to span the *failure modes*, not just the directive types:
end-exclusive boundaries, reduction-vs-remaining percentages, percent-of-capacity
reserves, windows that wrap past midnight, durations, single-hour mentions,
multi-note scenarios, and distractors that must stay ``no_op``.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

DIRECTIVE_TYPES = [
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

SYSTEM_PROMPT = """You convert campus energy operator notes into structured directives for an optimizer.
You output JSON only. No prose, no markdown, no code fences.

The schedule covers hours 0..23 of one day (0 = midnight, 12 = noon, 23 = 11 PM).

Allowed directive_type values - nothing else is ever permitted:
  solar_reduction         : usable solar is reduced during specific hours
  minimum_battery_reserve : battery stored energy must stay at or above a level
  no_charge_window        : the battery cannot be charged during specific hours
  no_discharge_window     : the battery cannot be discharged during specific hours
  max_grid_window         : grid import is capped during specific hours
  no_op                   : the note does not affect this 24-hour energy schedule

TIME RULES (critical):
- Windows are start-inclusive and end-exclusive.
  "1 PM to 3 PM" -> [13,14].  "from 6 PM until 10 PM" -> [18,19,20,21].
  "between 11 AM and 2 PM" -> [11,12,13].  "2 AM until 5 AM" -> [2,3,4].
  "6 PM until 9 PM" -> [18,19,20] (NOT 21). The end hour is never included.
- "noon" = 12, "midnight" = 0, "13:00" = 13, "9 AM" = 9, "9 PM" = 21.
- A single hour mention ("at 3 PM", "during the 3 PM hour") -> [15].
- "for N hours starting at X" -> X, X+1, ... X+N-1.
- Windows that wrap past midnight are split and sorted ascending:
  "10 PM to 2 AM" -> [0,1,22,23].
- "morning"/"afternoon"/"evening" alone, with no clock time, is too vague:
  use no_op unless the note also gives explicit hours.
- hours must be unique integers 0..23 in ascending order.

NUMERIC RULES (critical):
- solar_reduction "factor" is the FRACTION OF SOLAR THAT REMAINS USABLE.
  "drops to about 20%"            -> factor 0.2
  "an 80% reduction"              -> factor 0.2
  "reduced by 70%"                -> factor 0.3
  "output 40% lower"              -> factor 0.6
  "roughly one-fifth of normal"   -> factor 0.2
  "about half the forecast"       -> factor 0.5
  "cut by half"                   -> factor 0.5
  "treated as roughly 25%"        -> factor 0.25
  Read carefully: "to X%" means X% REMAINS; "by X%" means X% is LOST.
  factor is between 0 and 1 inclusive.
- minimum_battery_reserve "minimum_energy_kwh" is an absolute kWh number.
  If the note states a percentage of battery capacity, multiply it by the
  capacity_kwh given in the scenario context. 50% of a 200 kWh battery -> 100.
- max_grid_window "max_grid_kwh" is the per-hour cap in kWh.

RELEVANCE RULES:
- A note is no_op when it does not change today's 24-hour electricity schedule:
  administrative notices, room bookings, deadlines, menus, events next week,
  announcements, staffing, anything about a different day.
- A note saying conditions are NORMAL or UNCHANGED is no_op - there is nothing
  to constrain.
- Mark a note no_op if it cannot be expressed as one of the five real directive
  types, even if it is about energy (for example a tariff complaint, a request
  to "save money", or a note about equipment that is working fine).
- Never invent demand, solar, tariff, or battery values.
- Never invent a directive type that is not in the list.
- Judge each note independently. One note produces exactly one entry.

OUTPUT SHAPE - exactly one entry per note, in note_index order 0..N-1.
Set the fields that do not apply to the directive type to null:
  solar_reduction         -> hours + factor
  minimum_battery_reserve -> hours + minimum_energy_kwh
  no_charge_window        -> hours only
  no_discharge_window     -> hours only
  max_grid_window         -> hours + max_grid_kwh
  no_op                   -> structured_adjustment is null

applies is false only for no_op. no_op always has structured_adjustment null.
Every other directive has applies true."""


#: Strict structured-output schema. The adjustment is kept flat with nullable
#: members because strict mode requires every property to be listed as required;
#: the deterministic guardrails then rebuild the per-type shape, dropping the
#: members that do not belong to the chosen directive_type.
RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "gridwise_directives",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["directives"],
        "properties": {
            "directives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "note_index",
                        "applies",
                        "directive_type",
                        "structured_adjustment",
                        "explanation",
                    ],
                    "properties": {
                        "note_index": {"type": "integer"},
                        "applies": {"type": "boolean"},
                        "directive_type": {"type": "string", "enum": DIRECTIVE_TYPES},
                        "structured_adjustment": {
                            "type": ["object", "null"],
                            "additionalProperties": False,
                            "required": [
                                "hours",
                                "factor",
                                "minimum_energy_kwh",
                                "max_grid_kwh",
                            ],
                            "properties": {
                                "hours": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                "factor": {"type": ["number", "null"]},
                                "minimum_energy_kwh": {"type": ["number", "null"]},
                                "max_grid_kwh": {"type": ["number", "null"]},
                            },
                        },
                        "explanation": {"type": "string"},
                    },
                },
            }
        },
    },
}


def _adjustment(
    hours: list[int] | None = None,
    factor: float | None = None,
    minimum_energy_kwh: float | None = None,
    max_grid_kwh: float | None = None,
) -> dict[str, Any] | None:
    if hours is None:
        return None
    return {
        "hours": hours,
        "factor": factor,
        "minimum_energy_kwh": minimum_energy_kwh,
        "max_grid_kwh": max_grid_kwh,
    }


def _entry(
    index: int, directive_type: str, adjustment: dict[str, Any] | None, explanation: str
) -> dict[str, Any]:
    return {
        "note_index": index,
        "applies": directive_type != "no_op",
        "directive_type": directive_type,
        "structured_adjustment": adjustment,
        "explanation": explanation,
    }


#: (battery, notes, reply). The user half is rendered through
#: ``build_user_message`` so a few-shot turn is byte-identical in format to the
#: real request - a format drift between the two is enough to cost a directive.
FEW_SHOT: list[tuple[dict[str, float], list[str], dict[str, Any]]] = [
    # 1. reduction wording (inversion) + a distractor
    (
        {"capacity_kwh": 300, "minimum_energy_kwh": 50,
         "max_charge_kwh_per_hour": 70, "max_discharge_kwh_per_hour": 70},
        ["Expect rooftop output to fall by 70% from 09:00 to 12:00 during the array inspection.",
         "Lecture theatre 3 is booked for a departmental seminar on Thursday."],
        {"directives": [
            _entry(0, "solar_reduction", _adjustment(hours=[9, 10, 11], factor=0.3),
                   "A 70% fall leaves 30% of forecast solar usable."),
            _entry(1, "no_op", None,
                   "A room booking on another day does not change today's schedule."),
        ]},
    ),
    # 2. percent-of-capacity reserve, end-exclusive boundary
    (
        {"capacity_kwh": 250, "minimum_energy_kwh": 40,
         "max_charge_kwh_per_hour": 60, "max_discharge_kwh_per_hour": 60},
        ["Hold no less than 60% of pack capacity in the battery from 7 PM until 10 PM."],
        {"directives": [
            _entry(0, "minimum_battery_reserve",
                   _adjustment(hours=[19, 20, 21], minimum_energy_kwh=150),
                   "60% of the 250 kWh pack is a 150 kWh floor."),
        ]},
    ),
    # 3. charger outage phrased as equipment status
    (
        {"capacity_kwh": 180, "minimum_energy_kwh": 30,
         "max_charge_kwh_per_hour": 45, "max_discharge_kwh_per_hour": 45},
        ["Chargers are locked out for switchgear work from 1 AM through 4 AM."],
        {"directives": [
            _entry(0, "no_charge_window", _adjustment(hours=[1, 2, 3]),
                   "Charging is unavailable while the chargers are locked out."),
        ]},
    ),
    # 4. no-discharge phrased indirectly ("drawn down")
    (
        {"capacity_kwh": 220, "minimum_energy_kwh": 35,
         "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50},
        ["During protection relay checks between 4 PM and 6 PM the pack must not be drawn down."],
        {"directives": [
            _entry(0, "no_discharge_window", _adjustment(hours=[16, 17]),
                   "The battery cannot discharge during the relay checks."),
        ]},
    ),
    # 5. grid cap + distractor, two notes
    (
        {"capacity_kwh": 240, "minimum_energy_kwh": 40,
         "max_charge_kwh_per_hour": 55, "max_discharge_kwh_per_hour": 55},
        ["Feeder works cap campus import at 140 kWh per hour from 8 PM until 11 PM.",
         "The canteen will trial a new breakfast menu starting next Monday."],
        {"directives": [
            _entry(0, "max_grid_window", _adjustment(hours=[20, 21, 22], max_grid_kwh=140),
                   "Grid import is limited to 140 kWh in each of those hours."),
            _entry(1, "no_op", None,
                   "A catering change next week does not affect today's schedule."),
        ]},
    ),
    # 6. window wrapping past midnight + a duration + a "nothing changes" note
    (
        {"capacity_kwh": 200, "minimum_energy_kwh": 40,
         "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50},
        ["Overnight from 10 PM to 2 AM, do not discharge the battery.",
         "Keep at least 120 kWh stored for four hours starting at 6 PM.",
         "Solar output is expected to be completely normal today."],
        {"directives": [
            _entry(0, "no_discharge_window", _adjustment(hours=[0, 1, 22, 23]),
                   "The overnight window wraps past midnight into hours 0 and 1."),
            _entry(1, "minimum_battery_reserve",
                   _adjustment(hours=[18, 19, 20, 21], minimum_energy_kwh=120),
                   "Four hours from 6 PM covers hours 18 through 21."),
            _entry(2, "no_op", None,
                   "Normal solar imposes no constraint on the schedule."),
        ]},
    ),
    # 7. single-hour mention + "to X%" remaining wording
    (
        {"capacity_kwh": 210, "minimum_energy_kwh": 45,
         "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50},
        ["During the 3 PM hour the battery must not be charged.",
         "Cloud cover brings panel output down to roughly 15% between 11:00 and 13:00."],
        {"directives": [
            _entry(0, "no_charge_window", _adjustment(hours=[15]),
                   "A single hour mention covers only hour 15."),
            _entry(1, "solar_reduction", _adjustment(hours=[11, 12], factor=0.15),
                   "Output falls to 15% of forecast, so 15% remains usable."),
        ]},
    ),
    # 8. energy-related but not expressible -> no_op
    (
        {"capacity_kwh": 200, "minimum_energy_kwh": 40,
         "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50},
        ["Electricity prices have risen again this quarter; please keep costs down where possible.",
         "The backup generator passed its annual inspection yesterday."],
        {"directives": [
            _entry(0, "no_op", None,
                   "A general cost request sets no specific schedule constraint."),
            _entry(1, "no_op", None,
                   "A completed inspection imposes no constraint on today's schedule."),
        ]},
    ),
]


def build_user_message(notes: Sequence[str], battery: Any) -> str:
    """Numbered notes plus the battery context needed for percentage resolution.

    The 24 hourly rows are deliberately left out (section 7.3): the model does not
    need them, they cost latency, and they invite hallucinated demand/tariff edits.
    """
    lines = [
        f"Battery: capacity_kwh={_num(_get(battery, 'capacity_kwh'))}, "
        f"minimum_energy_kwh={_num(_get(battery, 'minimum_energy_kwh'))}, "
        f"max_charge_kwh_per_hour={_num(_get(battery, 'max_charge_kwh_per_hour'))}, "
        f"max_discharge_kwh_per_hour={_num(_get(battery, 'max_discharge_kwh_per_hour'))}",
        "Operator notes:",
    ]
    for index, note in enumerate(notes):
        lines.append(f"[{index}] {note.strip()}")
    lines.append(f"Return exactly {len(notes)} directive entries as JSON only.")
    return "\n".join(lines)


def build_messages(notes: Sequence[str], battery: Any) -> list[dict[str, str]]:
    """System prompt + few-shot turns + the real request."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot_battery, shot_notes, reply in FEW_SHOT:
        messages.append(
            {"role": "user", "content": build_user_message(shot_notes, shot_battery)}
        )
        messages.append(
            {"role": "assistant", "content": json.dumps(reply, separators=(",", ":"))}
        )
    messages.append({"role": "user", "content": build_user_message(notes, battery)})
    return messages


def build_repair_messages(
    notes: Sequence[str],
    battery: Any,
    bad_output: str,
    error_text: str,
) -> list[dict[str, str]]:
    """One repair turn carrying the validator's complaint (section 7.5)."""
    messages = build_messages(notes, battery)
    messages.append({"role": "assistant", "content": bad_output[:2000]})
    messages.append(
        {
            "role": "user",
            "content": (
                f"That response was rejected: {error_text[:500]}\n"
                "Return the corrected result as JSON only - a single object with a "
                f'"directives" array holding exactly {len(notes)} entries, one per '
                "note, in note_index order starting at 0. No prose, no markdown, "
                "no code fences."
            ),
        }
    )
    return messages


def _get(battery: Any, field: str) -> float:
    """Read a battery field from either a pydantic model or a plain dict."""
    if isinstance(battery, dict):
        return float(battery[field])
    return float(getattr(battery, field))


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)
