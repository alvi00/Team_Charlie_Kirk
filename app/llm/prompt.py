"""System prompt, few-shot examples and user payload builder (PROJECT.md 7.2-7.4).

The few-shot wording is deliberately paraphrased rather than lifted from the
public sample pack: hidden notes paraphrase the same directives, so the examples
teach the pattern instead of memorising the cases. Hard-coding public note
wording is explicitly non-compliant.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

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
- "noon" = 12, "midnight" = 0, "13:00" = 13.
- A single hour mention ("at 3 PM", "during the 3 PM hour") -> [15].
- "for N hours starting at X" -> X, X+1, ... X+N-1.
- Windows that wrap past midnight are split and sorted ascending:
  "10 PM to 2 AM" -> [0,1,22,23].
- hours must be unique integers 0..23 in ascending order.

NUMERIC RULES (critical):
- solar_reduction "factor" is the FRACTION OF SOLAR THAT REMAINS USABLE.
  "drops to about 20%"            -> factor 0.2
  "an 80% reduction"              -> factor 0.2
  "roughly one-fifth of normal"   -> factor 0.2
  "about half the forecast"       -> factor 0.5
  "treated as roughly 25%"        -> factor 0.25
  factor is between 0 and 1 inclusive.
- minimum_battery_reserve "minimum_energy_kwh" is an absolute kWh number.
  If the note states a percentage of battery capacity, multiply it by the
  capacity_kwh given in the scenario context. 50% of a 200 kWh battery -> 100.
- max_grid_window "max_grid_kwh" is the per-hour cap in kWh.

RELEVANCE RULES:
- A note is no_op when it does not change today's 24-hour electricity schedule:
  administrative notices, room bookings, deadlines, menus, events next week,
  announcements, anything about a different day.
- Mark a note no_op if it cannot be expressed as one of the five real directive types.
- Never invent demand, solar, tariff, or battery values.
- Never invent a directive type that is not in the list.

OUTPUT SHAPE - exactly one entry per note, in note_index order:
{"directives":[
  {"note_index":0,"applies":true,"directive_type":"solar_reduction",
   "structured_adjustment":{"hours":[12,13],"factor":0.25},
   "explanation":"one short sentence"},
  {"note_index":1,"applies":false,"directive_type":"no_op",
   "structured_adjustment":null,"explanation":"one short sentence"}
]}

applies is false only for no_op. no_op always has structured_adjustment null.
Every other directive has applies true and the exact adjustment shape:
  solar_reduction         {"hours":[..],"factor":n}
  minimum_battery_reserve {"hours":[..],"minimum_energy_kwh":n}
  no_charge_window        {"hours":[..]}
  no_discharge_window     {"hours":[..]}
  max_grid_window         {"hours":[..],"max_grid_kwh":n}"""


#: Six examples covering every real directive type plus a distractor, in
#: paraphrased wording. The solar example uses "reduction" phrasing on purpose -
#: inverting it is the most expensive single mistake on this task.
#:
#: Kept deliberately terse. Every prompt token is charged against the provider's
#: per-minute budget on each request, and a smaller prompt is what keeps the
#: primary model reachable under repeated judge traffic.
#: (battery, notes, reply) - the user half is rendered through
#: ``build_user_message`` so a few-shot turn is byte-identical in format to the
#: real request. A format drift between the two is enough to cost a directive.
FEW_SHOT: list[tuple[dict[str, float], list[str], dict[str, Any]]] = [
    (
        {"capacity_kwh": 300, "minimum_energy_kwh": 50,
         "max_charge_kwh_per_hour": 70, "max_discharge_kwh_per_hour": 70},
        ["Expect rooftop output to fall by 70% from 09:00 to 12:00 during the array inspection."],
        {
            "directives": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "solar_reduction",
                    "structured_adjustment": {"hours": [9, 10, 11], "factor": 0.3},
                    "explanation": "A 70% fall leaves 30% of forecast solar usable.",
                }
            ]
        },
    ),
    (
        {"capacity_kwh": 250, "minimum_energy_kwh": 40,
         "max_charge_kwh_per_hour": 60, "max_discharge_kwh_per_hour": 60},
        ["Hold no less than 60% of pack capacity in the battery from 7 PM until 10 PM."],
        {
            "directives": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "minimum_battery_reserve",
                    "structured_adjustment": {
                        "hours": [19, 20, 21],
                        "minimum_energy_kwh": 150,
                    },
                    "explanation": "60% of the 250 kWh pack is a 150 kWh floor.",
                }
            ]
        },
    ),
    (
        {"capacity_kwh": 180, "minimum_energy_kwh": 30,
         "max_charge_kwh_per_hour": 45, "max_discharge_kwh_per_hour": 45},
        ["Chargers are locked out for switchgear work from 1 AM through 4 AM."],
        {
            "directives": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "no_charge_window",
                    "structured_adjustment": {"hours": [1, 2, 3]},
                    "explanation": "Charging is unavailable while the chargers are locked out.",
                }
            ]
        },
    ),
    (
        {"capacity_kwh": 220, "minimum_energy_kwh": 35,
         "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50},
        ["During protection relay checks between 4 PM and 6 PM the pack must not be drawn down."],
        {
            "directives": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "no_discharge_window",
                    "structured_adjustment": {"hours": [16, 17]},
                    "explanation": "The battery cannot discharge during the relay checks.",
                }
            ]
        },
    ),
    (
        {"capacity_kwh": 240, "minimum_energy_kwh": 40,
         "max_charge_kwh_per_hour": 55, "max_discharge_kwh_per_hour": 55},
        ["Feeder works cap campus import at 140 kWh per hour from 8 PM until 11 PM.",
         "The canteen will trial a new breakfast menu starting next Monday."],
        {
            "directives": [
                {
                    "note_index": 0,
                    "applies": True,
                    "directive_type": "max_grid_window",
                    "structured_adjustment": {
                        "hours": [20, 21, 22],
                        "max_grid_kwh": 140,
                    },
                    "explanation": "Grid import is limited to 140 kWh in each of those hours.",
                },
                {
                    "note_index": 1,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": "A catering change next week does not affect today's schedule.",
                },
            ]
        },
    ),
    (
        {"capacity_kwh": 200, "minimum_energy_kwh": 40,
         "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50},
        ["Lecture theatre 3 is reserved for a departmental seminar on Thursday afternoon."],
        {
            "directives": [
                {
                    "note_index": 0,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": "A room booking on another day does not change today's schedule.",
                }
            ]
        },
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
    lines.append("Return JSON only.")
    return "\n".join(lines)


def build_messages(notes: Sequence[str], battery: Any) -> list[dict[str, str]]:
    """System prompt + few-shot turns + the real request."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot_battery, shot_notes, reply in FEW_SHOT:
        # Rendered through the same builder as the real turn, so the model never
        # sees a format it has to generalise across.
        messages.append({"role": "user", "content": build_user_message(shot_notes, shot_battery)})
        messages.append(
            {"role": "assistant", "content": json.dumps(reply, separators=(",", ":"))}
        )
    messages.append({"role": "user", "content": build_user_message(notes, battery)})
    return messages


def _get(battery: Any, field: str) -> float:
    """Read a battery field from either a pydantic model or a plain dict."""
    if isinstance(battery, dict):
        return float(battery[field])
    return float(getattr(battery, field))


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
                "That response was rejected: "
                f"{error_text[:500]}\n"
                "Return the corrected result as JSON only - a single object with a "
                '"directives" array holding exactly one entry per note, in note_index '
                "order. No prose, no markdown, no code fences."
            ),
        }
    )
    return messages


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)
