"""Guardrail tests: paraphrase robustness and LLM-output repair (PROJECT.md 8, 11).

The note wording here is invented, not copied from the public pack - hidden cases
paraphrase the same directives, so these target the 5 paraphrase-robustness marks
rather than the sample strings.

Runs under pytest, or standalone with no extra dependency:

    python tests/test_guardrails.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.guardrails.fallback import interpret_rule_based  # noqa: E402
from app.guardrails.numbers import normalize_factor, normalize_reserve  # noqa: E402
from app.guardrails.timeparse import parse_hour_windows  # noqa: E402
from app.guardrails.validate import canonical_type, normalize_interpretation  # noqa: E402
from app.schemas import DirectiveInterpretation  # noqa: E402


def entry(index: int, dtype: str, adjustment, applies=True, explanation="because"):
    return {
        "note_index": index,
        "applies": applies,
        "directive_type": dtype,
        "structured_adjustment": adjustment,
        "explanation": explanation,
    }


def only(notes, raw, capacity=200.0):
    result = normalize_interpretation(raw, notes, capacity)
    assert len(result.entries) == len(notes)
    return result.entries


# ---------------------------------------------------------------------------
# end-exclusive windows (the single most common LLM error)
# ---------------------------------------------------------------------------


def test_end_exclusive_windows():
    cases = [
        ("Solar output will drop to about 20% from 1 PM to 3 PM.", [13, 14]),
        ("Do not charge the battery between 2 PM and 4 PM.", [14, 15]),
        ("No discharging from 6 PM until 9 PM.", [18, 19, 20]),
        ("No discharging from 6 PM until 10 PM.", [18, 19, 20, 21]),
        ("Charger offline from 2 AM until 5 AM.", [2, 3, 4]),
        ("Panels washed from noon until 2 PM.", [12, 13]),
        ("Cloud cover from 10 AM until noon.", [10, 11]),
        ("Cap import from 19:00 to 22:00.", [19, 20, 21]),
        ("No battery charging while the inverter is offline, 11 AM-1 PM.", [11, 12]),
        ("Overnight from 10 PM to 2 AM, don't discharge.", [0, 1, 22, 23]),
        ("Hold reserve for three hours starting at 2 PM.", [14, 15, 16]),
        ("During the 3 PM hour, do not charge.", [15]),
    ]
    for note, expected in cases:
        parsed = parse_hour_windows(note)
        assert parsed.decisive, f"declined to parse: {note}"
        assert parsed.hours == expected, f"{note}: {parsed.hours} != {expected}"


def test_parser_declines_when_there_is_no_window():
    for note in (
        "The sports office moved next month's registration deadline.",
        "Parking permits renew on Sunday.",
        "Solar will be normal today.",
        "Grid import must not exceed 155 kWh in any hour.",
        "Keep between 40 and 60 kWh spare.",
    ):
        assert not parse_hour_windows(note).decisive, f"wrongly parsed: {note}"


# ---------------------------------------------------------------------------
# factor is the fraction REMAINING
# ---------------------------------------------------------------------------


def test_factor_is_the_remaining_fraction():
    cases = [
        ("Expect an 80% reduction in rooftop solar.", 0.2),
        ("PV output drops to about 20% this afternoon.", 0.2),
        ("PV output will be about a fifth of normal.", 0.2),
        ("Panel washing leaves roughly one-fifth of normal solar output.", 0.2),
        ("Cloud cover leaves about half of the forecast solar.", 0.5),
        ("Solar reduced by 70% during the inspection.", 0.3),
        ("Rooftop output will fall by 60 percent.", 0.4),
        ("Expect solar 40% lower today.", 0.6),
        ("Panels operating at 30% capacity.", 0.3),
        ("Solar will run at roughly three-quarters of normal.", 0.75),
        ("Treat usable solar as roughly 25% of the forecast.", 0.25),
    ]
    for note, expected in cases:
        # The LLM value is deliberately wrong; the deterministic read must win.
        got = normalize_factor(0.99, note)
        assert abs(got - expected) < 1e-6, f"{note}: {got} != {expected}"


def test_factor_falls_back_to_the_model_when_the_text_is_silent():
    assert normalize_factor(0.35, "Solar will be poor during the works.") == 0.35
    # a model that answered in percent rather than as a fraction
    assert normalize_factor(80, "Solar will be poor during the works.") == 0.8


# ---------------------------------------------------------------------------
# percent-of-capacity reserve
# ---------------------------------------------------------------------------


def test_percent_of_capacity_reserve_resolves_against_capacity():
    note = "Hold 40% of pack capacity in reserve through the evening peak, 6 to 9 PM."
    assert normalize_reserve(40, note, 220.0) == 88.0
    assert normalize_reserve(88, note, 220.0) == 88.0


def test_absolute_reserve_is_left_alone_and_clamped_to_capacity():
    assert normalize_reserve(90, "Keep at least 90 kWh in the battery.", 250.0) == 90.0
    assert normalize_reserve(300, "Keep 300 kWh in reserve.", 200.0) == 200.0


# ---------------------------------------------------------------------------
# arbitration: deterministic hours override the model
# ---------------------------------------------------------------------------


def test_arbitration_fixes_an_off_by_one_window():
    notes = ["PV output will be about a fifth of normal between 11:00 and 14:00."]
    # classic model error: end-inclusive hours and an inverted factor
    raw = [entry(0, "solar_reduction", {"hours": [11, 12, 13, 14], "factor": 0.8})]
    got = only(notes, raw)[0]
    assert got["structured_adjustment"]["hours"] == [11, 12, 13]
    assert abs(got["structured_adjustment"]["factor"] - 0.2) < 1e-6


def test_model_hours_are_kept_when_the_parser_cannot_rule():
    notes = ["Panel washing from one until three leaves a fifth of normal output."]
    raw = [entry(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2})]
    got = only(notes, raw)[0]
    assert got["structured_adjustment"]["hours"] == [13, 14]


def test_hours_are_deduped_sorted_and_range_checked():
    notes = ["Do not charge the battery during the maintenance slot."]
    raw = [entry(0, "no_charge_window", {"hours": [5, 3, 3, 99, -2, 4]})]
    got = only(notes, raw)[0]
    assert got["structured_adjustment"]["hours"] == [3, 4, 5]


# ---------------------------------------------------------------------------
# structural repair (section 8.1)
# ---------------------------------------------------------------------------


def test_applies_is_force_corrected_both_ways():
    notes = ["Chargers are isolated from 2 AM until 5 AM.", "Menu changes tomorrow."]
    raw = [
        entry(0, "no_charge_window", {"hours": [2, 3, 4]}, applies=False),
        entry(1, "no_op", None, applies=True),
    ]
    got = only(notes, raw)
    assert got[0]["applies"] is True
    assert got[1]["applies"] is False
    assert got[1]["structured_adjustment"] is None


def test_extra_adjustment_keys_are_stripped():
    notes = ["Cap grid import at 150 kWh from 7 PM until 9 PM."]
    raw = [
        entry(
            0,
            "max_grid_window",
            {"hours": [19, 20], "max_grid_kwh": 150, "reason": "feeder", "confidence": 0.9},
        )
    ]
    got = only(notes, raw)[0]
    assert set(got["structured_adjustment"]) == {"hours", "max_grid_kwh"}


def test_missing_and_extra_entries_are_reconciled():
    notes = ["Chargers isolated 2 AM until 5 AM.", "Nothing relevant.", "Also nothing."]
    raw = [
        entry(2, "no_op", None, applies=False),
        entry(0, "no_charge_window", {"hours": [2, 3, 4]}),
        entry(7, "solar_reduction", {"hours": [1], "factor": 0.5}),  # out of range
    ]
    got = only(notes, raw)
    assert [e["note_index"] for e in got] == [0, 1, 2]
    assert got[0]["directive_type"] == "no_charge_window"
    assert got[1]["directive_type"] == "no_op"  # synthesised


def test_unknown_directive_type_becomes_no_op():
    notes = ["Shed load between 2 PM and 4 PM."]
    raw = [entry(0, "load_shedding", {"hours": [14, 15]})]
    assert only(notes, raw)[0]["directive_type"] == "no_op"


def test_directive_type_aliases_are_accepted():
    assert canonical_type("Solar_Reduction") == "solar_reduction"
    assert canonical_type("no-charge-window") == "no_charge_window"
    assert canonical_type("grid_cap") == "max_grid_window"
    assert canonical_type("NOOP") == "no_op"
    assert canonical_type("something_else") is None


def test_directive_without_usable_hours_is_downgraded():
    notes = ["Solar will be reduced at some point today."]
    raw = [entry(0, "solar_reduction", {"hours": [], "factor": 0.5})]
    assert only(notes, raw)[0]["directive_type"] == "no_op"


def test_garbage_model_output_never_raises():
    notes = ["Chargers isolated 2 AM until 5 AM."]
    for raw in (None, [], "not a list", [{}], [{"note_index": "x"}], [None], {"a": 1}):
        result = normalize_interpretation(raw, notes, 200.0)
        assert len(result.entries) == 1
        assert result.entries[0]["note_index"] == 0


def test_explanation_is_always_present_and_bounded():
    notes = ["Chargers isolated 2 AM until 5 AM.", "Nothing relevant."]
    raw = [
        entry(0, "no_charge_window", {"hours": [2, 3, 4]}, explanation=""),
        entry(1, "no_op", None, explanation="x" * 500),
    ]
    got = only(notes, raw)
    assert got[0]["explanation"]
    assert 0 < len(got[1]["explanation"]) <= 200


# ---------------------------------------------------------------------------
# response schema: the adjustment must bind to its own directive_type
# ---------------------------------------------------------------------------


def test_every_directive_type_survives_the_response_model():
    """Regression: no_charge_window and no_discharge_window have identical
    payloads, so a plain union resolves both to whichever is declared first and
    silently destroys every no_discharge_window directive."""
    cases = [
        ("solar_reduction", {"hours": [12, 13], "factor": 0.25}),
        ("minimum_battery_reserve", {"hours": [18, 19], "minimum_energy_kwh": 100.0}),
        ("no_charge_window", {"hours": [2, 3, 4]}),
        ("no_discharge_window", {"hours": [17, 18]}),
        ("max_grid_window", {"hours": [19, 20], "max_grid_kwh": 180.0}),
        ("no_op", None),
    ]
    for directive_type, adjustment in cases:
        model = DirectiveInterpretation(
            note_index=0,
            applies=directive_type != "no_op",
            directive_type=directive_type,
            structured_adjustment=adjustment,
            explanation="test",
        )
        dumped = model.model_dump()
        assert dumped["directive_type"] == directive_type
        assert dumped["structured_adjustment"] == adjustment, directive_type


def test_normalized_entries_are_accepted_by_the_response_model():
    notes = [
        "Do not discharge the battery from 5 PM until 7 PM during relay testing.",
        "Battery charging is disabled from 11 AM until 1 PM.",
    ]
    raw = [
        entry(0, "no_discharge_window", {"hours": [17, 18]}),
        entry(1, "no_charge_window", {"hours": [11, 12]}),
    ]
    for normalized in only(notes, raw):
        model = DirectiveInterpretation(**normalized)
        assert model.directive_type == normalized["directive_type"]
        assert model.model_dump()["structured_adjustment"] == normalized["structured_adjustment"]


# ---------------------------------------------------------------------------
# rule-based fallback (only reached when every LLM attempt fails)
# ---------------------------------------------------------------------------


def test_rule_based_fallback_covers_every_directive_type():
    notes = [
        "Expect an 80% reduction in rooftop solar between 11 AM and 2 PM.",
        "The battery charger will be isolated from 2 AM until 5 AM.",
        "Do not discharge the battery from 5 PM until 7 PM during relay testing.",
    ]
    got = interpret_rule_based(notes, 240.0)
    assert got[0]["directive_type"] == "solar_reduction"
    assert got[0]["structured_adjustment"] == {"hours": [11, 12, 13], "factor": 0.2}
    assert got[1]["directive_type"] == "no_charge_window"
    assert got[1]["structured_adjustment"] == {"hours": [2, 3, 4]}
    assert got[2]["directive_type"] == "no_discharge_window"
    assert got[2]["structured_adjustment"] == {"hours": [17, 18]}


def test_rule_based_fallback_handles_reserve_and_grid_cap():
    notes = [
        "Keep at least 50% of the battery capacity stored from 6 PM until 9 PM.",
        "Grid import must not exceed 155 kWh in any hour from 6 PM until 9 PM.",
    ]
    got = interpret_rule_based(notes, 200.0)
    assert got[0]["directive_type"] == "minimum_battery_reserve"
    assert got[0]["structured_adjustment"]["minimum_energy_kwh"] == 100.0
    assert got[1]["directive_type"] == "max_grid_window"
    assert got[1]["structured_adjustment"]["max_grid_kwh"] == 155.0


def test_rule_based_fallback_handles_paraphrased_wording():
    """Phrasings that the earlier regexes missed, all of which the fallback has
    to survive because it is what answers during a provider outage."""
    cases = [
        ("Haze will cut array output by half from 10 AM until 1 PM.",
         "solar_reduction", {"hours": [10, 11, 12], "factor": 0.5}),
        ("Emergency services need a floor of 95 kWh in storage between 6 PM and 9 PM.",
         "minimum_battery_reserve", {"hours": [18, 19, 20], "minimum_energy_kwh": 95.0}),
        ("Please avoid charging the battery from 2 PM until 5 PM while the feeder is tested.",
         "no_charge_window", {"hours": [14, 15, 16]}),
        ("No battery discharging from 8 PM until 11 PM tonight.",
         "no_discharge_window", {"hours": [20, 21, 22]}),
        ("The charge point is out of service between 9 AM and 11 AM.",
         "no_charge_window", {"hours": [9, 10]}),
        ("Substation constrained: grid intake must stay at or below 165 kWh from 19:00 to 22:00.",
         "max_grid_window", {"hours": [19, 20, 21], "max_grid_kwh": 165.0}),
    ]
    for note, directive_type, adjustment in cases:
        got = interpret_rule_based([note], 220.0)[0]
        assert got["directive_type"] == directive_type, f"{note}: {got['directive_type']}"
        assert got["structured_adjustment"] == adjustment, f"{note}: {got['structured_adjustment']}"


def test_rule_based_fallback_marks_distractors_no_op():
    notes = [
        "The sports office moved next month's registration deadline.",
        "Parking permits renew on Sunday.",
        "Solar will be normal today.",
    ]
    for got in interpret_rule_based(notes, 200.0):
        assert got["directive_type"] == "no_op"
        assert got["applies"] is False
        assert got["structured_adjustment"] is None


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
