"""Pydantic v2 request/response models - the API contract of PROJECT.md section 5.

The models here are the single source of truth for what the service accepts and
emits. Everything downstream (optimizer, validator) works on plain floats and
lists; these models only guard the boundary.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# shared enums / aliases
# ---------------------------------------------------------------------------

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]

#: every directive type except ``no_op`` carries a structured_adjustment
REAL_DIRECTIVE_TYPES: tuple[str, ...] = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
)

Hour = Annotated[int, Field(ge=0, le=23)]


def _finite(value: float, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


# ---------------------------------------------------------------------------
# request
# ---------------------------------------------------------------------------


class HourInput(BaseModel):
    """One hourly row of the scenario (Problem Statement section 7.2)."""

    model_config = ConfigDict(extra="ignore")

    hour: Hour
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)

    @field_validator("demand_kwh", "solar_kwh", "tariff_bdt_per_kwh")
    @classmethod
    def _check_finite(cls, v: float, info) -> float:
        return _finite(v, info.field_name)


class BatteryInput(BaseModel):
    """Battery parameters (Problem Statement section 7.3)."""

    model_config = ConfigDict(extra="ignore")

    capacity_kwh: float = Field(ge=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)

    @field_validator("*")
    @classmethod
    def _check_finite(cls, v: float, info) -> float:
        return _finite(v, info.field_name)


class OptimizeRequest(BaseModel):
    """POST /optimize-energy request body.

    Liberal on input order: ``hours`` is sorted by ``hour`` in the validator, so
    every consumer downstream can index ``request.hours[h]`` safely.
    """

    model_config = ConfigDict(extra="ignore")

    scenario_id: str = Field(min_length=1)
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourInput] = Field(min_length=24, max_length=24)
    battery: BatteryInput

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, notes: list[str]) -> list[str]:
        for i, note in enumerate(notes):
            if not note or not note.strip():
                raise ValueError(f"operator_notes[{i}] must be a non-empty string")
        return notes

    @model_validator(mode="after")
    def _sort_and_check_hours(self) -> "OptimizeRequest":
        seen = {row.hour for row in self.hours}
        if seen != set(range(24)):
            raise ValueError("hours must contain exactly one entry for each hour 0..23")
        # "Be liberal on input order": sort by hour before anything else uses it.
        self.hours.sort(key=lambda row: row.hour)
        return self


# ---------------------------------------------------------------------------
# response - structured adjustments
# ---------------------------------------------------------------------------


class _AdjustmentBase(BaseModel):
    # extra="forbid" is what keeps stray "reason"/"confidence" keys out of the
    # emitted structured_adjustment (section 5.3).
    model_config = ConfigDict(extra="forbid")

    hours: list[Hour] = Field(min_length=1)

    @field_validator("hours")
    @classmethod
    def _unique_ascending(cls, hours: list[int]) -> list[int]:
        if len(set(hours)) != len(hours):
            raise ValueError("hours must be unique")
        if hours != sorted(hours):
            raise ValueError("hours must be in ascending order")
        return hours


class SolarReductionAdjustment(_AdjustmentBase):
    factor: float = Field(ge=0.0, le=1.0)


class MinimumBatteryReserveAdjustment(_AdjustmentBase):
    minimum_energy_kwh: float = Field(ge=0.0)


class MaxGridWindowAdjustment(_AdjustmentBase):
    max_grid_kwh: float = Field(ge=0.0)


class NoChargeWindowAdjustment(_AdjustmentBase):
    pass


class NoDischargeWindowAdjustment(_AdjustmentBase):
    pass


# NOTE: no_charge_window and no_discharge_window are structurally identical - both
# carry only ``hours`` - so a union cannot tell them apart from the payload alone.
# DirectiveInterpretation therefore builds the adjustment from ``directive_type``
# in a "before" validator rather than letting the union guess; leaving it to the
# union silently coerces every no_discharge_window into a no_charge_window.
StructuredAdjustment = Union[
    SolarReductionAdjustment,
    MinimumBatteryReserveAdjustment,
    MaxGridWindowAdjustment,
    NoChargeWindowAdjustment,
    NoDischargeWindowAdjustment,
]

#: directive_type -> the adjustment model it must carry
ADJUSTMENT_MODEL_FOR_TYPE: dict[str, type[_AdjustmentBase]] = {
    "solar_reduction": SolarReductionAdjustment,
    "minimum_battery_reserve": MinimumBatteryReserveAdjustment,
    "no_charge_window": NoChargeWindowAdjustment,
    "no_discharge_window": NoDischargeWindowAdjustment,
    "max_grid_window": MaxGridWindowAdjustment,
}


# ---------------------------------------------------------------------------
# response - interpretation + plan
# ---------------------------------------------------------------------------


class DirectiveInterpretation(BaseModel):
    """One entry per operator note, in ``note_index`` order."""

    model_config = ConfigDict(extra="forbid")

    note_index: int = Field(ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: StructuredAdjustment | None = None
    explanation: str = Field(min_length=1, max_length=200)

    @model_validator(mode="before")
    @classmethod
    def _bind_adjustment_to_type(cls, data: Any) -> Any:
        """Build the adjustment from ``directive_type``, not by union guessing.

        ``no_charge_window`` and ``no_discharge_window`` have identical payloads,
        so the union would always resolve both to the first one declared.
        """
        if not isinstance(data, dict):
            return data
        model = ADJUSTMENT_MODEL_FOR_TYPE.get(data.get("directive_type"))
        adjustment = data.get("structured_adjustment")
        if model is None or adjustment is None:
            return data
        if isinstance(adjustment, BaseModel):
            adjustment = adjustment.model_dump()
        if isinstance(adjustment, dict):
            return {**data, "structured_adjustment": model(**adjustment)}
        return data

    @model_validator(mode="after")
    def _applies_semantics(self) -> "DirectiveInterpretation":
        # Section 8.1: force-correct `applies` rather than reject - a model that
        # says applies:false on a real directive loses a point we can recover
        # deterministically.
        if self.directive_type == "no_op":
            self.applies = False
            self.structured_adjustment = None
            return self
        self.applies = True
        if self.structured_adjustment is None:
            raise ValueError(f"{self.directive_type} requires a structured_adjustment")
        expected = ADJUSTMENT_MODEL_FOR_TYPE[self.directive_type]
        if not isinstance(self.structured_adjustment, expected):
            raise ValueError(
                f"structured_adjustment shape does not match {self.directive_type}"
            )
        return self


class HourPlan(BaseModel):
    """One hour of the emitted 24-hour schedule (Problem Statement 10.3)."""

    model_config = ConfigDict(extra="forbid")

    hour: Hour
    grid_kwh: float = Field(ge=0)
    solar_used_kwh: float = Field(ge=0)
    battery_action: BatteryAction
    battery_kwh: float = Field(ge=0)
    battery_energy_after_kwh: float = Field(ge=0)

    @model_validator(mode="after")
    def _idle_is_zero(self) -> "HourPlan":
        if self.battery_action == "idle" and self.battery_kwh != 0:
            raise ValueError("battery_kwh must be 0 when battery_action is idle")
        return self


class OptimizeResponse(BaseModel):
    """POST /optimize-energy 200 response body (Problem Statement 10.1)."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourPlan] = Field(min_length=24, max_length=24)
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    """Controlled error body - never carries provider text or a stack trace."""

    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str | None = None
