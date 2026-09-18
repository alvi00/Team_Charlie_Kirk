"""FastAPI application: routes, pipeline orchestration and exception handlers.

Pipeline (PROJECT.md section 3):

    request -> validation -> LLM interpretation -> guardrails -> LP optimizer
            -> post-solve conditioning -> replay validation -> response

P0/P1 status: the interpreter is a stub that returns ``no_op`` for every note, so
the contract is exercisable end to end. The optimizer, the post-solve
conditioning and the replay validator are complete.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import get_settings
from .llm.interpreter import interpret_notes
from .optimizer.model import Scenario, compile_directives
from .optimizer.solve import InfeasibleError, solve
from .schemas import (
    DirectiveInterpretation,
    ErrorResponse,
    HealthResponse,
    HourPlan,
    OptimizeRequest,
    OptimizeResponse,
)
from .summary import build_plan_summary
from .validator.replay import replay, safe_fallback_plan, validate_interpretation

log = logging.getLogger("gridwise")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # safe_dump() never contains the API key
    log.info("gridwise starting: %s", settings.safe_dump())
    yield
    log.info("gridwise shutting down")


app = FastAPI(
    title="GridWise LLM",
    version="0.1.0",
    description="LLM-assisted campus energy scheduling (BUP CSE Fest 2026 preliminary).",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# error handling - controlled, never leaking provider text or stack traces
# ---------------------------------------------------------------------------


def _error(code: int, error: str, detail: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content=ErrorResponse(error=error, detail=detail).model_dump(exclude_none=True),
    )


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    # The spec wants 400 for malformed/structurally invalid input, not FastAPI's
    # default 422.
    problems = []
    for err in exc.errors()[:5]:
        location = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        problems.append(f"{location or 'body'}: {err.get('msg', 'invalid')}")
    return _error(status.HTTP_400_BAD_REQUEST, "bad_request", "; ".join(problems))


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error")


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness probe. Deliberately trivial: no LLM call, no I/O."""
    return HealthResponse(status="ok")


@app.post(
    "/optimize-energy",
    response_model=OptimizeResponse,
    response_model_exclude_none=False,
)
def optimize_energy(request: OptimizeRequest) -> Any:
    unsolvable = _semantic_check(request)
    if unsolvable:
        return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, "unprocessable", unsolvable)
    return run_pipeline(request)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def _semantic_check(request: OptimizeRequest) -> str | None:
    """Well-formed but genuinely unsolvable battery parameters (section 5.2)."""
    b = request.battery
    if b.initial_energy_kwh > b.capacity_kwh:
        return "initial_energy_kwh exceeds capacity_kwh"
    if b.minimum_energy_kwh > b.capacity_kwh:
        return "minimum_energy_kwh exceeds capacity_kwh"
    if b.initial_energy_kwh < b.minimum_energy_kwh:
        return "initial_energy_kwh is below minimum_energy_kwh"
    return None


def run_pipeline(request: OptimizeRequest) -> OptimizeResponse:
    """Interpret, optimize, replay-validate and assemble the response."""
    scenario = Scenario.from_request(request)

    # [2] LLM interpretation (P0: stub -> all no_op) ...
    interpretation = interpret_notes(request.operator_notes, request.battery)
    # ... [3] guardrails land in P3; for now the raw entries go straight into the
    # response models, which already force-correct the `applies` semantics.
    entries = [DirectiveInterpretation(**raw) for raw in interpretation.entries]

    interp_failures = validate_interpretation(
        entries, len(request.operator_notes), request.battery.capacity_kwh
    )
    if interp_failures:
        # Logged only - the response never carries validator internals.
        log.warning("interpretation failed self-check: %s", interp_failures)

    # [4] LP + [5] post-solve conditioning
    cons = compile_directives(entries, scenario)
    degraded_note: str | None = None
    try:
        result = solve(scenario, cons)
    except InfeasibleError:
        log.warning("scenario %s infeasible on every rung", scenario.scenario_id)
        result = None

    # [6] replay validation against our own interpretation
    if result is not None:
        check = replay(
            result.hourly_plan,
            scenario,
            entries,
            {
                "total_grid_kwh": result.total_grid_kwh,
                "total_cost_bdt": result.total_cost_bdt,
                "peak_grid_kwh": result.peak_grid_kwh,
            },
        )
        if not check.ok:
            log.warning("replay failed, retrying solve: %s", check.failures)
            result = _repair(scenario, cons, entries)

    if result is None:
        plan_data = safe_fallback_plan(scenario, entries)
        degraded_note = "A conservative fallback schedule was used for this scenario."
    else:
        plan_data = {
            "hourly_plan": result.hourly_plan,
            "total_grid_kwh": result.total_grid_kwh,
            "total_cost_bdt": result.total_cost_bdt,
            "peak_grid_kwh": result.peak_grid_kwh,
        }
        if result.notes:
            degraded_note = "Constraints were relaxed to keep this scenario solvable."

    # [7] response assembly
    return OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=entries,
        hourly_plan=[HourPlan(**row) for row in plan_data["hourly_plan"]],
        total_grid_kwh=plan_data["total_grid_kwh"],
        total_cost_bdt=plan_data["total_cost_bdt"],
        peak_grid_kwh=plan_data["peak_grid_kwh"],
        plan_summary=build_plan_summary(
            entries,
            plan_data["hourly_plan"],
            plan_data["total_cost_bdt"],
            plan_data["peak_grid_kwh"],
            degraded_note,
        ),
    )


def _repair(scenario: Scenario, cons, entries):
    """Section 10.2: one repair pass - re-solve and re-condition, then give up."""
    try:
        retry = solve(scenario, cons)
    except InfeasibleError:
        return None
    check = replay(
        retry.hourly_plan,
        scenario,
        entries,
        {
            "total_grid_kwh": retry.total_grid_kwh,
            "total_cost_bdt": retry.total_cost_bdt,
            "peak_grid_kwh": retry.peak_grid_kwh,
        },
    )
    if check.ok:
        return retry
    log.error("repair pass still invalid: %s", check.failures)
    return None
