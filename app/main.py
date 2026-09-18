"""FastAPI application: routes, pipeline orchestration and exception handlers.

Pipeline (PROJECT.md section 3):

    request -> validation -> LLM interpretation -> guardrails -> LP optimizer
            -> post-solve conditioning -> replay validation -> response

Failure is always controlled: a malformed body is a 400, a genuinely unsolvable
scenario is a 422, and every other path - provider outage, malformed model
output, an infeasible program - still returns a schema-valid 200.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import get_settings
from .llm.interpreter import interpret_notes, reset_client
from .optimizer.model import CompiledConstraints, Scenario, compile_directives
from .optimizer.solve import InfeasibleError, SolveResult, solve
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
    if not settings.llm_configured:
        log.warning("GROQ_API_KEY is not set - the deterministic interpreter will be used")
    yield
    reset_client()
    log.info("gridwise shutting down")


app = FastAPI(
    title="GridWise LLM",
    version="1.0.0",
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


@app.exception_handler(StarletteHTTPException)
async def _http_handler(request: Request, exc: StarletteHTTPException):
    label = "not_found" if exc.status_code == 404 else "http_error"
    detail = exc.detail if isinstance(exc.detail, str) else None
    return _error(exc.status_code, label, detail)


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    # Logged in full server-side; the client gets no stack trace and no provider
    # message, which is what the secret-handling criterion asks for.
    log.exception("unhandled error on %s", request.url.path)
    return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error")


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness probe. Deliberately trivial: no LLM call, no I/O."""
    return HealthResponse(status="ok")


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(request: OptimizeRequest) -> Any:
    unsolvable = _semantic_check(request)
    if unsolvable:
        return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, "unprocessable", unsolvable)

    started = time.perf_counter()
    response = run_pipeline(request)
    log.info(
        "scenario=%s notes=%d cost=%.2f elapsed_ms=%.0f",
        request.scenario_id,
        len(request.operator_notes),
        response.total_cost_bdt,
        (time.perf_counter() - started) * 1000,
    )
    return response


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

    # [2] LLM interpretation + [3] deterministic guardrails
    interpretation = interpret_notes(request.operator_notes, request.battery)
    entries = _to_models(interpretation.entries, request.operator_notes)

    interp_failures = validate_interpretation(
        entries, len(request.operator_notes), request.battery.capacity_kwh
    )
    if interp_failures:
        # Logged only - the response never carries validator internals.
        log.warning("interpretation failed self-check: %s", interp_failures)
    log.info(
        "interpretation source=%s model=%s types=%s",
        interpretation.source,
        interpretation.model,
        [entry.directive_type for entry in entries],
    )

    # [4] LP + [5] post-solve conditioning
    cons = compile_directives(entries, scenario)
    degraded_note: str | None = None
    result: SolveResult | None
    try:
        result = solve(scenario, cons)
    except InfeasibleError:
        log.warning("scenario %s infeasible on every rung", scenario.scenario_id)
        result = None

    # [6] replay validation against our own interpretation
    if result is not None and not _replay_ok(result, scenario, entries):
        result = _repair(scenario, cons, entries)

    if result is None:
        plan_data = safe_fallback_plan(scenario, entries)
        degraded_note = "A conservative fallback schedule was used for this scenario."
    else:
        plan_data = _as_plan_data(result)
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


def _to_models(
    raw_entries: list[dict[str, Any]], notes: list[str]
) -> list[DirectiveInterpretation]:
    """Build the response models, degrading any entry the schema rejects to no_op.

    The guardrails already guarantee the shape; this is the last belt-and-braces
    step that keeps a surprise from becoming a 500.
    """
    models: list[DirectiveInterpretation] = []
    for index in range(len(notes)):
        raw = raw_entries[index] if index < len(raw_entries) else None
        try:
            models.append(DirectiveInterpretation(**raw))
        except Exception:
            log.warning("entry %d rejected by the response schema, using no_op", index)
            models.append(
                DirectiveInterpretation(
                    note_index=index,
                    applies=False,
                    directive_type="no_op",
                    structured_adjustment=None,
                    explanation="This note does not affect today's 24-hour energy schedule.",
                )
            )
    return models


def _as_plan_data(result: SolveResult) -> dict[str, Any]:
    return {
        "hourly_plan": result.hourly_plan,
        "total_grid_kwh": result.total_grid_kwh,
        "total_cost_bdt": result.total_cost_bdt,
        "peak_grid_kwh": result.peak_grid_kwh,
    }


def _replay_ok(
    result: SolveResult, scenario: Scenario, entries: list[DirectiveInterpretation]
) -> bool:
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
        log.warning("replay failed: %s", check.failures)
    return check.ok


def _repair(
    scenario: Scenario,
    cons: CompiledConstraints,
    entries: list[DirectiveInterpretation],
) -> SolveResult | None:
    """Section 10.2: one repair pass - re-solve and re-condition, then give up."""
    try:
        retry = solve(scenario, cons)
    except InfeasibleError:
        return None
    if _replay_ok(retry, scenario, entries):
        return retry
    log.error("repair pass still invalid for %s", scenario.scenario_id)
    return None
