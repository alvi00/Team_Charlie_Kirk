# GridWise LLM — Smart Campus Energy Optimization

**BUP CSE Fest 2026 · Hackathon · Online Preliminary**

An HTTP service that reads a 24-hour campus energy scenario plus 1–3 natural-language
operator notes, uses an **LLM to interpret the notes into structured directives**, validates
those directives deterministically, applies them as hard constraints to a linear program,
and returns the cheapest valid 24-hour schedule.

| | |
|---|---|
| Health endpoint | `GET /health` → `{"status":"ok"}` |
| Main endpoint | `POST /optimize-energy` |
| LLM provider | OpenAI (Chat Completions API) |
| Model | `gpt-5.6-luna`, falling back to `gpt-4.1-mini`, then an optional second provider |
| Solver | SciPy `linprog(method="highs")` — exact global optimum |
| Public-sample result | 10/10 interpretation · 10/10 valid · mean cost ratio **1.000** |
| Unseen-paraphrase result | 24/24 on a held-out paraphrase set (no public-pack wording) |
| Latency | p50 ~2.3 s · p95 ~3.0 s (requirement: p95 ≤ 5 s) |

---

## 1. Quickstart

From a clean machine:

```bash
git clone https://github.com/alvi00/Team_Charlie_Kirk.git && cd Team_Charlie_Kirk

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # then open .env and set OPENAI_API_KEY
python tools/verify_model.py       # confirms the key and model id actually work

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Verify in a second terminal:

```bash
curl localhost:8000/health
# {"status":"ok"}
```

Python 3.11 is the tested runtime (3.10+ works). No build tools or system packages
are required beyond the wheels in `requirements.txt`.

---

## 2. Environment & model configuration

Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`. **No secret values appear in
this repository** — only variable names.

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key. **Required** for the LLM path; without it the service still answers using its deterministic interpreter. | *(empty)* |
| `OPENAI_BASE_URL` | Chat Completions base URL | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Primary interpretation model | `gpt-5.6-luna` |
| `OPENAI_FALLBACK_MODEL` | Second model at the same provider, tried if the primary errors | `gpt-4.1-mini` |
| `FALLBACK_PROVIDER_API_KEY` | Optional **cross-provider** failover. Leave empty to disable. | *(empty)* |
| `FALLBACK_PROVIDER_BASE_URL` | Failover provider base URL (any OpenAI-compatible vendor) | `https://api.groq.com/openai/v1` |
| `FALLBACK_PROVIDER_MODEL` | Failover model id | `openai/gpt-oss-120b` |
| `LLM_TIMEOUT_SECONDS` | Per-call HTTP timeout. A normal answer takes 2–4 s; past 6 s the call is treated as hung and the next model is tried. | `6` |
| `LLM_MAX_RETRIES` | Attempts on the primary model before moving down the ladder. Kept at 1 so a hung primary hands the same request to the fallback model rather than being retried. | `1` |
| `LLM_TOTAL_BUDGET_SECONDS` | Wall-clock ceiling for the whole interpretation stage. Once spent, the service stops calling providers and interprets deterministically, so a hanging provider can never push a request past the judge's 30s limit. | `12` |
| `LLM_REASONING_EFFORT` | Sent to reasoning models; dropped automatically if rejected | `low` |
| `PORT` | Listen port (honoured by Railway and the Docker image) | `8000` |
| `LOG_LEVEL` | Python log level | `INFO` |

### Verifying the model id

A wrong model id does not crash — it 404s and interpretation silently degrades to the
deterministic fallback, which is exactly the failure that is hardest to notice. Run:

```bash
python tools/verify_model.py
```

It checks that each configured model authenticates, exists, answers, and interprets a
probe note correctly, and prints which request parameters the model accepted. Run it
after setting the key and after changing any model id.

### Capability negotiation

Model families disagree about request parameters — newer reasoning models require
`max_completion_tokens` instead of `max_tokens`, may reject a non-default `temperature`,
and may or may not support `reasoning_effort` or strict `json_schema` structured outputs.
Rather than hard-coding assumptions, the client starts from the strictest, most capable
parameter set and, on a 400 naming an offending parameter, drops or swaps that parameter
and retries. The working profile is cached per (endpoint, model), so discovery is paid
once per process. **Switching models or providers needs no code change.**

---

## 3. Testing against the public sample cases

The harness in `tests/run_public_cases.py` scores the pack exactly the way the judge
does: it replays each returned schedule against the **organizer's ground-truth
directives**, not against our own interpretation.

**Offline** — optimizer and validator only, no server and no API key needed:

```bash
python tests/run_public_cases.py
```

**Against a running service** — the full LLM → guardrails → optimizer pipeline:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000     # terminal 1
python tests/run_public_cases.py --mode api --base-url http://localhost:8000
```

Expected output in both modes:

```
interpretation    : 10/10 cases (18/18 entries)
replay-valid      : 10/10
cost matches exact: 10/10
mean cost ratio   : 1.0000

ACCEPTANCE GATE: PASS
```

Per-case costs match the organizer's reference exactly: 38365, 42885, 35480, 40495,
33950, 34090, 38550, 37665, 34873, 41620 BDT (total 377,973 BDT).

> If your provider account has a low tokens-per-minute ceiling, firing ten cases in ten
> seconds can trip it, and you will be measuring the fallback path rather than the model.
> Add `--delay 14` to pace the run.

The rest of the suite (no network or API key required):

```bash
python tests/test_optimizer.py     # LP, post-solve conditioning, replay validator
python tests/test_guardrails.py    # paraphrase robustness, LLM-output repair
python tests/test_api.py           # request/response contract, 400/422, secret safety
```

### Sample request

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  -d '{
    "scenario_id": "SAMPLE-01",
    "operator_notes": [
      "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
      "The sports office moved next month'"'"'s registration deadline."
    ],
    "hours": [
      {"hour": 0, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
      "... 22 more hourly entries ...",
      {"hour": 23, "demand_kwh": 105, "solar_kwh": 0, "tariff_bdt_per_kwh": 7}
    ],
    "battery": {
      "capacity_kwh": 220, "initial_energy_kwh": 110, "minimum_energy_kwh": 40,
      "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50
    }
  }'
```

### Sample response

```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
      "explanation": "Cleaning reduces usable solar to roughly 25% during those hours."
    },
    {
      "note_index": 1,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "The deadline change is unrelated to today's schedule."
    }
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 50.0, "solar_used_kwh": 0.0, "battery_action": "discharge", "battery_kwh": 40.0, "battery_energy_after_kwh": 70.0},
    {"hour": 1, "grid_kwh": 85.0, "solar_used_kwh": 0.0, "battery_action": "idle", "battery_kwh": 0.0, "battery_energy_after_kwh": 70.0},
    {"hour": 2, "grid_kwh": 130.0, "solar_used_kwh": 0.0, "battery_action": "charge", "battery_kwh": 50.0, "battery_energy_after_kwh": 120.0},
    "... 20 more hours ...",
    {"hour": 23, "grid_kwh": 155.0, "solar_used_kwh": 0.0, "battery_action": "charge", "battery_kwh": 50.0, "battery_energy_after_kwh": 110.0}
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 187.5,
  "plan_summary": "Applied reduced solar availability as hard constraints. 1 note(s) were irrelevant and treated as no_op. The battery charges in hour(s) 2, 3, 4, 13, 14, 15, 22, 23 and discharges in hour(s) 0, 10, 11, 12, 17, 18, 19, 20, ending the day at its starting energy. Solar is used first, then the cheapest grid hours, for a total of 38365 BDT with a peak import of 187.5 kWh."
}
```

---

## 4. Architecture

Human notes are never trusted as math. They are converted to a fixed structured format,
checked by deterministic guardrails, and only then applied to the optimization model.

```
POST /optimize-energy
      │
      ▼
[1] Request validation (Pydantic v2)  ──malformed──▶ 400
      │
      ▼
[2] LLM interpreter (OpenAI, strict JSON schema, ONE call for all notes)
      │   → raw candidate directives (UNTRUSTED)
      ▼
[3] Deterministic guardrails + normalizer
      │   → one entry per note, repaired and normalized
      ▼
[4] LP optimizer (SciPy HiGHS) — directives compiled into hard constraints
      │   → grid[h], solar_used[h], charge[h], discharge[h]
      ▼
[5] Post-solve conditioning: net charge/discharge, clamp, round,
      │   forward-simulate battery energy, recompute totals
      ▼
[6] Replay validator — an independent clone of the judge
      │   pass ──▶ response       fail ──▶ repair pass ──▶ safe fallback plan
      ▼
[7] Response assembly (+ plan_summary) → 200
```

### What the LLM does

The LLM is the **primary interpreter of `operator_notes`**. It produces the
`directive_interpretation` block, and that block is what compiles into the optimizer's
constraints — it is not cosmetic text. Each note becomes exactly one of:

| Directive | Effect on the model |
|---|---|
| `solar_reduction` | `effective_solar[h] = solar_kwh[h] × factor` |
| `minimum_battery_reserve` | `E_after[h] ≥ max(base minimum, directive minimum)` |
| `no_charge_window` | `charge[h] = 0` |
| `no_discharge_window` | `discharge[h] = 0` |
| `max_grid_window` | `grid[h] ≤ max_grid_kwh` |
| `no_op` | nothing — the note is a distractor |

One call handles all 1–3 notes. The request uses **strict structured outputs** (a
`json_schema` response format with `additionalProperties: false` and an enum-constrained
`directive_type`), so the model cannot invent a directive type or an extra field. Where a
model does not support strict schemas the client degrades to JSON mode automatically.

The prompt carries the directive schema, the end-exclusive time convention, the
"factor is the fraction *remaining*" rule, and **eight paraphrased few-shot examples**
chosen to span the failure modes rather than just the directive types: end-exclusive
boundaries, reduction-vs-remaining percentages, percent-of-capacity reserves, windows
that wrap past midnight, durations, single-hour mentions, multi-note scenarios, and
distractors that must stay `no_op`.

### Guardrails (`app/guardrails/`)

LLM output is untrusted structured data. Every entry is repaired rather than rejected,
because a model that gets `applies` backwards has still understood the note:

- **Structure** — exactly one entry per note, `note_index` 0..N-1; missing entries
  synthesized as `no_op`; unknown `directive_type` (with alias normalization) → `no_op`;
  extra adjustment keys stripped; `applies` force-corrected; explanation bounded.
- **Hour arbitration** (`timeparse.py`) — a deterministic parser reads the window
  straight from the note text (`from X until Y`, `between X and Y`, `noon`/`midnight`,
  `HH:MM`, `for N hours starting at X`, windows wrapping past midnight). When it finds
  exactly one unambiguous window it **overrides** the model's hours; when it finds
  nothing or is ambiguous, the model's hours stand. This is what fixes the most common
  LLM error on this task — an off-by-one on the end-exclusive convention.
- **Numeric normalization** (`numbers.py`) — `factor` is the fraction remaining, so
  "an 80% reduction" → `0.2` while "drops to 20%" → `0.2` and "reduced by one fifth" →
  `0.8`; percent-of-capacity reserves resolve against `capacity_kwh`; word fractions
  ("half", "a fifth", "three-quarters") are handled; everything is range-clamped and
  checked for NaN/Infinity.

### Optimizer (`app/optimizer/`)

The problem is a pure linear program — linear objective, linear constraints, continuous
variables — so HiGHS returns the exact global optimum. 96 variables (`grid`,
`solar_used`, `charge`, `discharge` for each of 24 hours):

```
minimize   Σ_h tariff[h] · grid[h]
subject to grid[h] + solar_used[h] + discharge[h] − charge[h] = demand[h]   ∀h
           Σ_h (charge[h] − discharge[h]) = 0                        (neutrality)
           Σ_{k≤h} (charge[k] − discharge[k]) ≤ capacity − initial          ∀h
          −Σ_{k≤h} (charge[k] − discharge[k]) ≤ initial − active_min[h]     ∀h
           0 ≤ grid[h] ≤ max_grid[h],  0 ≤ solar_used[h] ≤ effective_solar[h]
           0 ≤ charge[h] ≤ rate (0 in a no-charge window), likewise discharge
```

Post-solve conditioning nets charge against discharge (a degenerate optimum can return
both in one hour, which is a schema violation), clamps and rounds to 3 decimals,
**forward-simulates** `battery_energy_after_kwh` from the rounded actions rather than
emitting solver state, and recomputes all three totals from the rounded plan. The solve
plus validation costs ~1.5 ms; essentially all request latency is the LLM call.

### Replay validator (`app/validator/replay.py`)

Our own private clone of the judge, run on every response before it is returned. It
re-derives effective solar, reserves, windows and caps from the directives **itself**
rather than reusing the optimizer's compiler, so a bug in the compiler surfaces as a
validation failure instead of being replayed back at us. If validation fails, one repair
pass runs; if that fails too, a conservative always-valid fallback schedule is emitted.

### Failure handling

Every branch ends in a schema-valid `200`:

```
primary model (N attempts) → one repair call with the validator's complaint
  → second model, same provider → failover provider (if configured)
  → rule-based deterministic interpreter
```

A rate-limit (429) skips straight to the next rung instead of burning retries. A
wall-clock budget caps the whole stage so a hanging provider cannot exceed the judge's
30s limit. Interpretations are cached (LRU 512, keyed on the notes and battery capacity).
The deterministic interpreter is a safety net for a provider outage, never the main path —
the LLM remains the primary interpreter, as the rules require.

### Latency

The LP solve plus replay validation costs ~1.5 ms; essentially all request latency is the
single LLM call. Two things keep p95 inside budget:

- **Startup warm-up.** Each worker opens the provider connection and learns the model's
  parameter profile in a background thread at boot, so the first judged request does not
  pay for DNS, the TLS handshake and capability discovery. This measurably moved p95 from
  5.2 s to 3.0 s. It never blocks readiness and never runs inside `/health`.
- **Interpretation cache.** Repeated or related scenarios return in ~0.01 ms.

---

## 5. Deployment (Railway) and Docker fallback

### Railway (primary)

The repository deploys to Railway as-is; `railway.json` pins the builder, start command
and health check.

1. **New Project → Deploy from GitHub repo**, pointing at this repository.
2. **Variables** → set:
   - `OPENAI_API_KEY` = your key
   - `PORT` = `8000`
   - optionally `FALLBACK_PROVIDER_API_KEY` (+ base URL and model) for cross-provider failover
3. **Networking → Generate Domain**, target port `8000` (must match `PORT`).
4. Confirm: `curl https://<your-app>.up.railway.app/health` → `{"status":"ok"}`

`railway.json` sets `healthcheckPath: /health` with a 60 s timeout, which matches the
readiness requirement, and restarts on failure up to 3 times.

### Docker fallback image

**The judge's command** — pull and run, nothing else needed:

```bash
docker pull alvi00/team_charlie_kirk:preli-v1

docker run --rm -p 8000:8000 -e OPENAI_API_KEY=<your-key> \
  alvi00/team_charlie_kirk:preli-v1

curl localhost:8000/health        # {"status":"ok"}
```

To build and publish it:

```bash
docker build -t alvi00/team_charlie_kirk:preli-v1 .
docker login
docker push alvi00/team_charlie_kirk:preli-v1
```

Verified locally on `python:3.11-slim`:

| Check | Result |
|---|---|
| Image size | 557 MB |
| `/health` ready after start | **2 s** (requirement: < 60 s) |
| Public sample cases in-container | 10/10 interpretation · 10/10 valid · ratio 1.000 · p95 2.9 s |
| Starts with **no** API key | yes — serves a valid schedule via the deterministic interpreter |
| `.env` present inside the image | no |
| Key material anywhere in the image | none |
| Platform | `linux/amd64` |
| Anonymous pull (no Docker Hub account) | verified — registry manifest returns HTTP 200 |

- Exposed port: **8000**; the container binds `0.0.0.0` and honours `$PORT`.
- **No secrets are baked into the image.** `OPENAI_API_KEY` is supplied at run time with
  `-e`, and `.dockerignore` excludes `.env` from the build context. The only baked
  environment variables are `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED` and `PORT=8000`.
- Image tag: `alvi00/team_charlie_kirk:preli-v1` (**public**, pullable without credentials)
- Digest: `sha256:974bcf67ebf696751f352248167499ee9d7aff4a093fff990a7f418457a73459`
- Pinned-by-digest pull: `docker pull alvi00/team_charlie_kirk@sha256:974bcf67ebf696751f352248167499ee9d7aff4a093fff990a7f418457a73459`

---

## 6. Dependencies, limitations and secret handling

### Dependencies

| Package | Role |
|---|---|
| FastAPI + Uvicorn | HTTP service and ASGI server |
| Pydantic v2 + pydantic-settings | Request/response contract and env configuration |
| SciPy (HiGHS) + NumPy | Linear programming |
| httpx | Chat Completions calls (used directly for precise timeout control) |
| python-dotenv | Local `.env` loading |

External services: **OpenAI** for LLM inference, with an optional second
OpenAI-compatible provider for failover. Development was assisted by an AI coding
assistant (Claude); the architecture, formulation and guardrail logic are the team's own.

### Known limitations

- **Provider rate limits.** On a low tokens-per-minute tier, rapid back-to-back requests
  can be throttled. The service degrades cleanly — second model, failover provider, then
  the deterministic interpreter — and still returns a valid schedule.
- Unusual phrasings with no explicit clock reading (for example "from one until three"
  with no am/pm) are left to the LLM; the deterministic parser declines rather than guess.
- Vague time words ("this afternoon") with no clock time are treated as `no_op`, since no
  specific hours can be justified.
- Grid export and round-trip battery efficiency losses are not modelled — neither is part
  of this challenge.
- Scenarios whose directives are genuinely infeasible (mutually contradictory hard
  constraints, which the Problem Statement excludes from scoring cases) fall back to a
  conservative schedule that satisfies energy balance, battery bounds, rate limits and
  end-of-day neutrality, even if an impossible cap cannot be met.

### Secret handling

- Configuration is read from environment variables only. `.env` is gitignored and
  excluded from the Docker build context; `.env.example` carries names, never values.
- The API key is used only in an `Authorization` header. No code path logs it.
- Provider error bodies are truncated and scrubbed for key-shaped strings
  (`sk-…`, `sk-proj-…`, `gsk_…`) before they reach a log line.
- A global exception handler returns `{"error":"internal_error"}` with no stack trace and
  no provider message; details are logged server-side only.
- `tests/test_api.py` asserts that no response contains a stack trace or key material.

---

## 7. Repository layout

```
Team_Charlie_Kirk/
├── app/
│   ├── main.py               # FastAPI app, routes, pipeline, exception handlers
│   ├── config.py             # env configuration (no secrets in code)
│   ├── schemas.py            # Pydantic v2 request/response contract
│   ├── llm/
│   │   ├── client.py         # Chat Completions client, capability negotiation, cache
│   │   ├── prompt.py         # system prompt + response schema + few-shot examples
│   │   └── interpreter.py    # LLM call → directives, with the failure ladder
│   ├── guardrails/
│   │   ├── validate.py       # deterministic validation/repair of LLM output
│   │   ├── timeparse.py      # deterministic hour-window parser (arbitration)
│   │   ├── numbers.py        # percent→factor, %-of-capacity→kWh, clamping
│   │   └── fallback.py       # rule-based interpreter (provider-outage safety net)
│   ├── optimizer/
│   │   ├── model.py          # directive → constraint compilation
│   │   └── solve.py          # LP build, HiGHS solve, post-solve conditioning
│   ├── validator/replay.py   # judge-equivalent replay + safe fallback plan
│   └── summary.py            # deterministic plan_summary
├── tools/verify_model.py     # confirm the key + model id before deploying
├── tests/
│   ├── run_public_cases.py   # scoreboard: offline and api modes
│   ├── test_optimizer.py
│   ├── test_guardrails.py
│   └── test_api.py
├── data/                     # public sample case pack
├── railway.json              # Railway builder, start command, health check
├── Procfile                  # generic PaaS start command
├── Dockerfile
├── requirements.txt
└── .env.example
```
