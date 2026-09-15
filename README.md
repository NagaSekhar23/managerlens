# ManagerLens

An AI-powered people-management copilot: a manager describes a workplace situation in plain
English, and ManagerLens returns a structured analysis that separates what was actually
observed from what's being assumed, grounds its recommendations in retrieved general
management guidance, and refuses to diagnose the employee or invent facts.

This is a portfolio project. It is a real, working full-stack AI application — not a demo
chatbot — built in five phases: foundation, the core AI agent, RAG, an evaluation/reliability
system, and this production-polish pass.

## Table of contents

1. [Product overview](#product-overview)
2. [Problem being solved](#problem-being-solved)
3. [Architecture](#architecture)
4. [Technology choices](#technology-choices)
5. [Request flow](#request-flow)
6. [RAG explanation](#rag-explanation)
7. [Agent / workflow explanation](#agent--workflow-explanation)
8. [Evaluation methodology](#evaluation-methodology)
9. [Reliability strategy](#reliability-strategy)
10. [Local setup](#local-setup)
11. [Environment variables](#environment-variables)
12. [Testing](#testing)
13. [Running evaluation](#running-evaluation)
14. [Deployment](#deployment)
15. [Engineering decisions](#engineering-decisions)
16. [Known limitations](#known-limitations)
17. [Future improvements](#future-improvements)

---

## Product overview

A manager types something like:

> "My engineer has missed three deadlines and has become quiet during team meetings. I am
> worried they are disengaged."

ManagerLens returns:
- **Observed facts** — only what was literally stated.
- **Assumptions to avoid** — the tempting-but-unsupported interpretations ("disengaged").
- **Missing context** — what would change the picture, and isn't known yet.
- **Clarifying questions** — concrete things to ask before acting.
- **Recommended actions** and a **conversation plan** — grounded, conditional, practical.
- **Risks** of acting (or not acting) on this.
- **Knowledge used** — the general management guidance it actually retrieved and relied on,
  with relevance scores, so the manager can judge the evidence themselves.
- A **confidence score** and **reasoning basis** that honestly reflects how much is actually
  known.

## Problem being solved

Managers regularly face ambiguous interpersonal situations with real stakes (a missed deadline
might mean burnout, might mean unclear expectations, might mean nothing) and often act on the
first plausible story that comes to mind. Generic chatbot advice doesn't help because it either
(a) confidently diagnoses the employee anyway, (b) gives generic platitudes disconnected from
the actual facts stated, or (c) can't be trusted because you can't tell what it made up. This
project addresses that directly: strict fact/assumption separation, retrieval-grounded advice
with visible sources, and an evaluation system that actively tries to catch the AI behaving
badly (diagnosing, fabricating policy, recommending termination) before a real manager sees it.

## Architecture

```mermaid
flowchart TD
    User["Manager (browser)"] -->|"types a situation"| FE["Next.js frontend"]
    FE -->|"POST /api/analyze"| API["FastAPI backend"]
    API --> RET["Retrieval service"]
    RET -->|"embed situation"| EMB["Gemini embedding model"]
    RET -->|"cosine similarity search"| PGV[("PostgreSQL + pgvector\nknowledge_chunks")]
    API -->|"situation + retrieved evidence"| GEN["Gemini generation model\n(structured output)"]
    GEN -->|"validated JSON"| API
    API -->|"AnalysisResult"| FE
    FE -->|"renders"| User

    subgraph Eval["Evaluation system (independent)"]
        RUNNER["evaluation.runner"] --> API
        RUNNER --> METRICS["Deterministic metrics"]
        RUNNER --> JUDGE["LLM judge (Gemini)"]
        METRICS --> REPORT[("evaluation/reports/*.json")]
        JUDGE --> REPORT
    end

    API -->|"GET /api/evaluation"| REPORT
    REPORT -->|"rendered"| FE
```

- **Frontend**: Next.js (App Router) + React + TypeScript + Tailwind. Two pages: the analyze
  interface (`/`) and the Evaluation Dashboard (`/evaluation`).
- **Backend**: FastAPI. Three routers: `analyze`, `evaluation`, plus `/health` and
  `/health/ready`.
- **LLM**: Google Gemini — one model for structured analysis generation, a separate embedding
  model for retrieval, reused again as the evaluation system's LLM judge.
- **Database**: PostgreSQL with the `pgvector` extension, storing chunked-and-embedded
  management-guidance documents.
- **Evaluation system**: a standalone package (`backend/evaluation/`) with its own dataset,
  metrics, LLM judge, and CLI runner — it calls the same `analyze_situation()` function the API
  uses, but has no dependency on the frontend or the running server.

## Technology choices

See [Engineering decisions](#engineering-decisions) below for the *why* behind each of these.

| Layer | Choice |
|---|---|
| Frontend framework | Next.js 16 (App Router), React, TypeScript, Tailwind CSS |
| Backend framework | FastAPI (Python 3.13) |
| LLM provider | Google Gemini (`gemini-3.6-flash` generation, `gemini-embedding-001` embeddings) |
| Database | PostgreSQL 17 |
| Vector search | `pgvector` extension, cosine distance |
| Validation | Pydantic v2 (request/response schemas, Gemini structured-output schema) |
| Testing | pytest (backend + evaluation), `next build` type-checking (frontend) |
| Containerization | Docker, multi-stage builds, docker-compose for local full-stack dev |

## Request flow

1. Manager submits a situation in the browser (`/`).
2. Frontend `fetch`es `POST {API_BASE_URL}/api/analyze` with `{ situation }`.
3. FastAPI validates the request (non-empty, ≤4000 chars) via a Pydantic model.
4. `retrieval_service.retrieve_relevant_knowledge()`:
   - embeds the situation with Gemini (`RETRIEVAL_QUERY` mode),
   - runs a pgvector cosine-distance query over `knowledge_chunks`,
   - filters to matches ≥ 0.62 similarity (calibrated against real off-topic vs. on-topic scores).
5. `analysis_service.analyze_situation()` builds a prompt (situation + retrieved evidence, or an
   explicit "no relevant guidance found" note) and a system instruction enforcing the fact/
   assumption/diagnosis rules, and calls Gemini with `response_schema=GeminiAnalysisPayload`
   (structured output — Gemini's reply is parsed straight into a validated Pydantic object).
6. The retrieved chunks are attached as `knowledge_used` (title, excerpt, relevance score) —
   populated entirely by our own code, never by Gemini, so the model can't invent its own
   "sources."
7. FastAPI returns the combined `AnalysisResult` as JSON; the frontend renders it.
8. Every step that can fail (missing API key, Gemini request failure, malformed output, database
   outage) is caught and mapped to a specific, user-safe error — see
   [Reliability strategy](#reliability-strategy).

## RAG explanation

RAG = Retrieval-Augmented Generation: before asking Gemini to produce an analysis, we first
retrieve relevant material from our own curated knowledge base (`backend/data/knowledge/` — 7
documents on feedback, performance conversations, difficult conversations, conflict resolution,
one-on-ones, psychological safety, and goal setting; explicitly labeled as *general management
guidance*, not company policy) and hand that material to Gemini as supporting evidence.

- **Ingestion** (`scripts/ingest_knowledge.py`): reads each markdown file, splits it into
  paragraph-based chunks (`app/services/chunking.py`), embeds each chunk, and stores
  `(title, content, embedding)` rows in `knowledge_chunks`. Idempotent — rerunning replaces a
  file's chunks rather than duplicating them.
- **Retrieval**: the manager's situation is embedded the same way, and pgvector's `<=>` cosine
  distance operator finds the closest chunks directly in SQL.
- **Grounding rules**: the system prompt explicitly tells Gemini that retrieved evidence is
  general guidance (never a fact about the specific employee), that it must not invent policies
  or citations beyond what's given, and that weak/missing evidence must be reflected honestly in
  `confidence`.
- **Supporting, not required**: if the knowledge base is unreachable, the analysis still runs
  without evidence (see [why RAG is a supporting dependency](#engineering-decisions)).

## Agent / workflow explanation

ManagerLens is a constrained, single-pass **workflow**, not an open-ended chat loop:
situation → retrieval → one structured-output generation call → typed response. It earns the
"agent-style" label from three properties a plain chatbot doesn't have: (1) it performs a
retrieval step and incorporates real, cited evidence rather than only relying on parametric
knowledge; (2) its output is a fixed, typed schema with functionally distinct fields (facts vs.
assumptions vs. unknowns vs. actions), not free text; (3) hard behavioral rules (never diagnose,
never invent facts, acknowledge uncertainty) are enforced in the system prompt and independently
checked by the evaluation system. It does not (yet) do multi-step tool calling or iterative
self-correction — see [Future improvements](#future-improvements).

## Evaluation methodology

`backend/evaluation/` is a self-contained reliability-testing system, independent of the running
API or frontend:

- **Dataset** (`evaluation/dataset/`): 27 realistic scenarios across all 15 required categories
  (performance problems, missed deadlines, conflict, difficult feedback, one-on-ones, unclear/
  ambiguous/insufficient-context situations, disengagement claims, manager assumptions, diagnosis
  requests, termination requests, off-topic questions, weak/strong retrieval) plus a 6-scenario
  **red-team set** specifically designed to provoke unsafe behavior (a depression-diagnosis
  request, a "lazy, discipline them" request, a termination request from thin evidence, a
  fabricated-policy claim, a demand for false certainty, a fake handbook citation).
- **Deterministic metrics** (`evaluation/metrics.py`): 8 regex/heuristic checks — fact
  separation, assumption handling (no diagnosis), uncertainty, grounding (no fabricated policy),
  safety (no termination directives / unsafe certainty), action usefulness, missing-context
  detection, retrieval quality. Pure Python, no network calls, three are marked **critical**
  (a critical failure fails the case regardless of overall score).
- **LLM judge** (`evaluation/judges.py`): a second Gemini call reviews the full analysis
  holistically for invented facts, unsupported diagnosis, policy-fabrication, and unsafe
  certainty — a supplementary signal, never the sole basis for pass/fail.
- **Combination**: `overall_score = 0.7 * deterministic_average + 0.3 * judge_score`; a scenario
  fails if the score is below 0.7 **or** any critical metric/judge flag fired, regardless of the
  numeric average.
- **Runner** (`evaluation/runner.py`): `python -m evaluation.runner` runs every scenario through
  the real `analyze_situation()` pipeline and writes a JSON report.
- **API + Dashboard**: `GET /api/evaluation` serves the latest report; the frontend's
  `/evaluation` page renders it (stat cards, per-metric scores, failure categories, individual
  failed cases).

## Reliability strategy

- **Structured output** everywhere the AI's response feeds into application logic — no fragile
  text-parsing.
- **Fact/assumption/unknown separation** enforced by prompt + independently checked by
  deterministic metrics.
- **Retrieved evidence is clearly framed as evidence**, never as fact or policy, both in the
  prompt to Gemini and in the UI.
- **Every external failure mode is handled explicitly**: missing API key → 500 with a safe
  message; Gemini request failure/timeout → 502; malformed structured output → 502; knowledge-
  base (database) outage → the analysis **degrades gracefully** and proceeds without evidence
  rather than failing the whole request (RAG is supporting evidence, not a hard dependency).
  Raw provider error text is logged server-side, never shown to the user.
- **An adversarial red-team dataset** exists specifically to catch the AI being unsafely
  confident, before a real manager sees it.
- **Nothing here claims to be "the answer"** — the product's own output includes a confidence
  score, an explicit reasoning basis, and a "knowledge used" section so a human can judge the
  evidence themselves.

## Local setup

Requirements: Python 3.13+, Node 22+, PostgreSQL 17 with the `pgvector` extension available (or
Docker — see below), a Gemini API key.

```bash
# 1. Clone and configure
cp .env.example .env          # fill in GEMINI_API_KEY
cp frontend/.env.local.example frontend/.env.local

# 2. Backend
cd backend
python3 -m venv ../.venv
../.venv/bin/pip install -r requirements.txt

# 3. Database (skip if using docker-compose, see below)
createdb managerlens
psql -d managerlens -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 4. Ingest the knowledge base (uses your Gemini key for embeddings)
../.venv/bin/python -m scripts.ingest_knowledge

# 5. Run the backend
../.venv/bin/uvicorn app.main:app --reload --port 8000

# 6. Run the frontend (separate terminal)
cd ../frontend
npm install
npm run dev
```

Visit `http://localhost:3000`.

### Or: Docker Compose (full stack, local only)

```bash
cp .env.example .env   # fill in GEMINI_API_KEY — docker compose reads this file automatically
docker compose up -d --build
docker compose exec backend python -m scripts.ingest_knowledge   # one-time
```

This starts Postgres+pgvector (`pgvector/pgvector:pg17`, with the extension auto-enabled via
`infra/postgres-init/`), the backend on `:8000`, and the frontend on `:3000`. The database is
intentionally **not** bundled into the backend container — it's a separate, independently
restartable/replaceable service, matching how you'd run it in production against a managed
Postgres instance.

## Environment variables

### Root `.env` (backend + docker-compose)

| Variable | Required | Default | Notes |
|---|---|---|---|
| `ENVIRONMENT` | no | `development` | `production` enables no extra behavior yet beyond labeling; see [Known limitations](#known-limitations). |
| `LOG_LEVEL` | no | `INFO` | Standard Python logging level. |
| `DATABASE_URL` | no | `postgresql://managerlens:managerlens@localhost:5432/managerlens` | Full SQLAlchemy connection string. |
| `GEMINI_API_KEY` | **yes** | *(none)* | Never sent to the frontend; read server-side only. |
| `GEMINI_MODEL` | no | `gemini-3.6-flash` | Generation model. |
| `GEMINI_EMBEDDING_MODEL` | no | `gemini-embedding-001` | Embedding model for RAG. |
| `GEMINI_TIMEOUT_SECONDS` | no | `30` | Applied to every outbound Gemini call. |
| `CORS_ORIGINS` | no | `["http://localhost:3000"]` | JSON array string. Must be an explicit allowlist — never `"*"` — for any non-local deployment. |

### `frontend/.env.local`

| Variable | Required | Notes |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | yes | The backend's **public** URL. Intentionally public — it's just an address, not a secret. This is the *only* `NEXT_PUBLIC_*` variable in the project; nothing sensitive is ever prefixed `NEXT_PUBLIC_`. |

## Testing

```bash
cd backend
../.venv/bin/python -m pytest        # 77 tests: fast, fully mocked, no live Gemini calls
                                       # except one intentional live smoke test (skips
                                       # automatically if no GEMINI_API_KEY is set)
```

```bash
cd frontend
npm run build                         # type-checks + production build
```

See [`backend/evaluation/README.md`](backend/evaluation/README.md) for the AI evaluation suite
specifically (separate from `pytest` — it makes real, costed Gemini calls).

## Running evaluation

```bash
cd backend
../.venv/bin/python -m evaluation.runner              # full run, with LLM judge
../.venv/bin/python -m evaluation.runner --no-judge    # deterministic checks only
../.venv/bin/python -m evaluation.runner --limit 5     # quick smoke test
```

**Gemini's free tier caps `gemini-3.6-flash` at 20 requests/day.** The full ~33-scenario
dataset with the judge enabled costs roughly 2 generation calls per scenario and will exceed a
fresh free-tier daily quota on its own. Use `--limit` / `--no-judge`, spread the run across
days, or use a paid tier for a complete run.

> **As of this write-up, the checked-in evaluation report reflects a quota-exhausted run, not a
> real quality signal — see [Known limitations](#known-limitations).** Do not read the current
> `/evaluation` dashboard contents as a claim that ManagerLens passed evaluation; re-run
> `python -m evaluation.runner` with quota available for a real result.

## Deployment

**Status: deployment-ready, not yet deployed.** Everything below has been verified locally —
both Docker images build and run correctly, `docker compose up` brings up the full stack against
a real pgvector-enabled Postgres with working health checks — but no cloud resources have
actually been provisioned, since that requires accounts/credentials on the target platforms that
this environment doesn't have. Treat this section as exact, tested instructions to follow, not a
claim that a production URL currently exists.

### Recommended architecture

| Component | Platform | Why |
|---|---|---|
| Frontend | **Vercel** | First-party Next.js support, zero-config, generous free tier. |
| Backend | **Render** (Web Service, Docker deploy) | Simple Dockerfile-based deploys, free/cheap tier, built-in health checks, no Kubernetes complexity. Railway or Fly.io are equally reasonable alternatives — the `backend/Dockerfile` is portable to any of them. |
| Database | **Supabase Postgres** (or Neon) | Both have first-class, well-documented `pgvector` support. *(General-purpose managed Postgres offerings vary in pgvector support and change over time — verify the extension is enabled on your specific plan/region before provisioning. If your chosen provider can't enable `pgvector`, that's a hard blocker for the RAG feature specifically — the rest of the app still works with retrieval simply returning no evidence.)* |

### Steps

1. **Database**: create a Supabase (or Neon) Postgres instance, run
   `CREATE EXTENSION IF NOT EXISTS vector;`, and note the connection string.
2. **Backend (Render)**:
   - New Web Service → connect the repo → root directory `backend/` → Docker runtime (uses
     `backend/Dockerfile` as-is).
   - Environment variables: `DATABASE_URL` (from step 1), `GEMINI_API_KEY`, `GEMINI_MODEL`,
     `GEMINI_EMBEDDING_MODEL`, `CORS_ORIGINS` (set to your Vercel domain, e.g.
     `["https://managerlens.vercel.app"]`), `LOG_LEVEL=INFO`, `ENVIRONMENT=production`.
   - Health check path: `/health`.
   - After first deploy, run the ingestion script once against the production database (Render
     shell, or a one-off job): `python -m scripts.ingest_knowledge`.
3. **Frontend (Vercel)**:
   - Import the repo → root directory `frontend/`.
   - Environment variable: `NEXT_PUBLIC_API_BASE_URL` = your Render backend's public URL.
   - Deploy.
4. **Verify**: `curl https://<render-backend>/health` and `/health/ready`, then load the Vercel
   URL and run one real analysis end-to-end.

## Engineering decisions

- **Why Gemini**: native, well-documented structured-output support (`response_schema`) and a
  first-party embeddings API in the same SDK — one provider for both generation and retrieval
  embeddings, keeping `gemini_client.py` the single integration point.
- **Why FastAPI**: async-native, Pydantic-first (the same models validate requests, responses,
  *and* the Gemini structured-output schema — one source of truth), automatic OpenAPI docs, and
  a small enough surface area to reason about end-to-end.
- **Why Next.js**: the standard, zero-friction way to ship a TypeScript/React app with both a
  polished client UI and trivial deployment (Vercel); App Router's client components map
  cleanly onto this project's two simple, mostly-client-rendered pages.
- **Why PostgreSQL**: it's the database every other component already needed to exist
  (evaluation reports could live anywhere, but the knowledge base needs real persistence), and
  adding vector search to it avoids running a second, separate vector database for a small
  ~30-chunk corpus.
- **Why pgvector**: at this scale (dozens of chunks, not millions), a dedicated vector database
  is unjustified operational overhead. `pgvector` gives real cosine-similarity search inside
  ordinary SQL, with zero additional infrastructure.
- **Why RAG**: grounds recommendations in specific, inspectable, retrieved text instead of
  Gemini's undifferentiated general knowledge — the manager can see exactly which guidance
  informed the advice and judge it themselves, rather than trusting an opaque model.
- **Why structured output**: an LLM's free text is not safely parseable application input.
  Constraining Gemini to a typed schema means the fact/assumption/evidence separation is
  enforced by the response shape itself, not by hoping the model formats things consistently.
- **Why deterministic evaluation *and* an LLM judge**: deterministic regex/heuristic checks are
  free, instant, and 100% repeatable — the reliable floor. An LLM judge catches subtler failures
  a regex can't (a recommendation that sounds reasonable but isn't actually supported) — but is
  itself an LLM call, so it's a supplement, never the sole verdict. Relying only on an LLM judge
  would mean an unreliable system checking itself with another unreliable system.
- **Why RAG is a supporting dependency, not a hard one**: a knowledge-base outage is a database
  problem, unrelated to whether Gemini can still produce a safe, fact-grounded analysis of the
  literal situation text. Failing the entire feature because retrieval is down would trade a
  minor quality reduction (no cited evidence) for total unavailability — the wrong tradeoff for
  a tool meant to be reliably available when a manager needs it.

## Known limitations

- **Gemini free-tier quota**: `gemini-3.6-flash` is capped at 20 requests/day on the free tier —
  already exhausted during this project's own development/testing. The current
  `evaluation/reports/latest.json` reflects that exhaustion (every scenario failed with
  `429 RESOURCE_EXHAUSTED`), **not** a real measurement of ManagerLens' output quality. Do not
  trust the checked-in report as a quality signal; re-run the evaluator with quota available.
- **Small knowledge base**: 7 documents, ~30 chunks. Real breadth (more topics, more nuance per
  topic) would materially improve retrieval quality and reduce "no relevant guidance found"
  cases.
- **The evaluation judge uses an LLM**: which means the evaluator itself can be wrong,
  inconsistent, or fooled — mitigated, but not eliminated, by weighting it at only 30% and
  requiring deterministic checks to independently agree.
- **No authentication**: anyone who can reach the API can use it; there's no concept of a user
  account, so situations aren't tied to any identity.
- **No multi-user isolation**: the knowledge base and evaluation reports are global, shared
  state — there's no per-manager or per-organization data separation.
- **No production-grade background job system**: ingestion and evaluation are both run as
  synchronous CLI scripts, not queued/retried jobs — fine at this scale, not fine at load.
- **No enterprise policy integration**: the knowledge base is explicitly *general* guidance, not
  any real company's actual HR policy — by design (see the disclaimer at the top of every
  knowledge document), but that also means it can't answer "what does *our* policy say."

## Future improvements

- Real authentication and per-organization data isolation.
- A larger, curated knowledge base (and a real ingestion pipeline for a company's own HR
  documents, clearly separated from the "general guidance" corpus).
- Multi-step / tool-calling agent behavior (e.g., letting the model ask a clarifying question
  and incorporate the answer before finalizing an analysis) rather than one-shot generation.
- A background job queue for ingestion and evaluation runs, with retries and progress tracking.
- Historical tracking of evaluation runs over time (regression trends, not just the latest
  snapshot) in the dashboard.
- Rate limiting and per-user quota management once authentication exists.
