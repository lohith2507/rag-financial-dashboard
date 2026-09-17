# AI Financial Insights Dashboard

A full-stack portfolio app for exploring synthetic personal-finance data through **hybrid agentic RAG chat**, **statistical anomaly detection with LLM explanations**, **AI monthly insights**, and a **visual dashboard**. Numeric questions route to real SQL aggregations; fuzzy questions use semantic search — not naive vector-only RAG.

## Architecture

```
React (Vite + TS, Tremor)  ──HTTP──>  FastAPI backend
                                        │
      ┌──────────────────────────────────┼────────────────────────────┐
      │                                  │                              │
 SQLite (transactions + JSON embeddings) Groq (chat)          NVIDIA NIM (embeddings)
```

**Why hybrid RAG:** "How much did I spend on dining in May?" is a SQL `SUM` problem. Pure vector RAG stuffs similar rows into context and lets the LLM guess — unreliable for money. This app's agent calls `aggregate_spend` for totals and `semantic_search` only for fuzzy questions.

## Eval results (A vs B)

Run `python -m app.eval.run_eval --limit 25` after seeding to produce measured accuracy in `docs/eval-results.md`. Hybrid agentic RAG (B) vs pure-vector baseline (A) on generator-derived ground truth.

## Quickstart

### 1. Environment

Copy `.env.example` to `.env` and set your keys (never commit `.env`):

- `GROQ_API_KEY` — https://console.groq.com/keys
- `NVIDIA_API_KEY` — https://build.nvidia.com

Optional overrides (defaults match `.env.example`):

- `GROQ_MODEL` — chat model id (default `openai/gpt-oss-120b`)
- `NVIDIA_EMBED_MODEL` — embedding model id (default `nvidia/nv-embedqa-e5-v5`)
- `DATABASE_URL` — SQLAlchemy URL (default SQLite under `backend/data/findb.db`)

Default database: SQLite file at `backend/data/findb.db` (no Postgres install required).

### 2. Backend

```powershell
cd backend
pip install -e ".[dev]"
alembic upgrade head
python -m app.seed
uvicorn app.main:app --reload
```

### 3. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — Vite proxies API calls to http://localhost:8000.

With the backend running, check `GET http://localhost:8000/health` for liveness and open http://localhost:8000/docs for the interactive OpenAPI UI (chat, anomalies, insights, stats).

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.12+, FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic |
| Database | SQLite (default); Postgres + pgvector as production-scale swap |
| LLM | Groq (pluggable) |
| Embeddings | NVIDIA NIM (pluggable) |
| Frontend | React 18, Vite, TypeScript, TanStack Query, Tailwind 3, Tremor 3 |

## Project layout

```
backend/          FastAPI app, RAG agent, anomalies, insights, eval
frontend/         React dashboard + chat UI
docs/             Design specs, plans, eval-results.md
scripts/          DB readiness helper
```

## Testing

```powershell
cd backend && pytest
cd frontend && npm test -- --run
```

## Production-scale swaps

- **Database:** Postgres + pgvector for ANN search at scale (swap `DATABASE_URL`, embedding column type).
- **Infra:** Docker Compose with pgvector image, optional CI on GitHub Actions.
