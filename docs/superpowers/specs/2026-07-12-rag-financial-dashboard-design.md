# AI-Powered Financial Insights Dashboard — Design

**Date:** 2026-07-12
**Status:** Approved (design), pending implementation plan
**Type:** Portfolio / resume project

## Overview & goals

A full-stack app where a user explores synthetic-but-realistic financial data through:

1. **RAG chat** grounded in their transactions
2. **AI-generated plain-English insights**
3. **Anomaly detection** with LLM explanations
4. A **visual dashboard**

Constraints and priorities:

- **Free to run.** No paid infrastructure. LLM inference offloaded to free cloud APIs (Groq, NVIDIA NIM).
- **Live-demoable.** Because inference is cloud-hosted, a live demo link is feasible.
- **One-command startup** via Docker Compose.
- **Single user, no auth** for the demo (multi-user is a noted future extension) to keep scope finishable solo.
- **Honest, defensible metrics** — the "+40% accuracy" claim is *measured* by an eval harness, not asserted.

## Architecture

```
React (Vite + TS, Tremor)  ──HTTP──>  FastAPI backend
                                        │
      ┌──────────────────────────────────┼────────────────────────────┐
      │                                  │                              │
 Postgres + pgvector             Groq (chat/gen)            NVIDIA NIM (embeddings)
 (transactions + embeddings)     llama-3.3-70b              nv-embedqa-e5-v5
```

### Backend modules (each with one responsibility)

- `data/generator.py` — synthetic transaction generator (with planted anomalies for verifiable detection)
- `ingest/` — categorize + embed transactions → store in pgvector
- `rag/agent.py` — the hybrid agent (tool router)
- `rag/tools.py` — `aggregate_spend`, `semantic_search`, `get_anomalies`, `list_transactions`
- `insights/` — plain-English summary generation
- `anomalies/` — statistical detection (z-score / IQR) + LLM explanation
- `providers/` — pluggable LLM + embedding interface (Groq, NVIDIA, Ollama, OpenAI)
- `api/` — FastAPI routers

## Environment adaptation (2026-07-12)

The dev machine has no Docker and its local PostgreSQL 18 lacks the `pgvector` extension (hard to build on Windows without Docker). Adopted decision for the initial build: use the **local PostgreSQL 18**, store embeddings as a **`double precision[]` array column**, and compute **cosine similarity in Python** (the synthetic dataset is small, so brute-force search is instant). `pgvector` + Docker containerization remain the documented **production-scale swap** but are out of scope for the current plans. The rest of the design below is unchanged; read "pgvector" as "array column + Python cosine" for now.

## Data model (Postgres + pgvector)

**`transactions`**
- `id`, `date`, `merchant`, `amount`, `category`, `description`
- `is_anomaly` (boolean, seeded ground truth for eval)
- `embedding vector(1024)`

**`insights`**
- `id`, `period`, `summary_text`, `generated_at`

**`chat_messages`**
- `id`, `session_id`, `role`, `content`, `tool_calls` (jsonb), `created_at`

Single-user demo — no auth. Multi-user support is an explicit future extension.

## Core: hybrid agentic RAG (approach B)

**Key insight:** naive vector RAG is unreliable for quantitative questions. "How much did I spend on food in June?" is an aggregation (SQL) problem, not a semantic-similarity problem. The agent routes each question to the right tool.

The agent receives a question plus a toolset and decides which tool(s) to call:

- **`aggregate_spend(category?, start, end)`** → real **SQL SUM / GROUP BY**. Accurate numbers.
- **`semantic_search(query, k)`** → **pgvector** cosine search over embeddings. Fuzzy questions ("subscriptions I forgot about", "weird purchases").
- **`get_anomalies(start?, end?)`** → returns flagged outliers.
- **`list_transactions(filters)`** → precise lookups.

**Flow:** question → LLM tool-selection (Groq function-calling) → execute tool(s) against Postgres → LLM composes a grounded answer citing retrieved rows. Numeric questions never rely on the LLM doing math from fuzzy context.

### The "+40%" claim, made honest & measurable

Build a small eval set (~20–30 Q&A pairs with known-correct answers derived from the seeded data). Run it against **pure vector RAG (A)** vs **hybrid agentic RAG (B)**. The measured accuracy delta *is* the resume number — real, defensible, reproducible. This is the single strongest interview asset in the project.

## Features

- **RAG chat** — chat UI → `/chat` endpoint → agent. Surfaces which tools were used ("Searched 1,240 transactions", "Summed category=Food").
- **AI insights** — `/insights` generates a monthly summary (top categories, trends, notable changes) via Groq, grounded in aggregated stats.
- **Anomaly detection** — statistical pass (z-score per category) flags outliers; Groq explains *why* each is unusual. Seeded known anomalies make detection verifiable.
- **Dashboard** — Tremor charts: spend-over-time, category breakdown, anomaly highlights, insight cards.

## Frontend

React + Vite + TypeScript, TanStack Query for data fetching, Tailwind + Tremor for the dashboard, a simple chat panel. Clean single-page app, dark-mode friendly.

## Infra & deployment

- **Docker Compose**: `db` (pgvector image), `backend`, `frontend`. `docker compose up` → working app.
- Secrets via **`.env` (gitignored)**; `.env.example` committed with placeholders.
- **API keys are never committed.** Keys shared during design must be rotated before use.
- Optional **GitHub Actions**: lint + tests on push.
- Optional live-demo path: backend on Render / Fly free tier + managed Postgres, since inference is offloaded to Groq/NVIDIA.

## Testing

- **Unit:** generator, tools (SQL correctness), anomaly math, provider adapters (mocked).
- **Integration:** `/chat` end-to-end against a seeded test DB.
- **Eval harness** (A vs B accuracy) doubles as a test and the metrics source.
- TDD where it fits — tools and detection logic especially.

## Suggested milestones

1. Repo scaffold + Docker Compose + `.gitignore` / `.env.example` + DB schema
2. Synthetic generator + ingestion (categorize + embed)
3. Provider layer (Groq + NVIDIA) with pluggable interface
4. RAG tools + agent (approach B)
5. Eval harness (A vs B) → capture the accuracy number
6. Insights + anomaly features
7. React dashboard + chat UI
8. Polish: README, demo recording, optional live deploy

## Explicit non-goals (YAGNI)

- Multi-user accounts / auth (future extension)
- Real bank integrations (Plaid etc.) — synthetic generator only
- Payment/transaction writes — read-and-analyze only
- Mobile app

## Tech stack summary

| Layer | Choice |
|---|---|
| Backend | Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic |
| Database | PostgreSQL + pgvector |
| LLM (chat/gen) | Groq (llama-3.3-70b) — pluggable |
| Embeddings | NVIDIA NIM (nv-embedqa-e5-v5) — pluggable |
| Frontend | React + Vite + TypeScript, TanStack Query, Tailwind, Tremor |
| Infra | Docker Compose, optional GitHub Actions CI |
