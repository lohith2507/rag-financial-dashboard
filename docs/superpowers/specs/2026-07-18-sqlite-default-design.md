# SQLite as Default Database — Design Amendment

**Date:** 2026-07-18  
**Status:** Approved  
**Amends:** `docs/superpowers/specs/2026-07-12-rag-financial-dashboard-design.md`

## Decision

Use **SQLite** as the default database for this portfolio build. PostgreSQL + pgvector remain a documented **production-scale swap**, not required to run the demo.

## Motivation

Local PostgreSQL setup (role/DB creation) blocked Plan 1. The dataset is small (~169 synthetic transactions), single-user, and cosine similarity is already computed in Python — so a server database adds setup cost without changing the hybrid RAG story.

## Storage

| Item | Choice |
|---|---|
| Default URL | `sqlite:///./data/findb.db` (relative to `backend/` cwd) |
| File location | `backend/data/findb.db` (gitignored) |
| Driver | SQLAlchemy built-in SQLite (no `psycopg`, no `pgvector`) |
| Embeddings | `JSON` column storing `list[float]` length **1024** |
| Similarity | Python cosine (unchanged) |
| Schema tool | Alembic baseline migration |
| Tables | `transactions`, `insights`, `chat_messages` (same fields as before) |

## Config / deps

- `Settings.database_url` default → SQLite URL above.
- `.env.example` matches.
- Remove `psycopg[binary]` and `pgvector` from `backend/pyproject.toml`.
- Engine uses `connect_args={"check_same_thread": False}` for SQLite (needed once FastAPI shares the engine across threads).

## Plan 1 task changes

- **Task 3:** Ensure `data/` exists and the SQLite file accepts a connection (replace Postgres poller / `psql` prerequisite).
- **Task 4:** Models/migration use `JSON` for `embedding`; test cleanup uses `DELETE FROM …` (SQLite has no `TRUNCATE … RESTART IDENTITY`).
- **Task 8:** Unchanged pipeline logic; stores embeddings as JSON lists.

## Unchanged

- Synthetic generator, Groq/NVIDIA providers.
- Hybrid agent tools (`aggregate_spend`, etc.) — still SQLAlchemy SQL.
- Plans 2–4 feature scope (chat, insights, dashboard, eval).

## Production-scale swap (out of scope)

Swap `DATABASE_URL` to Postgres, change `embedding` to a native array/vector type (or keep JSON), optionally add pgvector for ANN search. Application tool interfaces stay the same.
