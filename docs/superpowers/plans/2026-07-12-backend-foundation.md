# Backend Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the backend foundation — a seeded PostgreSQL+pgvector database full of synthetic, embedded financial transactions, reachable through a pluggable provider layer.

**Architecture:** FastAPI project with SQLAlchemy 2.0 models over Postgres+pgvector. A synthetic generator produces realistic transactions (with planted anomalies for later verification). A pluggable provider layer wraps Groq (chat) and NVIDIA NIM (embeddings) behind interfaces, so any provider is a drop-in swap. An ingestion pipeline embeds transactions and stores vectors.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2 + pydantic-settings, SQLAlchemy 2.0, Alembic, pgvector, Docker Compose, pytest, httpx.

## Global Constraints

- Python 3.12+.
- SQLAlchemy 2.0 style (typed `Mapped[...]`, `mapped_column`).
- Pydantic v2.
- Embedding vector dimension: **1024** (NVIDIA `nv-embedqa-e5-v5`).
- Secrets ONLY via environment / `.env`; `.env` MUST be gitignored. `.env.example` committed with placeholders. NEVER commit real API keys.
- All dependencies pinned in `pyproject.toml`.
- Every task ends with passing tests and a commit.
- Provider adapters are tested with mocked HTTP — tests never make real network calls.

---

### Task 1: Project scaffold, tooling, and secret hygiene

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: installable `app` package; `pytest` runnable from `backend/`.

- [ ] **Step 1: Create `.gitignore`** (secret hygiene first — before any key ever touches disk)

```gitignore
# Secrets
.env
*.env
!.env.example

# Python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.pytest_cache/
.mypy_cache/

# Node
node_modules/
dist/

# OS
.DS_Store
```

- [ ] **Step 2: Create `.env.example`** (placeholders only — no real keys)

```dotenv
# LLM (Groq) — get a key at https://console.groq.com/keys
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Embeddings (NVIDIA NIM) — get a key at https://build.nvidia.com
NVIDIA_API_KEY=your_nvidia_key_here
NVIDIA_EMBED_MODEL=nvidia/nv-embedqa-e5-v5

# Database
DATABASE_URL=postgresql+psycopg://finuser:finpass@localhost:5432/findb
```

- [ ] **Step 3: Create `backend/pyproject.toml`**

```toml
[project]
name = "fin-dashboard-backend"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "sqlalchemy>=2.0",
    "psycopg[binary]>=3.2",
    "pgvector>=0.3.6",
    "alembic>=1.14",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "httpx>=0.28",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "pytest-asyncio>=0.25", "respx>=0.22"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 4: Create package + test files**

`backend/app/__init__.py`:
```python
__version__ = "0.1.0"
```

`backend/tests/__init__.py`: (empty file)

`backend/tests/test_smoke.py`:
```python
from app import __version__


def test_version_present():
    assert __version__ == "0.1.0"
```

- [ ] **Step 5: Install and run the smoke test**

Run (from `backend/`):
```bash
pip install -e ".[dev]"
pytest tests/test_smoke.py -v
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add .gitignore .env.example backend/
git commit -m "feat: scaffold backend project with secret hygiene"
```

---

### Task 2: Settings loader

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/tests/test_config.py`

**Interfaces:**
- Consumes: env vars from Task 1's `.env.example` schema.
- Produces: `Settings` class and `get_settings() -> Settings`. Fields: `groq_api_key: str`, `groq_model: str`, `nvidia_api_key: str`, `nvidia_embed_model: str`, `database_url: str`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_config.py`:
```python
from app.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("NVIDIA_API_KEY", "n-key")
    monkeypatch.setenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")

    s = Settings()

    assert s.groq_api_key == "g-key"
    assert s.nvidia_embed_model == "nvidia/nv-embedqa-e5-v5"
    assert s.database_url.endswith("/db")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'`.

- [ ] **Step 3: Write minimal implementation**

`backend/app/config.py`:
```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    nvidia_api_key: str = ""
    nvidia_embed_model: str = "nvidia/nv-embedqa-e5-v5"
    database_url: str = "postgresql+psycopg://finuser:finpass@localhost:5432/findb"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/config.py backend/tests/test_config.py
git commit -m "feat: add settings loader"
```

---

### Task 3: Docker Compose with Postgres + pgvector

**Files:**
- Create: `docker-compose.yml`
- Create: `scripts/wait_for_db.py`

**Interfaces:**
- Consumes: `DATABASE_URL` env.
- Produces: a running Postgres on `localhost:5432` with the `vector` extension available (image ships it). DB `findb`, user `finuser`, password `finpass`.

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: finuser
      POSTGRES_PASSWORD: finpass
      POSTGRES_DB: findb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U finuser -d findb"]
      interval: 3s
      timeout: 3s
      retries: 10

volumes:
  pgdata:
```

- [ ] **Step 2: Create a readiness checker**

`scripts/wait_for_db.py`:
```python
"""Poll the database until it accepts connections, then exit 0."""
import sys
import time

import psycopg

from app.config import get_settings


def main() -> int:
    dsn = get_settings().database_url.replace("+psycopg", "")
    for _ in range(30):
        try:
            with psycopg.connect(dsn, connect_timeout=2):
                print("DB ready")
                return 0
        except Exception as exc:  # noqa: BLE001
            print(f"waiting for db: {exc}")
            time.sleep(2)
    print("DB never became ready", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Start the database and verify readiness**

Run (from `backend/`, with env loaded):
```bash
docker compose -f ../docker-compose.yml up -d
python ../scripts/wait_for_db.py
```
Expected: `DB ready` printed, exit 0.

- [ ] **Step 4: Verify the vector extension can be created**

Run:
```bash
docker compose -f ../docker-compose.yml exec db psql -U finuser -d findb -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT '1'::vector(3) IS NULL;"
```
Expected: command succeeds (prints `f`).

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml scripts/wait_for_db.py
git commit -m "feat: add postgres+pgvector via docker compose"
```

---

### Task 4: Database models, session, and Alembic baseline

**Files:**
- Create: `backend/app/db.py`
- Create: `backend/app/models.py`
- Create: `backend/alembic.ini`
- Create: `backend/migrations/env.py`
- Create: `backend/migrations/script.py.mako`
- Create: `backend/migrations/versions/0001_baseline.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_models.py`

**Interfaces:**
- Consumes: `get_settings().database_url`.
- Produces:
  - `Base` (DeclarativeBase), `engine`, `SessionLocal`.
  - `Transaction(id:int, date:date, merchant:str, amount:float, category:str, description:str, is_anomaly:bool, embedding:list[float]|None)` — table `transactions`, `embedding` is `Vector(1024)`.
  - `Insight(id:int, period:str, summary_text:str, generated_at:datetime)` — table `insights`.
  - `ChatMessage(id:int, session_id:str, role:str, content:str, tool_calls:dict|None, created_at:datetime)` — table `chat_messages`.

- [ ] **Step 1: Create the DB session module**

`backend/app/db.py`:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


engine = create_engine(get_settings().database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
```

- [ ] **Step 2: Create the models**

`backend/app/models.py`:
```python
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, Date, DateTime, Float, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

EMBED_DIM = 1024


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    merchant: Mapped[str] = mapped_column(String(200))
    amount: Mapped[float] = mapped_column(Float)
    category: Mapped[str] = mapped_column(String(80), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)


class Insight(Base):
    __tablename__ = "insights"

    id: Mapped[int] = mapped_column(primary_key=True)
    period: Mapped[str] = mapped_column(String(20), index=True)
    summary_text: Mapped[str] = mapped_column(Text)
    generated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    tool_calls: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 3: Initialize Alembic and write the baseline migration**

Run (from `backend/`):
```bash
alembic init migrations
```
Then set `sqlalchemy.url` handling in `backend/migrations/env.py` — replace its body with:
```python
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import get_settings
from app.db import Base
from app import models  # noqa: F401  (registers tables)

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url)
if config.config_file_name:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
```

Create `backend/migrations/versions/0001_baseline.py`:
```python
"""baseline schema

Revision ID: 0001_baseline
Revises:
Create Date: 2026-07-12
"""
import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("date", sa.Date, index=True),
        sa.Column("merchant", sa.String(200)),
        sa.Column("amount", sa.Float),
        sa.Column("category", sa.String(80), index=True),
        sa.Column("description", sa.Text),
        sa.Column("is_anomaly", sa.Boolean, server_default=sa.false()),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(1024), nullable=True),
    )
    op.create_table(
        "insights",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("period", sa.String(20), index=True),
        sa.Column("summary_text", sa.Text),
        sa.Column("generated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("session_id", sa.String(64), index=True),
        sa.Column("role", sa.String(16)),
        sa.Column("content", sa.Text),
        sa.Column("tool_calls", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("insights")
    op.drop_table("transactions")
```

- [ ] **Step 4: Create the test fixture**

`backend/tests/conftest.py`:
```python
import pytest
from sqlalchemy import text

from app.db import SessionLocal, engine


@pytest.fixture(autouse=True)
def _clean_transactions():
    """Truncate mutable tables between tests (DB must be migrated first)."""
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE transactions, insights, chat_messages RESTART IDENTITY"))
    yield


@pytest.fixture
def session():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
```

- [ ] **Step 5: Write the failing model test**

`backend/tests/test_models.py`:
```python
from datetime import date

from app.models import Transaction


def test_insert_transaction_with_embedding(session):
    txn = Transaction(
        date=date(2026, 6, 1),
        merchant="Whole Foods",
        amount=54.20,
        category="Groceries",
        description="grocery run",
        is_anomaly=False,
        embedding=[0.0] * 1024,
    )
    session.add(txn)
    session.commit()

    fetched = session.get(Transaction, txn.id)
    assert fetched.merchant == "Whole Foods"
    assert len(fetched.embedding) == 1024
```

- [ ] **Step 6: Migrate the DB, then run the test**

Run (from `backend/`, DB up from Task 3):
```bash
alembic upgrade head
pytest tests/test_models.py -v
```
Expected: `alembic upgrade head` succeeds; test PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/db.py backend/app/models.py backend/alembic.ini backend/migrations backend/tests/conftest.py backend/tests/test_models.py
git commit -m "feat: add db models, session, and alembic baseline"
```

---

### Task 5: Synthetic transaction generator

**Files:**
- Create: `backend/app/data/__init__.py`
- Create: `backend/app/data/generator.py`
- Create: `backend/tests/test_generator.py`

**Interfaces:**
- Consumes: nothing (pure function).
- Produces: `generate_transactions(months: int = 6, seed: int = 42) -> list[GeneratedTxn]` where `GeneratedTxn` is a Pydantic model with fields `date: date, merchant: str, amount: float, category: str, description: str, is_anomaly: bool`. Deterministic for a given seed. Plants a small number of anomalies (unusually large amounts) flagged `is_anomaly=True`.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_generator.py`:
```python
from app.data.generator import generate_transactions


def test_generator_is_deterministic():
    a = generate_transactions(months=3, seed=7)
    b = generate_transactions(months=3, seed=7)
    assert [t.model_dump() for t in a] == [t.model_dump() for t in b]


def test_generator_plants_anomalies():
    txns = generate_transactions(months=6, seed=42)
    anomalies = [t for t in txns if t.is_anomaly]
    assert 1 <= len(anomalies) <= 15
    # anomalies are unusually large
    normal_max = max(t.amount for t in txns if not t.is_anomaly)
    assert all(t.amount > normal_max for t in anomalies)


def test_categories_are_known():
    txns = generate_transactions(months=2, seed=1)
    known = {"Groceries", "Rent", "Salary", "Dining", "Subscriptions", "Transport", "Utilities"}
    assert {t.category for t in txns}.issubset(known)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.data'`.

- [ ] **Step 3: Write minimal implementation**

`backend/app/data/__init__.py`: (empty file)

`backend/app/data/generator.py`:
```python
import random
from datetime import date, timedelta

from pydantic import BaseModel

CATEGORY_PROFILES = {
    "Groceries": (40, 120, ["Whole Foods", "Trader Joe's", "Safeway"]),
    "Dining": (12, 60, ["Chipotle", "Olive Garden", "Local Cafe"]),
    "Transport": (5, 40, ["Uber", "Shell", "Metro Transit"]),
    "Subscriptions": (5, 20, ["Netflix", "Spotify", "iCloud"]),
    "Utilities": (60, 200, ["City Power", "Water Dept", "Comcast"]),
}
MONTHLY_RENT = ("Rent", 1800.0, "Greenfield Apartments")
MONTHLY_SALARY = ("Salary", 5200.0, "ACME Corp Payroll")


class GeneratedTxn(BaseModel):
    date: date
    merchant: str
    amount: float
    category: str
    description: str
    is_anomaly: bool = False


def generate_transactions(months: int = 6, seed: int = 42) -> list[GeneratedTxn]:
    rng = random.Random(seed)
    start = date(2026, 1, 1)
    txns: list[GeneratedTxn] = []

    for m in range(months):
        month_start = _add_months(start, m)
        # Fixed monthly income + rent
        cat, amt, merch = MONTHLY_SALARY
        txns.append(GeneratedTxn(date=month_start, merchant=merch, amount=amt,
                                 category=cat, description="monthly salary"))
        cat, amt, merch = MONTHLY_RENT
        txns.append(GeneratedTxn(date=month_start + timedelta(days=1), merchant=merch,
                                 amount=amt, category=cat, description="monthly rent"))
        # Variable spend
        for _ in range(rng.randint(20, 35)):
            category = rng.choice(list(CATEGORY_PROFILES))
            low, high, merchants = CATEGORY_PROFILES[category]
            amount = round(rng.uniform(low, high), 2)
            day = month_start + timedelta(days=rng.randint(0, 27))
            merchant = rng.choice(merchants)
            txns.append(GeneratedTxn(date=day, merchant=merchant, amount=amount,
                                     category=category, description=f"{category.lower()} at {merchant}"))

    _plant_anomalies(txns, rng)
    txns.sort(key=lambda t: t.date)
    return txns


def _plant_anomalies(txns: list[GeneratedTxn], rng: random.Random) -> None:
    normal_max = max(t.amount for t in txns if t.category not in ("Rent", "Salary"))
    count = rng.randint(2, 4)
    victims = rng.sample([t for t in txns if t.category in CATEGORY_PROFILES], count)
    for t in victims:
        t.amount = round(normal_max * rng.uniform(3.0, 6.0), 2)
        t.is_anomaly = True
        t.description = f"UNUSUAL: large {t.category.lower()} charge at {t.merchant}"


def _add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year = d.year + month // 12
    return date(year, month % 12 + 1, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generator.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/data backend/tests/test_generator.py
git commit -m "feat: add synthetic transaction generator with planted anomalies"
```

---

### Task 6: Embedding provider (NVIDIA NIM)

**Files:**
- Create: `backend/app/providers/__init__.py`
- Create: `backend/app/providers/base.py`
- Create: `backend/app/providers/nvidia.py`
- Create: `backend/tests/test_embeddings.py`

**Interfaces:**
- Consumes: `get_settings().nvidia_api_key`, `.nvidia_embed_model`.
- Produces:
  - `EmbeddingProvider` protocol: `embed(texts: list[str]) -> list[list[float]]`.
  - `NvidiaEmbeddingProvider` implementing it, POSTing to `https://integrate.api.nvidia.com/v1/embeddings`, returning 1024-dim vectors.

- [ ] **Step 1: Write the failing test (mocked HTTP — no real network)**

`backend/tests/test_embeddings.py`:
```python
import httpx
import respx

from app.providers.nvidia import NvidiaEmbeddingProvider

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/embeddings"


@respx.mock
def test_nvidia_embed_returns_vectors():
    respx.post(NVIDIA_URL).mock(
        return_value=httpx.Response(
            200,
            json={"data": [
                {"embedding": [0.1] * 1024, "index": 0},
                {"embedding": [0.2] * 1024, "index": 1},
            ]},
        )
    )
    provider = NvidiaEmbeddingProvider(api_key="test", model="nvidia/nv-embedqa-e5-v5")

    vectors = provider.embed(["hello", "world"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 1024
    assert vectors[1][0] == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.providers'`.

- [ ] **Step 3: Write the interface and implementation**

`backend/app/providers/__init__.py`: (empty file)

`backend/app/providers/base.py`:
```python
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class ChatProvider(Protocol):
    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        ...
```

`backend/app/providers/nvidia.py`:
```python
import httpx

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/embeddings"


class NvidiaEmbeddingProvider:
    def __init__(self, api_key: str, model: str, timeout: float = 30.0):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            NVIDIA_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"input": texts, "model": self._model, "input_type": "passage"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embeddings.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/providers backend/tests/test_embeddings.py
git commit -m "feat: add nvidia embedding provider behind pluggable interface"
```

---

### Task 7: Chat provider (Groq)

**Files:**
- Create: `backend/app/providers/groq.py`
- Create: `backend/tests/test_chat_provider.py`

**Interfaces:**
- Consumes: `get_settings().groq_api_key`, `.groq_model`.
- Produces: `GroqChatProvider` implementing `ChatProvider.complete(messages, tools=None) -> dict`. Returns the first choice's `message` dict (with `content` and optional `tool_calls`), POSTing to `https://api.groq.com/openai/v1/chat/completions`.

- [ ] **Step 1: Write the failing test (mocked HTTP)**

`backend/tests/test_chat_provider.py`:
```python
import httpx
import respx

from app.providers.groq import GroqChatProvider

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


@respx.mock
def test_groq_complete_returns_message():
    respx.post(GROQ_URL).mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Hi there"}}]},
        )
    )
    provider = GroqChatProvider(api_key="test", model="llama-3.3-70b-versatile")

    msg = provider.complete([{"role": "user", "content": "hello"}])

    assert msg["content"] == "Hi there"


@respx.mock
def test_groq_passes_tools_through():
    route = respx.post(GROQ_URL).mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": []}}]},
        )
    )
    provider = GroqChatProvider(api_key="test", model="llama-3.3-70b-versatile")

    provider.complete([{"role": "user", "content": "spend?"}], tools=[{"type": "function"}])

    sent = route.calls.last.request
    assert b'"tools"' in sent.content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.providers.groq'`.

- [ ] **Step 3: Write the implementation**

`backend/app/providers/groq.py`:
```python
import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqChatProvider:
    def __init__(self, api_key: str, model: str, timeout: float = 60.0):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload: dict = {"model": self._model, "messages": messages, "temperature": 0.2}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        resp = httpx.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chat_provider.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/providers/groq.py backend/tests/test_chat_provider.py
git commit -m "feat: add groq chat provider behind pluggable interface"
```

---

### Task 8: Ingestion pipeline + seed script

**Files:**
- Create: `backend/app/ingest/__init__.py`
- Create: `backend/app/ingest/pipeline.py`
- Create: `backend/app/seed.py`
- Create: `backend/tests/test_ingest.py`

**Interfaces:**
- Consumes: `generate_transactions` (Task 5), `EmbeddingProvider` (Task 6), `SessionLocal`/`Transaction` (Task 4).
- Produces:
  - `txn_to_text(txn) -> str` — builds the string embedded for a transaction.
  - `ingest(txns: list[GeneratedTxn], embedder: EmbeddingProvider, session) -> int` — embeds and stores, returns count inserted.
  - `backend/app/seed.py` runnable as `python -m app.seed` to populate the real DB.

- [ ] **Step 1: Write the failing test (fake embedder — no network)**

`backend/tests/test_ingest.py`:
```python
from sqlalchemy import select

from app.data.generator import generate_transactions
from app.ingest.pipeline import ingest, txn_to_text
from app.models import Transaction


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7)] * 1024 for t in texts]


def test_txn_to_text_includes_key_fields():
    txns = generate_transactions(months=1, seed=3)
    text = txn_to_text(txns[0])
    assert txns[0].merchant in text
    assert txns[0].category in text


def test_ingest_stores_transactions_with_embeddings(session):
    txns = generate_transactions(months=2, seed=5)
    count = ingest(txns, FakeEmbedder(), session)

    rows = session.scalars(select(Transaction)).all()
    assert count == len(txns)
    assert len(rows) == len(txns)
    assert all(r.embedding is not None and len(r.embedding) == 1024 for r in rows)
    assert any(r.is_anomaly for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingest'`.

- [ ] **Step 3: Write the pipeline and seed script**

`backend/app/ingest/__init__.py`: (empty file)

`backend/app/ingest/pipeline.py`:
```python
from app.data.generator import GeneratedTxn
from app.models import Transaction
from app.providers.base import EmbeddingProvider


def txn_to_text(txn: GeneratedTxn) -> str:
    return (
        f"{txn.date.isoformat()} | {txn.category} | {txn.merchant} | "
        f"${txn.amount:.2f} | {txn.description}"
    )


def ingest(txns: list[GeneratedTxn], embedder: EmbeddingProvider, session) -> int:
    vectors = embedder.embed([txn_to_text(t) for t in txns])
    for txn, vector in zip(txns, vectors, strict=True):
        session.add(Transaction(
            date=txn.date,
            merchant=txn.merchant,
            amount=txn.amount,
            category=txn.category,
            description=txn.description,
            is_anomaly=txn.is_anomaly,
            embedding=vector,
        ))
    session.commit()
    return len(txns)
```

`backend/app/seed.py`:
```python
"""Populate the database with synthetic, embedded transactions.

Usage: python -m app.seed
"""
from app.config import get_settings
from app.data.generator import generate_transactions
from app.db import SessionLocal
from app.ingest.pipeline import ingest
from app.providers.nvidia import NvidiaEmbeddingProvider


def main() -> None:
    settings = get_settings()
    embedder = NvidiaEmbeddingProvider(
        api_key=settings.nvidia_api_key, model=settings.nvidia_embed_model
    )
    txns = generate_transactions(months=6, seed=42)
    with SessionLocal() as session:
        count = ingest(txns, embedder, session)
    print(f"Seeded {count} transactions")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full backend test suite**

Run (from `backend/`): `pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/ingest backend/app/seed.py backend/tests/test_ingest.py
git commit -m "feat: add ingestion pipeline and seed script"
```

- [ ] **Step 7: (Manual, optional) Seed the real DB end-to-end**

Only after rotating and setting real keys in `.env`:
```bash
alembic upgrade head
python -m app.seed
```
Expected: `Seeded <N> transactions`. This confirms Groq/NVIDIA wiring works with live keys before Plan 2.

---

## Self-Review

**Spec coverage (this plan's slice — Backend Foundation):**
- Postgres + pgvector store → Tasks 3, 4 ✓
- Data model (transactions/insights/chat_messages, `vector(1024)`, `is_anomaly` ground truth) → Task 4 ✓
- Synthetic generator with planted anomalies → Task 5 ✓
- Pluggable provider layer (Groq chat + NVIDIA embeddings, swappable) → Tasks 6, 7 ✓
- Ingestion (embed + store) → Task 8 ✓
- Secret hygiene (`.env` gitignored, `.env.example`, no committed keys) → Task 1 ✓
- Docker Compose one-command DB → Task 3 ✓
- Deferred to later plans (correctly out of scope here): RAG agent/tools + eval (Plan 2), insights + anomaly detection logic (Plan 3), React frontend (Plan 4).

**Placeholder scan:** No TBD/TODO; every code step contains complete, runnable code. ✓

**Type consistency:** `EmbeddingProvider.embed(texts)->list[list[float]]` used consistently in Tasks 6 and 8; `Vector(1024)`/`EMBED_DIM=1024` consistent across models, migration, and tests; `GeneratedTxn` fields consistent between generator (Task 5) and ingestion (Task 8). ✓
