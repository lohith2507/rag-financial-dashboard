# Insights + Anomalies Implementation Plan (Plan 3 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Statistical anomaly detection with LLM explanations, AI-generated monthly insights, and the dashboard data endpoints — completing the backend surface Plan 4's frontend will consume.

**Architecture:** A per-category z-score detector flags outliers (verifiable against the generator's planted `is_anomaly` ground truth). Groq explains flagged transactions on request. An insights service aggregates a month's stats via SQL, prompts Groq for a plain-English summary grounded in those numbers, and persists it. Thin FastAPI routers expose anomalies, insights, stats, and transactions.

**Tech Stack:** Python 3.12, SQLAlchemy 2.0, FastAPI + TestClient, Groq (faked in tests — no network).

**Prerequisite:** Plans 1–2 complete (models, providers, `app.main:app`, `app/api/deps.py`, `app/rag/tools.py`, session fixture).

## Global Constraints

- Python 3.12+, SQLAlchemy 2.0 style, Pydantic v2. Tests never hit the network — chat provider is faked via `app.dependency_overrides`.
- Anomaly detection is statistical (per-category z-score, default threshold **3.0**), computed only over spend categories (excludes `Salary`); categories with fewer than 3 transactions or zero variance are skipped.
- Detection must be verifiable: on the seeded generator dataset (seed 42, 6 months) every planted `is_anomaly=True` transaction must be detected.
- LLM explanations/insights are grounded: prompts embed the computed stats; the model never invents numbers that aren't in the prompt.
- Periods are `YYYY-MM` strings. Every task ends with passing tests and a commit.

---

### Task 1: Statistical anomaly detector

**Files:**
- Create: `backend/app/anomalies/__init__.py`
- Create: `backend/app/anomalies/detector.py`
- Test: `backend/tests/test_detector.py`

**Interfaces:**
- Consumes: `Transaction`, `txn_to_dict` (from `app/rag/tools.py`).
- Produces: `detect_anomalies(session, z_threshold: float = 3.0) -> list[dict]` — each dict is `txn_to_dict(...)` plus `"z_score": float` and `"category_mean": float`, sorted by `z_score` descending.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_detector.py`:
```python
from datetime import date

from app.anomalies.detector import detect_anomalies
from app.data.generator import generate_transactions
from app.models import Transaction


def _txn(amount, category="Groceries", d=date(2026, 6, 1), anomaly=False):
    return Transaction(date=d, merchant="M", amount=amount, category=category,
                       description="d", is_anomaly=anomaly)


def test_detects_planted_outlier(session):
    session.add_all([_txn(a) for a in [40, 50, 60, 45, 55, 50, 48, 52, 47, 53]])
    session.add(_txn(5000.0, anomaly=True))
    session.commit()

    flagged = detect_anomalies(session)

    assert len(flagged) == 1
    assert flagged[0]["amount"] == 5000.0
    assert flagged[0]["z_score"] > 3.0
    assert flagged[0]["category_mean"] > 0


def test_skips_tiny_and_constant_categories(session):
    session.add_all([_txn(1800.0, category="Rent", d=date(2026, m, 1)) for m in range(1, 7)])
    session.add_all([_txn(9.99, category="Subscriptions"), _txn(500.0, category="Subscriptions")])
    session.commit()

    assert detect_anomalies(session) == []  # Rent: zero variance; Subscriptions: n < 3


def test_recall_on_seeded_generator_data(session):
    txns = generate_transactions(months=6, seed=42)
    session.add_all([
        Transaction(date=t.date, merchant=t.merchant, amount=t.amount, category=t.category,
                    description=t.description, is_anomaly=t.is_anomaly)
        for t in txns
    ])
    session.commit()

    flagged_ids = {(f["date"], f["merchant"], f["amount"]) for f in detect_anomalies(session)}
    planted = [(t.date.isoformat(), t.merchant, t.amount) for t in txns if t.is_anomaly]

    assert planted, "generator must plant anomalies"
    for p in planted:
        assert p in flagged_ids, f"planted anomaly not detected: {p}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.anomalies'`.

- [ ] **Step 3: Implement**

`backend/app/anomalies/__init__.py`: (empty file)

`backend/app/anomalies/detector.py`:
```python
import statistics

from sqlalchemy import select

from app.models import Transaction
from app.rag.tools import txn_to_dict

Z_THRESHOLD = 3.0
SPEND_EXCLUDED = ("Salary",)
MIN_CATEGORY_SIZE = 3


def detect_anomalies(session, z_threshold: float = Z_THRESHOLD) -> list[dict]:
    txns = session.scalars(
        select(Transaction).where(Transaction.category.notin_(SPEND_EXCLUDED))
    ).all()

    by_category: dict[str, list[Transaction]] = {}
    for t in txns:
        by_category.setdefault(t.category, []).append(t)

    flagged: list[dict] = []
    for rows in by_category.values():
        if len(rows) < MIN_CATEGORY_SIZE:
            continue
        amounts = [r.amount for r in rows]
        mean = statistics.fmean(amounts)
        stdev = statistics.pstdev(amounts)
        if stdev == 0:
            continue
        for r in rows:
            z = (r.amount - mean) / stdev
            if z > z_threshold:
                flagged.append({**txn_to_dict(r),
                                "z_score": round(z, 2),
                                "category_mean": round(mean, 2)})

    flagged.sort(key=lambda d: d["z_score"], reverse=True)
    return flagged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_detector.py -v`
Expected: 3 passed. If `test_recall_on_seeded_generator_data` fails, the z-threshold vs planted-anomaly sizing is off — report it, do not silently lower the threshold.

- [ ] **Step 5: Commit**

```bash
git add app/anomalies tests/test_detector.py
git commit -m "feat: add z-score anomaly detector verified against planted ground truth"
```

---

### Task 2: Anomaly explanations + /anomalies endpoint

**Files:**
- Create: `backend/app/anomalies/explainer.py`
- Create: `backend/app/api/anomalies.py`
- Modify: `backend/app/main.py` (include router)
- Test: `backend/tests/test_api_anomalies.py`

**Interfaces:**
- Consumes: `detect_anomalies`, `ChatProvider`, `get_db`/`get_chat_provider` deps.
- Produces:
  - `explain_anomaly(anomaly: dict, chat: ChatProvider) -> str` — 1–2 sentence grounded explanation.
  - `GET /anomalies?explain=false` → `{"anomalies": [ ...detector dicts... ]}`; with `explain=true`, each of the top `MAX_EXPLAINED = 10` gains `"explanation": str`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_api_anomalies.py`:
```python
from datetime import date

from fastapi.testclient import TestClient

from app.api.deps import get_chat_provider
from app.main import app
from app.models import Transaction


class CannedChat:
    def complete(self, messages, tools=None):
        return {"role": "assistant", "content": "This charge is 50x the category average."}


def _seed(session):
    session.add_all([
        Transaction(date=date(2026, 6, d), merchant="M", amount=a, category="Groceries",
                    description="d", is_anomaly=False)
        for d, a in [(1, 40), (2, 50), (3, 60), (4, 45), (5, 55), (6, 50), (7, 48),
                     (8, 52), (9, 47), (10, 53)]
    ])
    session.add(Transaction(date=date(2026, 6, 15), merchant="Sketchy Store", amount=5000.0,
                            category="Groceries", description="d", is_anomaly=True))
    session.commit()


def test_anomalies_endpoint_without_explanations(session):
    _seed(session)
    client = TestClient(app)
    resp = client.get("/anomalies")

    assert resp.status_code == 200
    anomalies = resp.json()["anomalies"]
    assert len(anomalies) == 1
    assert anomalies[0]["merchant"] == "Sketchy Store"
    assert "explanation" not in anomalies[0]


def test_anomalies_endpoint_with_explanations(session):
    _seed(session)
    app.dependency_overrides[get_chat_provider] = lambda: CannedChat()
    client = TestClient(app)
    try:
        resp = client.get("/anomalies?explain=true")
    finally:
        app.dependency_overrides.clear()

    anomalies = resp.json()["anomalies"]
    assert anomalies[0]["explanation"] == "This charge is 50x the category average."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_anomalies.py -v`
Expected: FAIL — 404 (no `/anomalies` route yet).

- [ ] **Step 3: Implement**

`backend/app/anomalies/explainer.py`:
```python
import json

from app.providers.base import ChatProvider

EXPLAIN_SYSTEM = (
    "You are a personal-finance anomaly analyst. Using ONLY the numbers provided, "
    "explain in 1-2 plain-English sentences why this transaction looks unusual."
)


def explain_anomaly(anomaly: dict, chat: ChatProvider) -> str:
    user = (
        f"Transaction: {json.dumps(anomaly)}\n"
        f"The category's average transaction is ${anomaly['category_mean']:.2f}; "
        f"this one is {anomaly['z_score']} standard deviations above it."
    )
    msg = chat.complete([
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": user},
    ])
    return msg.get("content") or ""
```

`backend/app/api/anomalies.py`:
```python
from fastapi import APIRouter, Depends

from app.anomalies.detector import detect_anomalies
from app.anomalies.explainer import explain_anomaly
from app.api.deps import get_chat_provider, get_db

router = APIRouter()

MAX_EXPLAINED = 10


@router.get("/anomalies")
def anomalies(explain: bool = False, db=Depends(get_db),
              chat=Depends(get_chat_provider)):
    found = detect_anomalies(db)
    if explain:
        for a in found[:MAX_EXPLAINED]:
            a["explanation"] = explain_anomaly(a, chat)
    return {"anomalies": found}
```

In `backend/app/main.py`, add:
```python
from app.api.anomalies import router as anomalies_router
app.include_router(anomalies_router)
```

- [ ] **Step 4: Run tests, then full suite**

Run: `pytest tests/test_api_anomalies.py -v && pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/anomalies/explainer.py app/api/anomalies.py app/main.py tests/test_api_anomalies.py
git commit -m "feat: add /anomalies endpoint with grounded LLM explanations"
```

---

### Task 3: Monthly insights service + endpoints

**Files:**
- Create: `backend/app/insights/__init__.py`
- Create: `backend/app/insights/service.py`
- Create: `backend/app/api/insights.py`
- Modify: `backend/app/main.py` (include router)
- Test: `backend/tests/test_insights.py`

**Interfaces:**
- Consumes: `Transaction`, `Insight`, `ChatProvider`, deps.
- Produces:
  - `month_stats(session, year: int, month: int) -> dict` — `{"period": "YYYY-MM", "total_spend": float, "income": float, "by_category": {cat: total}, "anomaly_count": int}` (spend excludes `Salary`; income is the `Salary` sum).
  - `generate_monthly_insight(session, chat, period: str) -> Insight` — builds stats for `period` and the previous month, prompts Groq, persists and returns the `Insight` row.
  - `POST /insights/{period}/generate` → the new insight `{id, period, summary_text, generated_at}`.
  - `GET /insights` → `{"insights": [...]}` most recent first.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_insights.py`:
```python
from datetime import date

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.deps import get_chat_provider
from app.insights.service import month_stats
from app.main import app
from app.models import Insight, Transaction


class CannedChat:
    def __init__(self):
        self.last_user_content = None

    def complete(self, messages, tools=None):
        self.last_user_content = messages[-1]["content"]
        return {"role": "assistant", "content": "June spending was steady."}


def _seed(session):
    session.add_all([
        Transaction(date=date(2026, 6, 1), merchant="ACME", amount=5200.0,
                    category="Salary", description="d", is_anomaly=False),
        Transaction(date=date(2026, 6, 3), merchant="WF", amount=80.0,
                    category="Groceries", description="d", is_anomaly=False),
        Transaction(date=date(2026, 6, 9), merchant="NF", amount=15.0,
                    category="Subscriptions", description="d", is_anomaly=True),
        Transaction(date=date(2026, 5, 20), merchant="WF", amount=60.0,
                    category="Groceries", description="d", is_anomaly=False),
    ])
    session.commit()


def test_month_stats_aggregates_correctly(session):
    _seed(session)
    stats = month_stats(session, 2026, 6)

    assert stats["period"] == "2026-06"
    assert stats["total_spend"] == 95.0
    assert stats["income"] == 5200.0
    assert stats["by_category"] == {"Groceries": 80.0, "Subscriptions": 15.0}
    assert stats["anomaly_count"] == 1


def test_generate_and_list_insights(session):
    _seed(session)
    canned = CannedChat()
    app.dependency_overrides[get_chat_provider] = lambda: canned
    client = TestClient(app)
    try:
        resp = client.post("/insights/2026-06/generate")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["summary_text"] == "June spending was steady."
    # Prompt was grounded in the real stats
    assert "95.0" in canned.last_user_content
    assert "2026-05" in canned.last_user_content  # previous month included

    listed = TestClient(app).get("/insights").json()["insights"]
    assert listed[0]["period"] == "2026-06"
    row = session.scalars(select(Insight)).one()
    assert row.summary_text == "June spending was steady."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_insights.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.insights'`.

- [ ] **Step 3: Implement**

`backend/app/insights/__init__.py`: (empty file)

`backend/app/insights/service.py`:
```python
import json
from datetime import date

from sqlalchemy import extract, func, select

from app.models import Insight, Transaction
from app.providers.base import ChatProvider

SPEND_EXCLUDED = ("Salary",)

INSIGHT_SYSTEM = (
    "You are a personal-finance analyst. Write a short plain-English monthly summary "
    "(3-5 sentences) using ONLY the JSON stats provided: top categories, notable "
    "changes vs the previous month, and any anomalies. Cite actual numbers."
)


def month_stats(session, year: int, month: int) -> dict:
    def _in_month(stmt):
        return stmt.where(extract("year", Transaction.date) == year,
                          extract("month", Transaction.date) == month)

    spend_rows = session.execute(_in_month(
        select(Transaction.category, func.sum(Transaction.amount))
        .where(Transaction.category.notin_(SPEND_EXCLUDED))
        .group_by(Transaction.category)
    )).all()
    income = session.execute(_in_month(
        select(func.coalesce(func.sum(Transaction.amount), 0.0))
        .where(Transaction.category == "Salary")
    )).scalar_one()
    anomaly_count = session.execute(_in_month(
        select(func.count(Transaction.id)).where(Transaction.is_anomaly.is_(True))
    )).scalar_one()

    by_category = {cat: round(float(total), 2) for cat, total in spend_rows}
    return {
        "period": f"{year:04d}-{month:02d}",
        "total_spend": round(sum(by_category.values()), 2),
        "income": round(float(income), 2),
        "by_category": by_category,
        "anomaly_count": anomaly_count,
    }


def _previous(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def generate_monthly_insight(session, chat: ChatProvider, period: str) -> Insight:
    year, month = (int(p) for p in period.split("-"))
    current = month_stats(session, year, month)
    prev = month_stats(session, *_previous(year, month))

    user = (f"Current month stats: {json.dumps(current)}\n"
            f"Previous month stats: {json.dumps(prev)}")
    msg = chat.complete([
        {"role": "system", "content": INSIGHT_SYSTEM},
        {"role": "user", "content": user},
    ])

    insight = Insight(period=period, summary_text=msg.get("content") or "")
    session.add(insight)
    session.commit()
    return insight
```

`backend/app/api/insights.py`:
```python
from fastapi import APIRouter, Depends
from sqlalchemy import select

from app.api.deps import get_chat_provider, get_db
from app.insights.service import generate_monthly_insight
from app.models import Insight

router = APIRouter()


def _to_dict(i: Insight) -> dict:
    return {"id": i.id, "period": i.period, "summary_text": i.summary_text,
            "generated_at": i.generated_at.isoformat() if i.generated_at else None}


@router.post("/insights/{period}/generate")
def generate(period: str, db=Depends(get_db), chat=Depends(get_chat_provider)):
    return _to_dict(generate_monthly_insight(db, chat, period))


@router.get("/insights")
def list_insights(db=Depends(get_db)):
    rows = db.scalars(select(Insight).order_by(Insight.generated_at.desc())).all()
    return {"insights": [_to_dict(i) for i in rows]}
```

In `backend/app/main.py`, add:
```python
from app.api.insights import router as insights_router
app.include_router(insights_router)
```

- [ ] **Step 4: Run tests, then full suite**

Run: `pytest tests/test_insights.py -v && pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/insights app/api/insights.py app/main.py tests/test_insights.py
git commit -m "feat: add grounded monthly insights service and endpoints"
```

---

### Task 4: Dashboard data endpoints (stats + transactions)

**Files:**
- Create: `backend/app/api/stats.py`
- Modify: `backend/app/main.py` (include router)
- Test: `backend/tests/test_api_stats.py`

**Interfaces:**
- Consumes: `Transaction`, `list_transactions` (from `app/rag/tools.py`), `month_stats` helpers.
- Produces (all consumed by Plan 4's charts):
  - `GET /stats/by-category?start=&end=` → `{"by_category": [{"category": str, "total": float}]}` sorted by total desc, spend only.
  - `GET /stats/monthly` → `{"monthly": [{"period": "YYYY-MM", "spend": float, "income": float}]}` ascending by period.
  - `GET /transactions?category=&merchant=&start=&end=&limit=50` → `{"transactions": [...]}`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_api_stats.py`:
```python
from datetime import date

from fastapi.testclient import TestClient

from app.main import app
from app.models import Transaction


def _seed(session):
    session.add_all([
        Transaction(date=date(2026, 5, 1), merchant="ACME", amount=5200.0,
                    category="Salary", description="d", is_anomaly=False),
        Transaction(date=date(2026, 5, 10), merchant="WF", amount=100.0,
                    category="Groceries", description="d", is_anomaly=False),
        Transaction(date=date(2026, 6, 1), merchant="ACME", amount=5200.0,
                    category="Salary", description="d", is_anomaly=False),
        Transaction(date=date(2026, 6, 3), merchant="WF", amount=80.0,
                    category="Groceries", description="d", is_anomaly=False),
        Transaction(date=date(2026, 6, 9), merchant="NF", amount=15.0,
                    category="Subscriptions", description="d", is_anomaly=False),
    ])
    session.commit()


def test_by_category(session):
    _seed(session)
    resp = TestClient(app).get("/stats/by-category?start=2026-06-01&end=2026-06-30")
    assert resp.status_code == 200
    assert resp.json()["by_category"] == [
        {"category": "Groceries", "total": 80.0},
        {"category": "Subscriptions", "total": 15.0},
    ]


def test_monthly(session):
    _seed(session)
    monthly = TestClient(app).get("/stats/monthly").json()["monthly"]
    assert monthly == [
        {"period": "2026-05", "spend": 100.0, "income": 5200.0},
        {"period": "2026-06", "spend": 95.0, "income": 5200.0},
    ]


def test_transactions_listing(session):
    _seed(session)
    rows = TestClient(app).get("/transactions?category=Groceries").json()["transactions"]
    assert len(rows) == 2
    assert all(r["category"] == "Groceries" for r in rows)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_stats.py -v`
Expected: FAIL — 404 (routes don't exist).

- [ ] **Step 3: Implement**

`backend/app/api/stats.py`:
```python
from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy import extract, func, select

from app.api.deps import get_db
from app.models import Transaction
from app.rag.tools import list_transactions

router = APIRouter()

SPEND_EXCLUDED = ("Salary",)


@router.get("/stats/by-category")
def by_category(start: date | None = None, end: date | None = None, db=Depends(get_db)):
    stmt = (select(Transaction.category, func.sum(Transaction.amount).label("total"))
            .where(Transaction.category.notin_(SPEND_EXCLUDED))
            .group_by(Transaction.category)
            .order_by(func.sum(Transaction.amount).desc()))
    if start:
        stmt = stmt.where(Transaction.date >= start)
    if end:
        stmt = stmt.where(Transaction.date <= end)
    rows = db.execute(stmt).all()
    return {"by_category": [{"category": c, "total": round(float(t), 2)} for c, t in rows]}


@router.get("/stats/monthly")
def monthly(db=Depends(get_db)):
    year = extract("year", Transaction.date)
    month = extract("month", Transaction.date)
    spend = func.sum(
        func.case((Transaction.category.notin_(SPEND_EXCLUDED), Transaction.amount), else_=0.0)
    )
    income = func.sum(
        func.case((Transaction.category == "Salary", Transaction.amount), else_=0.0)
    )
    rows = db.execute(
        select(year, month, spend, income).group_by(year, month).order_by(year, month)
    ).all()
    return {"monthly": [
        {"period": f"{int(y):04d}-{int(m):02d}",
         "spend": round(float(s), 2), "income": round(float(i), 2)}
        for y, m, s, i in rows
    ]}


@router.get("/transactions")
def transactions(category: str | None = None, merchant: str | None = None,
                 start: date | None = None, end: date | None = None,
                 limit: int = 50, db=Depends(get_db)):
    return {"transactions": list_transactions(
        db, category=category, merchant=merchant, start=start, end=end, limit=limit)}
```

Note: if `func.case` raises on this SQLAlchemy version, use `sqlalchemy.case((cond, value), else_=0.0)` imported directly — same semantics.

In `backend/app/main.py`, add:
```python
from app.api.stats import router as stats_router
app.include_router(stats_router)
```

- [ ] **Step 4: Run tests, then full suite**

Run: `pytest tests/test_api_stats.py -v && pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/api/stats.py app/main.py tests/test_api_stats.py
git commit -m "feat: add dashboard stats and transactions endpoints"
```

---

## Self-Review

**Spec coverage (Plan 3 slice):** statistical anomaly detection verified against planted ground truth ✓ (Task 1, recall test); LLM explanations of anomalies ✓ (Task 2); grounded plain-English monthly insights, persisted ✓ (Task 3); dashboard data endpoints for Plan 4 charts ✓ (Task 4). Deferred: frontend (Plan 4).

**Placeholder scan:** none — complete code and exact commands in every step. ✓

**Type consistency:** detector dicts extend `txn_to_dict` with `z_score`/`category_mean`, and `explain_anomaly` consumes exactly those keys (Tasks 1→2); `month_stats` dict keys used verbatim in the insight prompt test (Task 3); endpoint shapes in Task 4 match the Interfaces block. ✓

**Bounded/grounded:** explanations capped at `MAX_EXPLAINED = 10` per request; prompts embed computed stats only. ✓
