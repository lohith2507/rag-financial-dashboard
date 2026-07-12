# RAG Agent + Eval Implementation Plan (Plan 2 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the hybrid agentic RAG core — SQL + semantic-search tools, the Groq function-calling agent, a `/chat` API endpoint, and the A-vs-B eval harness that produces the measured accuracy number.

**Architecture:** Four tools (`aggregate_spend`, `list_transactions`, `get_anomalies`, `semantic_search`) operate on the Plan-1 database. An agent loop hands the tools to Groq via function calling, executes requested calls (bounded rounds), and returns a grounded answer plus a tool-usage trace. A pure-vector baseline (approach A) and an eval harness measure hybrid (approach B) vs baseline accuracy on questions with ground truth derived from the deterministic generator.

**Tech Stack:** Python 3.12, FastAPI + TestClient, SQLAlchemy 2.0, Groq function calling, pytest (fakes for chat/embeddings — no network in tests).

**Prerequisite:** Plan 1 (Backend Foundation) fully complete: migrated DB, seeded or seedable transactions, providers, `backend/tests/conftest.py` session fixture.

## Global Constraints

- Python 3.12+, SQLAlchemy 2.0 style, Pydantic v2.
- Embedding dim **1024**; embeddings stored as `ARRAY(Float)`; similarity computed in Python (no pgvector).
- Tests NEVER make real network calls — chat/embedding providers are faked in tests. Live-API runs happen only in the manual eval step.
- The agent tool loop is **bounded at 5 rounds** (`MAX_TOOL_ROUNDS = 5`) — never an unbounded loop.
- "Spend" always excludes category `Salary` (income is not spend).
- Tool args crossing the LLM boundary are JSON: dates are ISO `YYYY-MM-DD` strings, parsed with `date.fromisoformat`.
- Secrets only via `.env` (gitignored). Every task ends with passing tests and a commit.

---

### Task 1: Cosine similarity + query-mode embeddings

**Files:**
- Create: `backend/app/rag/__init__.py`
- Create: `backend/app/rag/similarity.py`
- Modify: `backend/app/providers/nvidia.py` (add `input_type` constructor arg)
- Test: `backend/tests/test_similarity.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `cosine(a: list[float], b: list[float]) -> float`; `NvidiaEmbeddingProvider(api_key, model, timeout=30.0, input_type="passage")` — payload sends `self._input_type` (queries use a provider instance constructed with `input_type="query"`).

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_similarity.py`:
```python
import httpx
import pytest
import respx

from app.providers.nvidia import NVIDIA_URL, NvidiaEmbeddingProvider
from app.rag.similarity import cosine


def test_cosine_identical_vectors():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_returns_zero():
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


@respx.mock
def test_nvidia_provider_sends_query_input_type():
    route = respx.post(NVIDIA_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1] * 1024, "index": 0}]})
    )
    provider = NvidiaEmbeddingProvider(api_key="k", model="m", input_type="query")
    provider.embed(["how much on food"])
    assert b'"input_type": "query"' in route.calls.last.request.content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_similarity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag'` (and the nvidia test fails on the unknown `input_type` kwarg).

- [ ] **Step 3: Implement**

`backend/app/rag/__init__.py`: (empty file)

`backend/app/rag/similarity.py`:
```python
import math


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
```

In `backend/app/providers/nvidia.py`, change the constructor and payload:
```python
class NvidiaEmbeddingProvider:
    def __init__(self, api_key: str, model: str, timeout: float = 30.0, input_type: str = "passage"):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._input_type = input_type

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            NVIDIA_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"input": texts, "model": self._model, "input_type": self._input_type},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]
```

- [ ] **Step 4: Run the full suite (existing embedding tests must still pass)**

Run: `pytest tests/test_similarity.py tests/test_embeddings.py -v`
Expected: all pass (default `input_type="passage"` preserves old behavior).

- [ ] **Step 5: Commit**

```bash
git add app/rag tests/test_similarity.py app/providers/nvidia.py
git commit -m "feat: add cosine similarity and query-mode embeddings"
```

---

### Task 2: RAG tools (SQL + semantic search)

**Files:**
- Create: `backend/app/rag/tools.py`
- Test: `backend/tests/test_tools.py`

**Interfaces:**
- Consumes: `Transaction`, `cosine`, `EmbeddingProvider`, the `session` fixture.
- Produces (all return JSON-serializable dicts/lists; date params are `date` objects at this layer — string parsing happens in the agent):
  - `aggregate_spend(session, category: str | None = None, start: date | None = None, end: date | None = None) -> dict` — `{"total": float, "count": int, "category": ..., "start": ..., "end": ...}`; excludes `Salary`.
  - `list_transactions(session, category=None, merchant=None, start=None, end=None, limit=20) -> list[dict]`
  - `get_anomalies(session, start=None, end=None) -> list[dict]`
  - `semantic_search(session, embedder, query: str, k: int = 8) -> list[dict]` — each dict includes `"score"`.
  - `txn_to_dict(txn) -> dict` — `{id, date(iso str), merchant, amount, category, description, is_anomaly}`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_tools.py`:
```python
from datetime import date

import pytest

from app.models import Transaction
from app.rag.tools import aggregate_spend, get_anomalies, list_transactions, semantic_search


def _txn(d, merchant, amount, category, anomaly=False, embedding=None):
    return Transaction(date=d, merchant=merchant, amount=amount, category=category,
                       description=f"{category} at {merchant}", is_anomaly=anomaly,
                       embedding=embedding)


@pytest.fixture
def seeded(session):
    session.add_all([
        _txn(date(2026, 6, 1), "ACME Corp Payroll", 5200.0, "Salary"),
        _txn(date(2026, 6, 3), "Whole Foods", 80.0, "Groceries", embedding=[1.0] + [0.0] * 1023),
        _txn(date(2026, 6, 10), "Safeway", 20.0, "Groceries", embedding=[0.0, 1.0] + [0.0] * 1022),
        _txn(date(2026, 6, 15), "Netflix", 15.0, "Subscriptions", anomaly=False,
             embedding=[0.9, 0.1] + [0.0] * 1022),
        _txn(date(2026, 7, 2), "Uber", 900.0, "Transport", anomaly=True),
    ])
    session.commit()
    return session


def test_aggregate_spend_excludes_salary(seeded):
    result = aggregate_spend(seeded)
    assert result["total"] == pytest.approx(1015.0)  # 80+20+15+900, no 5200
    assert result["count"] == 4


def test_aggregate_spend_filters_category_and_dates(seeded):
    result = aggregate_spend(seeded, category="Groceries",
                             start=date(2026, 6, 1), end=date(2026, 6, 30))
    assert result["total"] == pytest.approx(100.0)
    assert result["count"] == 2


def test_list_transactions_filters_and_limits(seeded):
    rows = list_transactions(seeded, category="Groceries", limit=1)
    assert len(rows) == 1
    assert rows[0]["category"] == "Groceries"
    assert isinstance(rows[0]["date"], str)


def test_get_anomalies_returns_flagged_only(seeded):
    rows = get_anomalies(seeded)
    assert len(rows) == 1
    assert rows[0]["merchant"] == "Uber"


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]  # matches Whole Foods vector


def test_semantic_search_ranks_by_cosine(seeded):
    rows = semantic_search(seeded, FakeEmbedder(), "organic groceries", k=2)
    assert len(rows) == 2
    assert rows[0]["merchant"] == "Whole Foods"      # exact match, score 1.0
    assert rows[0]["score"] > rows[1]["score"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL — `ImportError` (no `app.rag.tools`).

- [ ] **Step 3: Implement**

`backend/app/rag/tools.py`:
```python
from datetime import date

from sqlalchemy import func, select

from app.models import Transaction
from app.providers.base import EmbeddingProvider
from app.rag.similarity import cosine

SPEND_EXCLUDED = ("Salary",)


def txn_to_dict(txn: Transaction) -> dict:
    return {
        "id": txn.id,
        "date": txn.date.isoformat(),
        "merchant": txn.merchant,
        "amount": txn.amount,
        "category": txn.category,
        "description": txn.description,
        "is_anomaly": txn.is_anomaly,
    }


def aggregate_spend(session, category: str | None = None,
                    start: date | None = None, end: date | None = None) -> dict:
    stmt = select(
        func.coalesce(func.sum(Transaction.amount), 0.0),
        func.count(Transaction.id),
    ).where(Transaction.category.notin_(SPEND_EXCLUDED))
    if category:
        stmt = stmt.where(Transaction.category == category)
    if start:
        stmt = stmt.where(Transaction.date >= start)
    if end:
        stmt = stmt.where(Transaction.date <= end)
    total, count = session.execute(stmt).one()
    return {
        "total": round(float(total), 2),
        "count": count,
        "category": category,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
    }


def list_transactions(session, category: str | None = None, merchant: str | None = None,
                      start: date | None = None, end: date | None = None,
                      limit: int = 20) -> list[dict]:
    stmt = select(Transaction).order_by(Transaction.date.desc()).limit(limit)
    if category:
        stmt = stmt.where(Transaction.category == category)
    if merchant:
        stmt = stmt.where(Transaction.merchant.ilike(f"%{merchant}%"))
    if start:
        stmt = stmt.where(Transaction.date >= start)
    if end:
        stmt = stmt.where(Transaction.date <= end)
    return [txn_to_dict(t) for t in session.scalars(stmt)]


def get_anomalies(session, start: date | None = None, end: date | None = None) -> list[dict]:
    stmt = select(Transaction).where(Transaction.is_anomaly.is_(True)).order_by(Transaction.date)
    if start:
        stmt = stmt.where(Transaction.date >= start)
    if end:
        stmt = stmt.where(Transaction.date <= end)
    return [txn_to_dict(t) for t in session.scalars(stmt)]


def semantic_search(session, embedder: EmbeddingProvider, query: str, k: int = 8) -> list[dict]:
    query_vec = embedder.embed([query])[0]
    stmt = select(Transaction).where(Transaction.embedding.is_not(None))
    scored = [
        (cosine(query_vec, t.embedding), t)
        for t in session.scalars(stmt)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**txn_to_dict(t), "score": round(s, 4)} for s, t in scored[:k]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/rag/tools.py tests/test_tools.py
git commit -m "feat: add RAG tools (sql aggregation + python cosine search)"
```

---

### Task 3: The hybrid agent (Groq function calling, bounded loop)

**Files:**
- Create: `backend/app/rag/agent.py`
- Test: `backend/tests/test_agent.py`

**Interfaces:**
- Consumes: `ChatProvider.complete(messages, tools) -> dict` (a message dict, possibly containing `tool_calls` in OpenAI format), the four tools from Task 2.
- Produces: `run_agent(question: str, chat: ChatProvider, session, embedder: EmbeddingProvider) -> dict` returning `{"answer": str, "tools_used": list[{"tool": str, "args": dict}]}`. Also `TOOL_SCHEMAS: list[dict]` and `MAX_TOOL_ROUNDS = 5`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_agent.py`:
```python
import json
from datetime import date

from app.models import Transaction
from app.rag.agent import MAX_TOOL_ROUNDS, run_agent


class ScriptedChat:
    """Returns canned messages in order; records what it was sent."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def complete(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self._script.pop(0)


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0] * 1024 for _ in texts]


def _tool_call(call_id, name, args):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_agent_executes_tool_then_answers(session):
    session.add(Transaction(date=date(2026, 6, 3), merchant="Whole Foods", amount=80.0,
                            category="Groceries", description="food", is_anomaly=False))
    session.commit()

    chat = ScriptedChat([
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c1", "aggregate_spend",
                                   {"category": "Groceries", "start": "2026-06-01", "end": "2026-06-30"})]},
        {"role": "assistant", "content": "You spent $80.00 on groceries in June."},
    ])

    result = run_agent("How much on groceries in June?", chat, session, FakeEmbedder())

    assert result["answer"] == "You spent $80.00 on groceries in June."
    assert result["tools_used"] == [{"tool": "aggregate_spend",
                                     "args": {"category": "Groceries",
                                              "start": "2026-06-01", "end": "2026-06-30"}}]
    # The tool result was fed back to the model as a tool message
    tool_msgs = [m for m in chat.calls[1]["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert json.loads(tool_msgs[0]["content"])["total"] == 80.0


def test_agent_answers_directly_without_tools(session):
    chat = ScriptedChat([{"role": "assistant", "content": "Hello!"}])
    result = run_agent("hi", chat, session, FakeEmbedder())
    assert result["answer"] == "Hello!"
    assert result["tools_used"] == []


def test_agent_loop_is_bounded(session):
    looping_call = {"role": "assistant", "content": None,
                    "tool_calls": [_tool_call("cx", "get_anomalies", {})]}
    chat = ScriptedChat([looping_call] * MAX_TOOL_ROUNDS)

    result = run_agent("weird stuff?", chat, session, FakeEmbedder())

    assert len(chat.calls) == MAX_TOOL_ROUNDS  # never exceeds the bound
    assert result["answer"]  # graceful fallback answer, not an exception
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL — `ImportError` (no `app.rag.agent`).

- [ ] **Step 3: Implement**

`backend/app/rag/agent.py`:
```python
import json
from datetime import date

from app.providers.base import ChatProvider, EmbeddingProvider
from app.rag import tools as rag_tools

MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = (
    "You are a personal-finance assistant. Ground EVERY answer in the user's real "
    "transaction data by calling tools. For any question about totals, sums, or "
    "'how much', ALWAYS call aggregate_spend — never compute totals from search "
    "results. Use semantic_search for fuzzy or descriptive questions. Dates are "
    "ISO YYYY-MM-DD. Answer in plain English and cite concrete numbers."
)

_DATE_PARAM = {"type": "string", "description": "ISO date YYYY-MM-DD"}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "aggregate_spend",
        "description": "Sum spending (excludes salary/income) with optional category and date range. The ONLY reliable way to answer 'how much did I spend'.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string"}, "start": _DATE_PARAM, "end": _DATE_PARAM},
            "required": []}}},
    {"type": "function", "function": {
        "name": "list_transactions",
        "description": "List individual transactions with optional filters.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string"}, "merchant": {"type": "string"},
            "start": _DATE_PARAM, "end": _DATE_PARAM,
            "limit": {"type": "integer", "default": 20}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_anomalies",
        "description": "Return transactions flagged as anomalous/unusual.",
        "parameters": {"type": "object", "properties": {
            "start": _DATE_PARAM, "end": _DATE_PARAM}, "required": []}}},
    {"type": "function", "function": {
        "name": "semantic_search",
        "description": "Fuzzy semantic search over transaction descriptions (e.g. 'subscriptions I forgot about').",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "k": {"type": "integer", "default": 8}},
            "required": ["query"]}}},
]

FALLBACK_ANSWER = "I couldn't finish answering that within my tool budget — try a more specific question."


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _execute(name: str, args: dict, session, embedder: EmbeddingProvider):
    if name == "aggregate_spend":
        return rag_tools.aggregate_spend(
            session, category=args.get("category"),
            start=_parse_date(args.get("start")), end=_parse_date(args.get("end")))
    if name == "list_transactions":
        return rag_tools.list_transactions(
            session, category=args.get("category"), merchant=args.get("merchant"),
            start=_parse_date(args.get("start")), end=_parse_date(args.get("end")),
            limit=args.get("limit", 20))
    if name == "get_anomalies":
        return rag_tools.get_anomalies(
            session, start=_parse_date(args.get("start")), end=_parse_date(args.get("end")))
    if name == "semantic_search":
        return rag_tools.semantic_search(
            session, embedder, args["query"], k=args.get("k", 8))
    return {"error": f"unknown tool: {name}"}


def run_agent(question: str, chat: ChatProvider, session,
              embedder: EmbeddingProvider) -> dict:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools_used: list[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        msg = chat.complete(messages, tools=TOOL_SCHEMAS)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return {"answer": msg.get("content") or "", "tools_used": tools_used}
        messages.append(msg)
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")
            result = _execute(name, args, session, embedder)
            tools_used.append({"tool": name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result)})

    return {"answer": FALLBACK_ANSWER, "tools_used": tools_used}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/rag/agent.py tests/test_agent.py
git commit -m "feat: add hybrid RAG agent with bounded tool loop"
```

---

### Task 4: FastAPI app + /chat endpoint

**Files:**
- Create: `backend/app/main.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/deps.py`
- Create: `backend/app/api/chat.py`
- Test: `backend/tests/test_api_chat.py`

**Interfaces:**
- Consumes: `run_agent`, providers, `SessionLocal`, `ChatMessage` model.
- Produces:
  - `app.main.app` — FastAPI instance with `GET /health` and `POST /chat`.
  - `POST /chat` body `{"message": str, "session_id": str = "default"}` → `{"answer": str, "tools_used": list}`; persists user + assistant `ChatMessage` rows (assistant row stores `tool_calls`).
  - `app.api.deps.get_db`, `get_chat_provider`, `get_embedder` — FastAPI dependencies, overridable in tests via `app.dependency_overrides`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_api_chat.py`:
```python
import json

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.deps import get_chat_provider, get_embedder
from app.main import app
from app.models import ChatMessage


class OneShotChat:
    def complete(self, messages, tools=None):
        return {"role": "assistant", "content": "You spent $0 last month."}


class FakeEmbedder:
    def embed(self, texts):
        return [[0.0] * 1024 for _ in texts]


def test_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_returns_answer_and_persists_messages(session):
    app.dependency_overrides[get_chat_provider] = lambda: OneShotChat()
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    client = TestClient(app)
    try:
        resp = client.post("/chat", json={"message": "how much last month?",
                                          "session_id": "t1"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "You spent $0 last month."
    assert body["tools_used"] == []

    rows = session.scalars(
        select(ChatMessage).where(ChatMessage.session_id == "t1").order_by(ChatMessage.id)
    ).all()
    assert [r.role for r in rows] == ["user", "assistant"]
    assert rows[1].content == "You spent $0 last month."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_chat.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Implement**

`backend/app/api/__init__.py`: (empty file)

`backend/app/api/deps.py`:
```python
from app.config import get_settings
from app.db import SessionLocal
from app.providers.groq import GroqChatProvider
from app.providers.nvidia import NvidiaEmbeddingProvider


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_chat_provider():
    s = get_settings()
    return GroqChatProvider(api_key=s.groq_api_key, model=s.groq_model)


def get_embedder():
    s = get_settings()
    return NvidiaEmbeddingProvider(api_key=s.nvidia_api_key, model=s.nvidia_embed_model,
                                   input_type="query")
```

`backend/app/api/chat.py`:
```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_chat_provider, get_db, get_embedder
from app.models import ChatMessage
from app.rag.agent import run_agent

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[dict]


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db=Depends(get_db),
         chat_provider=Depends(get_chat_provider), embedder=Depends(get_embedder)):
    result = run_agent(req.message, chat_provider, db, embedder)
    db.add(ChatMessage(session_id=req.session_id, role="user", content=req.message))
    db.add(ChatMessage(session_id=req.session_id, role="assistant",
                       content=result["answer"], tool_calls={"used": result["tools_used"]}))
    db.commit()
    return ChatResponse(answer=result["answer"], tools_used=result["tools_used"])
```

`backend/app/main.py`:
```python
from fastapi import FastAPI

from app.api.chat import router as chat_router

app = FastAPI(title="AI Financial Insights API")
app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Run tests, then the full suite**

Run: `pytest tests/test_api_chat.py -v && pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/main.py app/api tests/test_api_chat.py
git commit -m "feat: add fastapi app with /chat endpoint"
```

---

### Task 5: Pure-vector baseline (approach A)

**Files:**
- Create: `backend/app/rag/baseline.py`
- Test: `backend/tests/test_baseline.py`

**Interfaces:**
- Consumes: `semantic_search`, `ChatProvider`.
- Produces: `answer_pure_rag(question: str, chat, session, embedder, k: int = 8) -> str` — top-k semantic retrieval stuffed into a prompt, single completion, NO tools. This is the naive-RAG comparator for the eval.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_baseline.py`:
```python
from datetime import date

from app.models import Transaction
from app.rag.baseline import answer_pure_rag


class CapturingChat:
    def __init__(self):
        self.last_messages = None

    def complete(self, messages, tools=None):
        self.last_messages = messages
        assert tools is None  # baseline never uses tools
        return {"role": "assistant", "content": "Roughly $80."}


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]


def test_baseline_stuffs_retrieved_context(session):
    session.add(Transaction(date=date(2026, 6, 3), merchant="Whole Foods", amount=80.0,
                            category="Groceries", description="food",
                            is_anomaly=False, embedding=[1.0] + [0.0] * 1023))
    session.commit()

    chat = CapturingChat()
    answer = answer_pure_rag("groceries in June?", chat, session, FakeEmbedder(), k=3)

    assert answer == "Roughly $80."
    user_msg = chat.last_messages[-1]["content"]
    assert "Whole Foods" in user_msg
    assert "groceries in June?" in user_msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_baseline.py -v`
Expected: FAIL — `ImportError` (no `app.rag.baseline`).

- [ ] **Step 3: Implement**

`backend/app/rag/baseline.py`:
```python
from app.providers.base import ChatProvider, EmbeddingProvider
from app.rag.tools import semantic_search

BASELINE_SYSTEM = (
    "Answer the user's personal-finance question using ONLY the transaction "
    "context provided. If the context is insufficient, say so."
)


def answer_pure_rag(question: str, chat: ChatProvider, session,
                    embedder: EmbeddingProvider, k: int = 8) -> str:
    rows = semantic_search(session, embedder, question, k=k)
    context = "\n".join(
        f"{r['date']} | {r['category']} | {r['merchant']} | ${r['amount']:.2f}"
        for r in rows
    )
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    msg = chat.complete(messages)
    return msg.get("content") or ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_baseline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/rag/baseline.py tests/test_baseline.py
git commit -m "feat: add pure-vector RAG baseline for eval comparison"
```

---

### Task 6: Eval harness (A vs B → the accuracy number)

**Files:**
- Create: `backend/app/eval/__init__.py`
- Create: `backend/app/eval/questions.py`
- Create: `backend/app/eval/scoring.py`
- Create: `backend/app/eval/run_eval.py`
- Test: `backend/tests/test_eval.py`

**Interfaces:**
- Consumes: `generate_transactions` (ground truth source), `run_agent` (B), `answer_pure_rag` (A).
- Produces:
  - `EvalQuestion` (Pydantic): `question: str`, `expected: float`.
  - `build_eval_set(txns) -> list[EvalQuestion]` — per-month per-category "How much did I spend on X in <Month YYYY>?" questions whose expected values are computed directly from the generated data (skips months with zero spend in that category). Deterministic.
  - `extract_number(text) -> float | None` — last `$`-ish number in a text.
  - `is_correct(answer_text, expected) -> bool` — any extracted number within `max(0.51, 1% of expected)`.
  - `python -m app.eval.run_eval` — manual, live-API runner: scores A and B, prints a table, writes `docs/eval-results.md`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_eval.py`:
```python
from app.data.generator import generate_transactions
from app.eval.questions import build_eval_set
from app.eval.scoring import extract_number, is_correct


def test_extract_number_handles_currency_forms():
    assert extract_number("You spent $1,234.56 in June.") == 1234.56
    assert extract_number("Total: 80 dollars") == 80.0
    assert extract_number("no numbers here") is None


def test_is_correct_within_tolerance():
    assert is_correct("about $100.40", 100.0)          # within 1%
    assert is_correct("$100.50 total", 100.0)          # within $0.51
    assert not is_correct("$150.00", 100.0)
    assert not is_correct("I don't know", 100.0)


def test_build_eval_set_ground_truth_matches_data():
    txns = generate_transactions(months=3, seed=42)
    questions = build_eval_set(txns)

    assert len(questions) >= 10
    q = questions[0]
    # Recompute independently: sum for that category+month must equal expected
    # (the question embeds "<Category> in <Month YYYY>")
    assert q.expected > 0
    assert "How much did I spend on" in q.question


def test_build_eval_set_is_deterministic():
    a = build_eval_set(generate_transactions(months=3, seed=42))
    b = build_eval_set(generate_transactions(months=3, seed=42))
    assert [(x.question, x.expected) for x in a] == [(x.question, x.expected) for x in b]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.eval'`.

- [ ] **Step 3: Implement**

`backend/app/eval/__init__.py`: (empty file)

`backend/app/eval/questions.py`:
```python
from collections import defaultdict

from pydantic import BaseModel

from app.data.generator import GeneratedTxn

SPEND_EXCLUDED = ("Salary",)

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


class EvalQuestion(BaseModel):
    question: str
    expected: float


def build_eval_set(txns: list[GeneratedTxn]) -> list[EvalQuestion]:
    sums: dict[tuple[int, int, str], float] = defaultdict(float)
    for t in txns:
        if t.category in SPEND_EXCLUDED:
            continue
        sums[(t.date.year, t.date.month, t.category)] += t.amount

    questions = [
        EvalQuestion(
            question=(f"How much did I spend on {category} in "
                      f"{MONTH_NAMES[month - 1]} {year}?"),
            expected=round(total, 2),
        )
        for (year, month, category), total in sorted(sums.items())
        if total > 0
    ]
    return questions
```

`backend/app/eval/scoring.py`:
```python
import re

_NUMBER = re.compile(r"\$?(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{1,2}))?")


def extract_number(text: str) -> float | None:
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    whole, frac = matches[-1]
    value = float(whole.replace(",", ""))
    if frac:
        value += float(f"0.{frac}")
    return value


def is_correct(answer_text: str, expected: float) -> bool:
    tolerance = max(0.51, expected * 0.01)
    value = extract_number(answer_text)
    return value is not None and abs(value - expected) <= tolerance
```

`backend/app/eval/run_eval.py`:
```python
"""Live A-vs-B eval. Requires real API keys in .env and a seeded DB (seed 42).

Usage: python -m app.eval.run_eval [--limit N]
Writes docs/eval-results.md (relative to repo root) and prints a summary table.
"""
import argparse
import time
from pathlib import Path

from app.config import get_settings
from app.data.generator import generate_transactions
from app.db import SessionLocal
from app.eval.questions import build_eval_set
from app.eval.scoring import is_correct
from app.providers.groq import GroqChatProvider
from app.providers.nvidia import NvidiaEmbeddingProvider
from app.rag.agent import run_agent
from app.rag.baseline import answer_pure_rag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N questions")
    args = parser.parse_args()

    settings = get_settings()
    chat = GroqChatProvider(api_key=settings.groq_api_key, model=settings.groq_model)
    embedder = NvidiaEmbeddingProvider(api_key=settings.nvidia_api_key,
                                       model=settings.nvidia_embed_model,
                                       input_type="query")

    questions = build_eval_set(generate_transactions(months=6, seed=42))
    if args.limit:
        questions = questions[: args.limit]

    results = []
    with SessionLocal() as session:
        for i, q in enumerate(questions, 1):
            a_answer = answer_pure_rag(q.question, chat, session, embedder)
            time.sleep(1)  # be polite to free-tier rate limits
            b_answer = run_agent(q.question, chat, session, embedder)["answer"]
            time.sleep(1)
            a_ok = is_correct(a_answer, q.expected)
            b_ok = is_correct(b_answer, q.expected)
            results.append((q, a_ok, b_ok, a_answer, b_answer))
            print(f"[{i}/{len(questions)}] A={'Y' if a_ok else 'n'} "
                  f"B={'Y' if b_ok else 'n'}  {q.question}")

    a_acc = sum(1 for _, a, _, _, _ in results if a) / len(results)
    b_acc = sum(1 for _, _, b, _, _ in results if b) / len(results)

    print(f"\nPure vector RAG (A): {a_acc:.0%}")
    print(f"Hybrid agentic RAG (B): {b_acc:.0%}")
    print(f"Delta: +{(b_acc - a_acc):.0%}")

    out = Path(__file__).resolve().parents[3] / "docs" / "eval-results.md"
    lines = [
        "# RAG Eval Results (A: pure vector vs B: hybrid agentic)",
        "",
        f"- Questions: {len(results)}",
        f"- **Pure vector RAG (A): {a_acc:.0%}**",
        f"- **Hybrid agentic RAG (B): {b_acc:.0%}**",
        f"- **Delta: +{(b_acc - a_acc):.0%}**",
        "",
        "| # | Question | Expected | A ok | B ok |",
        "|---|----------|----------|------|------|",
    ]
    for i, (q, a_ok, b_ok, _, _) in enumerate(results, 1):
        lines.append(f"| {i} | {q.question} | {q.expected:.2f} | "
                     f"{'✅' if a_ok else '❌'} | {'✅' if b_ok else '❌'} |")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then the full suite**

Run: `pytest tests/test_eval.py -v && pytest -q`
Expected: all pass. (`run_eval.py` is NOT exercised by tests — it's the manual live runner.)

- [ ] **Step 5: Commit**

```bash
git add app/eval tests/test_eval.py
git commit -m "feat: add A-vs-B eval harness with generator-derived ground truth"
```

- [ ] **Step 6: (Manual, live keys required) Run the eval**

With rotated keys in `.env` and the DB seeded (`python -m app.seed`):
```bash
python -m app.eval.run_eval --limit 25
```
Expected: summary table printed; `docs/eval-results.md` written. **The B-minus-A delta is the project's headline accuracy number** — update the README/resume claim to whatever was actually measured.

---

## Self-Review

**Spec coverage (Plan 2 slice):** hybrid agent + 4 tools ✓ (Tasks 2–3); tool-usage trace surfaced ✓ (agent returns `tools_used`, `/chat` returns it, Task 4); `/chat` endpoint + message persistence ✓ (Task 4); semantic search via Python cosine per environment adaptation ✓ (Tasks 1–2); measurable A-vs-B accuracy claim ✓ (Tasks 5–6). Deferred: insights + anomaly explanation (Plan 3), frontend chat UI (Plan 4).

**Placeholder scan:** none — every step has complete code and exact commands. ✓

**Type consistency:** `run_agent(question, chat, session, embedder) -> {"answer", "tools_used"}` used identically in Tasks 3, 4, 6; tool signatures in Task 2 match `_execute` dispatch in Task 3; `EvalQuestion.question/.expected` consistent between Tasks 6's modules; `input_type` kwarg introduced in Task 1 and used in Tasks 4, 6. ✓

**Bounded loops:** agent loop capped at `MAX_TOOL_ROUNDS = 5` with graceful fallback (tested); eval loop is finite over the question list. ✓
