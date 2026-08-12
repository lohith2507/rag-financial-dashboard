from sqlalchemy import select

from fastapi.testclient import TestClient

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
        resp = client.post(
            "/chat", json={"message": "how much last month?", "session_id": "t1"}
        )
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
