from datetime import date

from fastapi.testclient import TestClient

from app.api.deps import get_optional_chat_provider
from app.main import app
from app.models import Transaction


class CannedChat:
    def complete(self, messages, tools=None):
        return {"role": "assistant", "content": "This charge is 50x the category average."}


def _seed(session):
    session.add_all(
        [
            Transaction(
                date=date(2026, 6, d),
                merchant="M",
                amount=a,
                category="Groceries",
                description="d",
                is_anomaly=False,
            )
            for d, a in [
                (1, 40),
                (2, 50),
                (3, 60),
                (4, 45),
                (5, 55),
                (6, 50),
                (7, 48),
                (8, 52),
                (9, 47),
                (10, 53),
            ]
        ]
    )
    session.add(
        Transaction(
            date=date(2026, 6, 15),
            merchant="Sketchy Store",
            amount=5000.0,
            category="Groceries",
            description="d",
            is_anomaly=True,
        )
    )
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
    app.dependency_overrides[get_optional_chat_provider] = lambda: CannedChat()
    client = TestClient(app)
    try:
        resp = client.get("/anomalies?explain=true")
    finally:
        app.dependency_overrides.clear()

    anomalies = resp.json()["anomalies"]
    assert anomalies[0]["explanation"] == "This charge is 50x the category average."
