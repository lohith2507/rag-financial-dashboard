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
    session.add_all(
        [
            Transaction(
                date=date(2026, 6, 1),
                merchant="ACME",
                amount=5200.0,
                category="Salary",
                description="d",
                is_anomaly=False,
            ),
            Transaction(
                date=date(2026, 6, 3),
                merchant="WF",
                amount=80.0,
                category="Groceries",
                description="d",
                is_anomaly=False,
            ),
            Transaction(
                date=date(2026, 6, 9),
                merchant="NF",
                amount=15.0,
                category="Subscriptions",
                description="d",
                is_anomaly=True,
            ),
            Transaction(
                date=date(2026, 5, 20),
                merchant="WF",
                amount=60.0,
                category="Groceries",
                description="d",
                is_anomaly=False,
            ),
        ]
    )
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
    assert "95.0" in canned.last_user_content
    assert "2026-05" in canned.last_user_content

    listed = TestClient(app).get("/insights").json()["insights"]
    assert listed[0]["period"] == "2026-06"
    row = session.scalars(select(Insight)).one()
    assert row.summary_text == "June spending was steady."
