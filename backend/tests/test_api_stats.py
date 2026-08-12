from datetime import date

from fastapi.testclient import TestClient

from app.main import app
from app.models import Transaction


def _seed(session):
    session.add_all(
        [
            Transaction(
                date=date(2026, 5, 1),
                merchant="ACME",
                amount=5200.0,
                category="Salary",
                description="d",
                is_anomaly=False,
            ),
            Transaction(
                date=date(2026, 5, 10),
                merchant="WF",
                amount=100.0,
                category="Groceries",
                description="d",
                is_anomaly=False,
            ),
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
                is_anomaly=False,
            ),
        ]
    )
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
