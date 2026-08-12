from datetime import date

import pytest

from app.models import Transaction
from app.rag.tools import aggregate_spend, get_anomalies, list_transactions, semantic_search


def _txn(d, merchant, amount, category, anomaly=False, embedding=None):
    return Transaction(
        date=d,
        merchant=merchant,
        amount=amount,
        category=category,
        description=f"{category} at {merchant}",
        is_anomaly=anomaly,
        embedding=embedding,
    )


@pytest.fixture
def seeded(session):
    session.add_all(
        [
            _txn(date(2026, 6, 1), "ACME Corp Payroll", 5200.0, "Salary"),
            _txn(date(2026, 6, 3), "Whole Foods", 80.0, "Groceries", embedding=[1.0] + [0.0] * 1023),
            _txn(date(2026, 6, 10), "Safeway", 20.0, "Groceries", embedding=[0.0, 1.0] + [0.0] * 1022),
            _txn(
                date(2026, 6, 15),
                "Netflix",
                15.0,
                "Subscriptions",
                anomaly=False,
                embedding=[0.9, 0.1] + [0.0] * 1022,
            ),
            _txn(date(2026, 7, 2), "Uber", 900.0, "Transport", anomaly=True),
        ]
    )
    session.commit()
    return session


def test_aggregate_spend_excludes_salary(seeded):
    result = aggregate_spend(seeded)
    assert result["total"] == pytest.approx(1015.0)
    assert result["count"] == 4


def test_aggregate_spend_filters_category_and_dates(seeded):
    result = aggregate_spend(
        seeded, category="Groceries", start=date(2026, 6, 1), end=date(2026, 6, 30)
    )
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
        return [[1.0] + [0.0] * 1023 for _ in texts]


def test_semantic_search_ranks_by_cosine(seeded):
    rows = semantic_search(seeded, FakeEmbedder(), "organic groceries", k=2)
    assert len(rows) == 2
    assert rows[0]["merchant"] == "Whole Foods"
    assert rows[0]["score"] > rows[1]["score"]
