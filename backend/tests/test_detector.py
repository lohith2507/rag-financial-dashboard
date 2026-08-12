from datetime import date

from app.anomalies.detector import detect_anomalies
from app.data.generator import generate_transactions
from app.models import Transaction


def _txn(amount, category="Groceries", d=date(2026, 6, 1), anomaly=False):
    return Transaction(
        date=d,
        merchant="M",
        amount=amount,
        category=category,
        description="d",
        is_anomaly=anomaly,
    )


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
    session.add_all(
        [_txn(1800.0, category="Rent", d=date(2026, m, 1)) for m in range(1, 7)]
    )
    session.add_all(
        [
            _txn(9.99, category="Subscriptions"),
            _txn(500.0, category="Subscriptions"),
        ]
    )
    session.commit()

    assert detect_anomalies(session) == []


def test_recall_on_seeded_generator_data(session):
    txns = generate_transactions(months=6, seed=42)
    session.add_all(
        [
            Transaction(
                date=t.date,
                merchant=t.merchant,
                amount=t.amount,
                category=t.category,
                description=t.description,
                is_anomaly=t.is_anomaly,
            )
            for t in txns
        ]
    )
    session.commit()

    flagged_ids = {
        (f["date"], f["merchant"], f["amount"]) for f in detect_anomalies(session)
    }
    planted = [
        (t.date.isoformat(), t.merchant, t.amount) for t in txns if t.is_anomaly
    ]

    assert planted, "generator must plant anomalies"
    for p in planted:
        assert p in flagged_ids, f"planted anomaly not detected: {p}"
