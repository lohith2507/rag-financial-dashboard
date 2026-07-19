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
