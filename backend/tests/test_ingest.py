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
