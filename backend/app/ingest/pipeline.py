from app.data.generator import GeneratedTxn
from app.models import Transaction
from app.providers.base import EmbeddingProvider


def txn_to_text(txn: GeneratedTxn) -> str:
    return (
        f"{txn.date.isoformat()} | {txn.category} | {txn.merchant} | "
        f"${txn.amount:.2f} | {txn.description}"
    )


def ingest(txns: list[GeneratedTxn], embedder: EmbeddingProvider, session) -> int:
    vectors = embedder.embed([txn_to_text(t) for t in txns])
    for txn, vector in zip(txns, vectors, strict=True):
        session.add(
            Transaction(
                date=txn.date,
                merchant=txn.merchant,
                amount=txn.amount,
                category=txn.category,
                description=txn.description,
                is_anomaly=txn.is_anomaly,
                embedding=vector,
            )
        )
    session.commit()
    return len(txns)
