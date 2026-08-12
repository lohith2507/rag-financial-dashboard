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


def aggregate_spend(
    session,
    category: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> dict:
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


def list_transactions(
    session,
    category: str | None = None,
    merchant: str | None = None,
    start: date | None = None,
    end: date | None = None,
    limit: int = 20,
) -> list[dict]:
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


def get_anomalies(
    session, start: date | None = None, end: date | None = None
) -> list[dict]:
    stmt = (
        select(Transaction)
        .where(Transaction.is_anomaly.is_(True))
        .order_by(Transaction.date)
    )
    if start:
        stmt = stmt.where(Transaction.date >= start)
    if end:
        stmt = stmt.where(Transaction.date <= end)
    return [txn_to_dict(t) for t in session.scalars(stmt)]


def semantic_search(
    session, embedder: EmbeddingProvider, query: str, k: int = 8
) -> list[dict]:
    query_vec = embedder.embed([query])[0]
    stmt = select(Transaction).where(Transaction.embedding.is_not(None))
    scored = [
        (cosine(query_vec, t.embedding), t)
        for t in session.scalars(stmt)
        if t.embedding is not None
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**txn_to_dict(t), "score": round(s, 4)} for s, t in scored[:k]]
