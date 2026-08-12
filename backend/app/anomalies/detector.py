import statistics

from sqlalchemy import select

from app.models import Transaction
from app.rag.tools import txn_to_dict

Z_THRESHOLD = 3.0
SPEND_EXCLUDED = ("Salary",)
MIN_CATEGORY_SIZE = 3


def detect_anomalies(session, z_threshold: float = Z_THRESHOLD) -> list[dict]:
    txns = session.scalars(
        select(Transaction).where(Transaction.category.notin_(SPEND_EXCLUDED))
    ).all()

    by_category: dict[str, list[Transaction]] = {}
    for t in txns:
        by_category.setdefault(t.category, []).append(t)

    flagged: list[dict] = []
    for rows in by_category.values():
        if len(rows) < MIN_CATEGORY_SIZE:
            continue
        amounts = [r.amount for r in rows]
        mean = statistics.fmean(amounts)
        stdev = statistics.pstdev(amounts)
        if stdev == 0:
            continue
        for r in rows:
            z = (r.amount - mean) / stdev
            if z > z_threshold:
                flagged.append(
                    {
                        **txn_to_dict(r),
                        "z_score": round(z, 2),
                        "category_mean": round(mean, 2),
                    }
                )

    flagged.sort(key=lambda d: d["z_score"], reverse=True)
    return flagged
