import json
from datetime import date

from sqlalchemy import extract, func, select

from app.models import Insight, Transaction
from app.providers.base import ChatProvider

SPEND_EXCLUDED = ("Salary",)

INSIGHT_SYSTEM = (
    "You are a personal-finance analyst. Write a short plain-English monthly summary "
    "(3-5 sentences) using ONLY the JSON stats provided: top categories, notable "
    "changes vs the previous month, and any anomalies. Cite actual numbers."
)


def month_stats(session, year: int, month: int) -> dict:
    def _in_month(stmt):
        return stmt.where(
            extract("year", Transaction.date) == year,
            extract("month", Transaction.date) == month,
        )

    spend_rows = session.execute(
        _in_month(
            select(Transaction.category, func.sum(Transaction.amount))
            .where(Transaction.category.notin_(SPEND_EXCLUDED))
            .group_by(Transaction.category)
        )
    ).all()
    income = session.execute(
        _in_month(
            select(func.coalesce(func.sum(Transaction.amount), 0.0)).where(
                Transaction.category == "Salary"
            )
        )
    ).scalar_one()
    anomaly_count = session.execute(
        _in_month(
            select(func.count(Transaction.id)).where(Transaction.is_anomaly.is_(True))
        )
    ).scalar_one()

    by_category = {cat: round(float(total), 2) for cat, total in spend_rows}
    return {
        "period": f"{year:04d}-{month:02d}",
        "total_spend": round(sum(by_category.values()), 2),
        "income": round(float(income), 2),
        "by_category": by_category,
        "anomaly_count": anomaly_count,
    }


def _previous(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def generate_monthly_insight(session, chat: ChatProvider, period: str) -> Insight:
    year, month = (int(p) for p in period.split("-"))
    current = month_stats(session, year, month)
    prev = month_stats(session, *_previous(year, month))

    user = (
        f"Current month stats: {json.dumps(current)}\n"
        f"Previous month stats: {json.dumps(prev)}"
    )
    msg = chat.complete(
        [
            {"role": "system", "content": INSIGHT_SYSTEM},
            {"role": "user", "content": user},
        ]
    )

    insight = Insight(period=period, summary_text=msg.get("content") or "")
    session.add(insight)
    session.commit()
    return insight
