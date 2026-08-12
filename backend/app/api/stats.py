from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy import case, extract, func, select

from app.api.deps import get_db
from app.models import Transaction
from app.rag.tools import list_transactions

router = APIRouter()

SPEND_EXCLUDED = ("Salary",)


@router.get("/stats/by-category")
def by_category(start: date | None = None, end: date | None = None, db=Depends(get_db)):
    stmt = (
        select(Transaction.category, func.sum(Transaction.amount).label("total"))
        .where(Transaction.category.notin_(SPEND_EXCLUDED))
        .group_by(Transaction.category)
        .order_by(func.sum(Transaction.amount).desc())
    )
    if start:
        stmt = stmt.where(Transaction.date >= start)
    if end:
        stmt = stmt.where(Transaction.date <= end)
    rows = db.execute(stmt).all()
    return {
        "by_category": [{"category": c, "total": round(float(t), 2)} for c, t in rows]
    }


@router.get("/stats/monthly")
def monthly(db=Depends(get_db)):
    year = extract("year", Transaction.date)
    month = extract("month", Transaction.date)
    spend = func.sum(
        case((Transaction.category.notin_(SPEND_EXCLUDED), Transaction.amount), else_=0.0)
    )
    income = func.sum(
        case((Transaction.category == "Salary", Transaction.amount), else_=0.0)
    )
    rows = db.execute(
        select(year, month, spend, income).group_by(year, month).order_by(year, month)
    ).all()
    return {
        "monthly": [
            {
                "period": f"{int(y):04d}-{int(m):02d}",
                "spend": round(float(s), 2),
                "income": round(float(i), 2),
            }
            for y, m, s, i in rows
        ]
    }


@router.get("/transactions")
def transactions(
    category: str | None = None,
    merchant: str | None = None,
    start: date | None = None,
    end: date | None = None,
    limit: int = 50,
    db=Depends(get_db),
):
    return {
        "transactions": list_transactions(
            db,
            category=category,
            merchant=merchant,
            start=start,
            end=end,
            limit=limit,
        )
    }
