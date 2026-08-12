from fastapi import APIRouter, Depends
from sqlalchemy import select

from app.api.deps import get_chat_provider, get_db
from app.insights.service import generate_monthly_insight
from app.models import Insight

router = APIRouter()


def _to_dict(i: Insight) -> dict:
    return {
        "id": i.id,
        "period": i.period,
        "summary_text": i.summary_text,
        "generated_at": i.generated_at.isoformat() if i.generated_at else None,
    }


@router.post("/insights/{period}/generate")
def generate(period: str, db=Depends(get_db), chat=Depends(get_chat_provider)):
    return _to_dict(generate_monthly_insight(db, chat, period))


@router.get("/insights")
def list_insights(db=Depends(get_db)):
    rows = db.scalars(select(Insight).order_by(Insight.generated_at.desc())).all()
    return {"insights": [_to_dict(i) for i in rows]}
