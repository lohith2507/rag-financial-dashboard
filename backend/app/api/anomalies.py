from fastapi import APIRouter, Depends, HTTPException

from app.anomalies.detector import detect_anomalies
from app.anomalies.explainer import explain_anomaly
from app.api.deps import get_db, get_optional_chat_provider

router = APIRouter()

MAX_EXPLAINED = 10


@router.get("/anomalies")
def anomalies(
    explain: bool = False,
    db=Depends(get_db),
    chat=Depends(get_optional_chat_provider),
):
    found = detect_anomalies(db)
    if explain:
        if chat is None:
            raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured")
        for a in found[:MAX_EXPLAINED]:
            a["explanation"] = explain_anomaly(a, chat)
    return {"anomalies": found}
