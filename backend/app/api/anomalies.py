from fastapi import APIRouter, Depends

from app.anomalies.detector import detect_anomalies
from app.anomalies.explainer import explain_anomaly
from app.api.deps import get_chat_provider, get_db

router = APIRouter()

MAX_EXPLAINED = 10


@router.get("/anomalies")
def anomalies(
    explain: bool = False,
    db=Depends(get_db),
    chat=Depends(get_chat_provider),
):
    found = detect_anomalies(db)
    if explain:
        for a in found[:MAX_EXPLAINED]:
            a["explanation"] = explain_anomaly(a, chat)
    return {"anomalies": found}
