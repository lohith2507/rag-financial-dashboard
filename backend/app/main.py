from fastapi import FastAPI

from app.api.anomalies import router as anomalies_router
from app.api.chat import router as chat_router
from app.api.insights import router as insights_router
from app.api.stats import router as stats_router

app = FastAPI(title="AI Financial Insights API")
app.include_router(chat_router)
app.include_router(anomalies_router)
app.include_router(insights_router)
app.include_router(stats_router)


@app.get("/health")
def health():
    return {"status": "ok"}
