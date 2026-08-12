import json

from app.providers.base import ChatProvider

EXPLAIN_SYSTEM = (
    "You are a personal-finance anomaly analyst. Using ONLY the numbers provided, "
    "explain in 1-2 plain-English sentences why this transaction looks unusual."
)


def explain_anomaly(anomaly: dict, chat: ChatProvider) -> str:
    user = (
        f"Transaction: {json.dumps(anomaly)}\n"
        f"The category's average transaction is ${anomaly['category_mean']:.2f}; "
        f"this one is {anomaly['z_score']} standard deviations above it."
    )
    msg = chat.complete(
        [
            {"role": "system", "content": EXPLAIN_SYSTEM},
            {"role": "user", "content": user},
        ]
    )
    return msg.get("content") or ""
