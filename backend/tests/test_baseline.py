from datetime import date

from app.models import Transaction
from app.rag.baseline import answer_pure_rag


class CapturingChat:
    def __init__(self):
        self.last_messages = None

    def complete(self, messages, tools=None):
        self.last_messages = messages
        assert tools is None
        return {"role": "assistant", "content": "Roughly $80."}


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]


def test_baseline_stuffs_retrieved_context(session):
    session.add(
        Transaction(
            date=date(2026, 6, 3),
            merchant="Whole Foods",
            amount=80.0,
            category="Groceries",
            description="food",
            is_anomaly=False,
            embedding=[1.0] + [0.0] * 1023,
        )
    )
    session.commit()

    chat = CapturingChat()
    answer = answer_pure_rag("groceries in June?", chat, session, FakeEmbedder(), k=3)

    assert answer == "Roughly $80."
    user_msg = chat.last_messages[-1]["content"]
    assert "Whole Foods" in user_msg
    assert "groceries in June?" in user_msg
