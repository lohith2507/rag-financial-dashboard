import json
from datetime import date

from app.models import Transaction
from app.rag.agent import MAX_TOOL_ROUNDS, run_agent


class ScriptedChat:
    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def complete(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self._script.pop(0)


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0] * 1024 for _ in texts]


def _tool_call(call_id, name, args):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_agent_executes_tool_then_answers(session):
    session.add(
        Transaction(
            date=date(2026, 6, 3),
            merchant="Whole Foods",
            amount=80.0,
            category="Groceries",
            description="food",
            is_anomaly=False,
        )
    )
    session.commit()

    chat = ScriptedChat(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call(
                        "c1",
                        "aggregate_spend",
                        {
                            "category": "Groceries",
                            "start": "2026-06-01",
                            "end": "2026-06-30",
                        },
                    )
                ],
            },
            {"role": "assistant", "content": "You spent $80.00 on groceries in June."},
        ]
    )

    result = run_agent("How much on groceries in June?", chat, session, FakeEmbedder())

    assert result["answer"] == "You spent $80.00 on groceries in June."
    assert result["tools_used"] == [
        {
            "tool": "aggregate_spend",
            "args": {
                "category": "Groceries",
                "start": "2026-06-01",
                "end": "2026-06-30",
            },
        }
    ]
    tool_msgs = [m for m in chat.calls[1]["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert json.loads(tool_msgs[0]["content"])["total"] == 80.0


def test_agent_answers_directly_without_tools(session):
    chat = ScriptedChat([{"role": "assistant", "content": "Hello!"}])
    result = run_agent("hi", chat, session, FakeEmbedder())
    assert result["answer"] == "Hello!"
    assert result["tools_used"] == []


def test_agent_loop_is_bounded(session):
    looping_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call("cx", "get_anomalies", {})],
    }
    chat = ScriptedChat([looping_call] * MAX_TOOL_ROUNDS)

    result = run_agent("weird stuff?", chat, session, FakeEmbedder())

    assert len(chat.calls) == MAX_TOOL_ROUNDS
    assert result["answer"]
