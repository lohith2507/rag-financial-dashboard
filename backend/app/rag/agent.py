import json
from datetime import date

from app.providers.base import ChatProvider, EmbeddingProvider
from app.rag import tools as rag_tools

MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = (
    "You are a personal-finance assistant. Ground EVERY answer in the user's real "
    "transaction data by calling tools. For any question about totals, sums, or "
    "'how much', ALWAYS call aggregate_spend — never compute totals from search "
    "results. Use semantic_search for fuzzy or descriptive questions. Dates are "
    "ISO YYYY-MM-DD. Answer in plain English and cite concrete numbers."
)

_DATE_PARAM = {"type": "string", "description": "ISO date YYYY-MM-DD"}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "aggregate_spend",
            "description": (
                "Sum spending (excludes salary/income) with optional category and date range. "
                "The ONLY reliable way to answer 'how much did I spend'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "start": _DATE_PARAM,
                    "end": _DATE_PARAM,
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transactions",
            "description": "List individual transactions with optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "merchant": {"type": "string"},
                    "start": _DATE_PARAM,
                    "end": _DATE_PARAM,
                    "limit": {"type": "integer", "default": 20},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anomalies",
            "description": "Return transactions flagged as anomalous/unusual.",
            "parameters": {
                "type": "object",
                "properties": {"start": _DATE_PARAM, "end": _DATE_PARAM},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Fuzzy semantic search over transaction descriptions "
                "(e.g. 'subscriptions I forgot about')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        },
    },
]

FALLBACK_ANSWER = (
    "I couldn't finish answering that within my tool budget — "
    "try a more specific question."
)


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _execute(name: str, args: dict, session, embedder: EmbeddingProvider):
    if name == "aggregate_spend":
        return rag_tools.aggregate_spend(
            session,
            category=args.get("category"),
            start=_parse_date(args.get("start")),
            end=_parse_date(args.get("end")),
        )
    if name == "list_transactions":
        return rag_tools.list_transactions(
            session,
            category=args.get("category"),
            merchant=args.get("merchant"),
            start=_parse_date(args.get("start")),
            end=_parse_date(args.get("end")),
            limit=args.get("limit", 20),
        )
    if name == "get_anomalies":
        return rag_tools.get_anomalies(
            session,
            start=_parse_date(args.get("start")),
            end=_parse_date(args.get("end")),
        )
    if name == "semantic_search":
        return rag_tools.semantic_search(
            session, embedder, args["query"], k=args.get("k", 8)
        )
    return {"error": f"unknown tool: {name}"}


def run_agent(
    question: str, chat: ChatProvider, session, embedder: EmbeddingProvider
) -> dict:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools_used: list[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        msg = chat.complete(messages, tools=TOOL_SCHEMAS)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return {"answer": msg.get("content") or "", "tools_used": tools_used}
        messages.append(msg)
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")
            result = _execute(name, args, session, embedder)
            tools_used.append({"tool": name, "args": args})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                }
            )

    return {"answer": FALLBACK_ANSWER, "tools_used": tools_used}
