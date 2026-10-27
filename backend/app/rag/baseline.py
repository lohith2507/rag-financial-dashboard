from app.providers.base import ChatProvider, EmbeddingProvider
from app.rag.tools import semantic_search

BASELINE_SYSTEM = (
    "Answer the user's personal-finance question using ONLY the transaction "
    "context provided. If the context is insufficient, say so."
)


def answer_pure_rag(
    question: str,
    chat: ChatProvider,
    session,
    embedder: EmbeddingProvider,
    k: int = 8,
) -> str:
    """Answer ``question`` with single-shot RAG, no tool calls.

    Retrieves the top ``k`` transactions by semantic similarity, formats each
    as ``date | category | merchant | $amount`` context lines, and asks the
    chat provider once. Serves as the baseline the agent is evaluated
    against. Returns an empty string if the model returns no content.
    """
    rows = semantic_search(session, embedder, question, k=k)
    context = "\n".join(
        f"{r['date']} | {r['category']} | {r['merchant']} | ${r['amount']:.2f}"
        for r in rows
    )
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    msg = chat.complete(messages)
    return msg.get("content") or ""
