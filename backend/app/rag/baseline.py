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
