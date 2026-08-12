"""Live A-vs-B eval. Requires real API keys in .env and a seeded DB (seed 42).

Usage: python -m app.eval.run_eval [--limit N]
Writes docs/eval-results.md (relative to repo root) and prints a summary table.
"""
import argparse
import time
from pathlib import Path

from app.config import get_settings
from app.data.generator import generate_transactions
from app.db import SessionLocal
from app.eval.questions import build_eval_set
from app.eval.scoring import is_correct
from app.providers.groq import GroqChatProvider
from app.providers.nvidia import NvidiaEmbeddingProvider
from app.rag.agent import run_agent
from app.rag.baseline import answer_pure_rag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N questions"
    )
    args = parser.parse_args()

    settings = get_settings()
    chat = GroqChatProvider(api_key=settings.groq_api_key, model=settings.groq_model)
    embedder = NvidiaEmbeddingProvider(
        api_key=settings.nvidia_api_key,
        model=settings.nvidia_embed_model,
        input_type="query",
    )

    questions = build_eval_set(generate_transactions(months=6, seed=42))
    if args.limit:
        questions = questions[: args.limit]

    results = []
    with SessionLocal() as session:
        for i, q in enumerate(questions, 1):
            a_answer = answer_pure_rag(q.question, chat, session, embedder)
            time.sleep(1)
            b_answer = run_agent(q.question, chat, session, embedder)["answer"]
            time.sleep(1)
            a_ok = is_correct(a_answer, q.expected)
            b_ok = is_correct(b_answer, q.expected)
            results.append((q, a_ok, b_ok, a_answer, b_answer))
            print(
                f"[{i}/{len(questions)}] A={'Y' if a_ok else 'n'} "
                f"B={'Y' if b_ok else 'n'}  {q.question}"
            )

    a_acc = sum(1 for _, a, _, _, _ in results if a) / len(results)
    b_acc = sum(1 for _, _, b, _, _ in results if b) / len(results)

    print(f"\nPure vector RAG (A): {a_acc:.0%}")
    print(f"Hybrid agentic RAG (B): {b_acc:.0%}")
    print(f"Delta: +{(b_acc - a_acc):.0%}")

    out = Path(__file__).resolve().parents[3] / "docs" / "eval-results.md"
    lines = [
        "# RAG Eval Results (A: pure vector vs B: hybrid agentic)",
        "",
        f"- Questions: {len(results)}",
        f"- **Pure vector RAG (A): {a_acc:.0%}**",
        f"- **Hybrid agentic RAG (B): {b_acc:.0%}**",
        f"- **Delta: +{(b_acc - a_acc):.0%}**",
        "",
        "| # | Question | Expected | A ok | B ok |",
        "|---|----------|----------|------|------|",
    ]
    for i, (q, a_ok, b_ok, _, _) in enumerate(results, 1):
        lines.append(
            f"| {i} | {q.question} | {q.expected:.2f} | "
            f"{'✅' if a_ok else '❌'} | {'✅' if b_ok else '❌'} |"
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
