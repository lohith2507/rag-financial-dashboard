"""Populate the database with synthetic, embedded transactions.

Usage: python -m app.seed
"""
from app.config import get_settings
from app.data.generator import generate_transactions
from app.db import SessionLocal
from app.ingest.pipeline import ingest
from app.providers.nvidia import NvidiaEmbeddingProvider


def main() -> None:
    settings = get_settings()
    if not settings.nvidia_api_key.strip():
        raise SystemExit("NVIDIA_API_KEY is not configured. Set it in .env before seeding.")

    embedder = NvidiaEmbeddingProvider(
        api_key=settings.nvidia_api_key, model=settings.nvidia_embed_model
    )
    txns = generate_transactions(months=6, seed=42)
    with SessionLocal() as session:
        count = ingest(txns, embedder, session)
    print(f"Seeded {count} transactions")


if __name__ == "__main__":
    main()
