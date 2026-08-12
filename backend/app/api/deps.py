from app.config import get_settings
from app.db import SessionLocal
from app.providers.groq import GroqChatProvider
from app.providers.nvidia import NvidiaEmbeddingProvider


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_chat_provider():
    s = get_settings()
    return GroqChatProvider(api_key=s.groq_api_key, model=s.groq_model)


def get_embedder():
    s = get_settings()
    return NvidiaEmbeddingProvider(
        api_key=s.nvidia_api_key,
        model=s.nvidia_embed_model,
        input_type="query",
    )
