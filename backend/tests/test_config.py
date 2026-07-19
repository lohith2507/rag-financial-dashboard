from app.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("NVIDIA_API_KEY", "n-key")
    monkeypatch.setenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./data/test.db")

    s = Settings()

    assert s.groq_api_key == "g-key"
    assert s.nvidia_embed_model == "nvidia/nv-embedqa-e5-v5"
    assert s.database_url.endswith("test.db")
