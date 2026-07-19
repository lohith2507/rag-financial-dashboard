from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    nvidia_api_key: str = ""
    nvidia_embed_model: str = "nvidia/nv-embedqa-e5-v5"
    database_url: str = "sqlite:///./data/findb.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
