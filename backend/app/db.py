from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {}


_url = get_settings().database_url
engine = create_engine(_url, future=True, **_engine_kwargs(_url))
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
