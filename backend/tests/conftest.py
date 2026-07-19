import os

# Force SQLite for tests before app.db binds the engine (overrides .env Postgres URLs).
os.environ["DATABASE_URL"] = "sqlite:///./data/test.db"

import pytest
from sqlalchemy import text

from app.db import Base, SessionLocal, engine


@pytest.fixture(autouse=True)
def _clean_tables():
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        for table in ("chat_messages", "insights", "transactions"):
            conn.execute(text(f"DELETE FROM {table}"))
    yield


@pytest.fixture
def session():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
