"""Ensure the SQLite DB path is usable, then exit 0."""
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

from app.config import get_settings


def main() -> int:
    url = get_settings().database_url
    if url.startswith("sqlite:///"):
        db_path = Path(url.removeprefix("sqlite:///"))
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        engine = create_engine(url, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("DB ready")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"DB not ready: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
