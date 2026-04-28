from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from web.api.config import settings

# Ensure the storage directory exists before SQLite tries to open the DB file.
_db_path = settings.database_url.removeprefix("sqlite:///")
if _db_path and not _db_path.startswith(":"):
    Path(_db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from web.api.models import run, run_item, user  # noqa: F401 — registers ORM classes

    Base.metadata.create_all(bind=engine)
