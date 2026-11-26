"""Database connection — async SQLite (default) or PostgreSQL via SQLAlchemy."""
from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

_db_url = settings.DATABASE_URL
_is_sqlite = _db_url.lower().startswith("sqlite")

if _is_sqlite and ":memory:" not in _db_url:
    # Ensure parent folder exists when DATABASE_URL points to a mounted volume (e.g. /data/waste.db)
    raw_path = _db_url.split(":///", 1)[-1] if ":///" in _db_url else ""
    if raw_path and not raw_path.startswith(":"):
        Path(raw_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

_engine_kwargs = {"echo": settings.DEBUG}
if _is_sqlite:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    _engine_kwargs["pool_pre_ping"] = True

engine = create_async_engine(_db_url, **_engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """
    Create all tables on startup and apply lightweight migrations safely
    across both SQLite and PostgreSQL.
    """
    from app.database import models  # noqa: F401 — registers ORM models

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        for column_ddl in [
            "ALTER TABLE predictions ADD COLUMN original_filename VARCHAR(255) NOT NULL DEFAULT ''",
            "ALTER TABLE predictions ADD COLUMN image_path VARCHAR(512) NOT NULL DEFAULT ''",
        ]:
            try:
                async with conn.begin_nested():
                    await conn.execute(text(column_ddl))
            except Exception:
                pass  # Column already exists — safe across SQLite & Postgres


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session for use as a FastAPI dependency."""
    async with AsyncSessionLocal() as session:
        yield session
