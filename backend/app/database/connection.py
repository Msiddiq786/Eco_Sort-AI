"""Database connection — async SQLite via aiosqlite + SQLAlchemy."""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=settings.DEBUG,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """
    Create all tables on startup and apply lightweight migrations.

    Uses ``CREATE TABLE IF NOT EXISTS`` semantics — safe to call repeatedly.
    For the new columns (original_filename, image_path) added in Phase 2,
    we apply ALTER TABLE ... ADD COLUMN so existing databases are upgraded
    automatically without data loss.
    """
    from app.database import models  # noqa: F401 — registers ORM models

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Phase 2 migration: add columns that did not exist in the Phase 1 schema.
        # SQLite ignores ``ADD COLUMN`` silently if the column already exists when
        # using ``IF NOT EXISTS`` — however SQLite < 3.37 does not support that
        # syntax, so we catch the OperationalError instead.
        from sqlalchemy import text

        for column_ddl in [
            "ALTER TABLE predictions ADD COLUMN original_filename VARCHAR(255) NOT NULL DEFAULT ''",
            "ALTER TABLE predictions ADD COLUMN image_path VARCHAR(512) NOT NULL DEFAULT ''",
        ]:
            try:
                await conn.execute(text(column_ddl))
            except Exception:
                pass  # Column already exists — safe to ignore


async def get_db() -> AsyncSession:
    """Yield a database session for use as a FastAPI dependency."""
    async with AsyncSessionLocal() as session:
        yield session
