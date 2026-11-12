"""
Phase 2 tests — database, CRUD, and history/stats API endpoints.

Run from backend/ with the venv active:
    pytest tests/ -v
"""
from __future__ import annotations

import asyncio
import io
import json
import pathlib

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


def _make_engine_and_session():
    engine = create_async_engine(_TEST_DB_URL, connect_args={"check_same_thread": False})
    Session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    return engine, Session


def _run(coro):
    """Run a coroutine in a fresh event loop (compatible with pytest-asyncio auto mode)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# DB-layer fixtures
# ---------------------------------------------------------------------------

async def _setup_db(engine):
    from sqlalchemy.orm import DeclarativeBase
    # Import Base from connection — but we need tables from the ORM model
    from app.database.connection import Base
    from app.database import models  # noqa: F401 — registers table metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _seed_prediction(session_factory, **overrides):
    """Insert one prediction record and return it."""
    from app.services.prediction_store import create_prediction

    defaults = dict(
        filename="abc123_photo.jpg",
        original_filename="photo.jpg",
        image_path="uploads/abc123_photo.jpg",
        predicted_category="cardboard",
        confidence=0.92,
        vit_prediction="cardboard",
        clip_prediction="cardboard",
        class_probabilities={
            "cardboard": 0.92, "glass": 0.02, "metal": 0.02,
            "paper": 0.02, "plastic": 0.01, "trash": 0.01,
        },
        model_version="1.0",
    )
    defaults.update(overrides)
    async with session_factory() as db:
        return await create_prediction(db, **defaults)


# ---------------------------------------------------------------------------
# 1. Database initialises
# ---------------------------------------------------------------------------

def test_database_initializes():
    """init_db() must create the predictions table without errors."""
    engine, _ = _make_engine_and_session()
    _run(_setup_db(engine))
    _run(engine.dispose())


# ---------------------------------------------------------------------------
# 2. Create prediction
# ---------------------------------------------------------------------------

def test_create_prediction():
    engine, Session = _make_engine_and_session()
    _run(_setup_db(engine))

    record = _run(_seed_prediction(Session))
    assert record.id is not None
    assert record.id >= 1
    assert record.predicted_category == "cardboard"
    assert record.original_filename == "photo.jpg"
    assert record.image_path == "uploads/abc123_photo.jpg"

    _run(engine.dispose())


# ---------------------------------------------------------------------------
# 3. Get prediction by ID
# ---------------------------------------------------------------------------

def test_get_prediction_by_id():
    from app.services.prediction_store import get_prediction_by_id

    engine, Session = _make_engine_and_session()
    _run(_setup_db(engine))
    record = _run(_seed_prediction(Session))

    async def _fetch():
        async with Session() as db:
            return await get_prediction_by_id(db, record.id)

    fetched = _run(_fetch())
    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.confidence == pytest.approx(0.92, abs=1e-4)

    _run(engine.dispose())


# ---------------------------------------------------------------------------
# 4. Pagination
# ---------------------------------------------------------------------------

def test_get_predictions_pagination():
    from app.services.prediction_store import get_predictions

    engine, Session = _make_engine_and_session()
    _run(_setup_db(engine))

    # Insert 5 records
    for i in range(5):
        _run(_seed_prediction(Session, filename=f"file_{i}.jpg", original_filename=f"file_{i}.jpg"))

    async def _fetch(page, page_size):
        async with Session() as db:
            return await get_predictions(db, page=page, page_size=page_size)

    records_p1, total = _run(_fetch(1, 3))
    assert total == 5
    assert len(records_p1) == 3

    records_p2, total2 = _run(_fetch(2, 3))
    assert total2 == 5
    assert len(records_p2) == 2

    _run(engine.dispose())


# ---------------------------------------------------------------------------
# 5. Category filter
# ---------------------------------------------------------------------------

def test_get_predictions_category_filter():
    from app.services.prediction_store import get_predictions

    engine, Session = _make_engine_and_session()
    _run(_setup_db(engine))

    _run(_seed_prediction(Session, predicted_category="glass", filename="g1.jpg", original_filename="g1.jpg"))
    _run(_seed_prediction(Session, predicted_category="metal", filename="m1.jpg", original_filename="m1.jpg"))
    _run(_seed_prediction(Session, predicted_category="glass", filename="g2.jpg", original_filename="g2.jpg"))

    async def _fetch(cat):
        async with Session() as db:
            return await get_predictions(db, category_filter=cat)

    glass_records, total = _run(_fetch("glass"))
    assert total == 2
    assert all(r.predicted_category == "glass" for r in glass_records)

    _run(engine.dispose())


# ---------------------------------------------------------------------------
# 6. Filename search
# ---------------------------------------------------------------------------

def test_get_predictions_search():
    from app.services.prediction_store import get_predictions

    engine, Session = _make_engine_and_session()
    _run(_setup_db(engine))

    _run(_seed_prediction(Session, original_filename="bottle_scan.jpg", filename="x1.jpg"))
    _run(_seed_prediction(Session, original_filename="newspaper.jpg", filename="x2.jpg"))
    _run(_seed_prediction(Session, original_filename="glass_bottle.jpg", filename="x3.jpg"))

    async def _search(q):
        async with Session() as db:
            return await get_predictions(db, search=q)

    records, total = _run(_search("bottle"))
    assert total == 2
    assert all("bottle" in r.original_filename for r in records)

    _run(engine.dispose())


# ---------------------------------------------------------------------------
# 7. Delete prediction (success)
# ---------------------------------------------------------------------------

def test_delete_prediction():
    from app.services.prediction_store import delete_prediction, get_prediction_by_id

    engine, Session = _make_engine_and_session()
    _run(_setup_db(engine))
    record = _run(_seed_prediction(Session))
    record_id = record.id

    async def _delete():
        async with Session() as db:
            return await delete_prediction(db, record_id)

    deleted = _run(_delete())
    assert deleted is True

    async def _fetch():
        async with Session() as db:
            return await get_prediction_by_id(db, record_id)

    fetched = _run(_fetch())
    assert fetched is None

    _run(engine.dispose())


# ---------------------------------------------------------------------------
# 8. Delete prediction (not found)
# ---------------------------------------------------------------------------

def test_delete_prediction_not_found():
    from app.services.prediction_store import delete_prediction

    engine, Session = _make_engine_and_session()
    _run(_setup_db(engine))

    async def _delete():
        async with Session() as db:
            return await delete_prediction(db, 999999)

    deleted = _run(_delete())
    assert deleted is False

    _run(engine.dispose())


# ---------------------------------------------------------------------------
# 9. Category statistics calculated correctly
# ---------------------------------------------------------------------------

def test_get_category_statistics_from_records():
    from app.services.prediction_store import get_category_statistics

    engine, Session = _make_engine_and_session()
    _run(_setup_db(engine))

    # 2 cardboard, 1 plastic, 1 trash
    _run(_seed_prediction(Session, predicted_category="cardboard", filename="a1.jpg", original_filename="a1.jpg"))
    _run(_seed_prediction(Session, predicted_category="cardboard", filename="a2.jpg", original_filename="a2.jpg"))
    _run(_seed_prediction(Session, predicted_category="plastic",   filename="b1.jpg", original_filename="b1.jpg"))
    _run(_seed_prediction(Session, predicted_category="trash",     filename="c1.jpg", original_filename="c1.jpg"))

    async def _stats():
        async with Session() as db:
            return await get_category_statistics(db)

    stats = _run(_stats())
    assert len(stats) == 3  # cardboard, plastic, trash

    top = stats[0]  # highest count first
    assert top["category"] == "cardboard"
    assert top["count"] == 2
    assert top["percentage"] == pytest.approx(50.0, abs=0.1)
    assert top["is_recyclable"] is True

    # trash is not recyclable
    trash_entry = next(s for s in stats if s["category"] == "trash")
    assert trash_entry["is_recyclable"] is False

    _run(engine.dispose())


# ---------------------------------------------------------------------------
# 10. Empty database stats
# ---------------------------------------------------------------------------

def test_empty_database_stats():
    from app.services.prediction_store import get_stats

    engine, Session = _make_engine_and_session()
    _run(_setup_db(engine))

    async def _stats():
        async with Session() as db:
            return await get_stats(db)

    data = _run(_stats())
    assert data["total"] == 0
    assert data["distribution"] == []
    assert data["avg_confidence"] == 0.0
    assert data["recyclable_count"] == 0

    _run(engine.dispose())


# ---------------------------------------------------------------------------
# 11. History API — 404 on delete
# ---------------------------------------------------------------------------

def test_history_api_404_on_delete():
    from app.main import app

    with TestClient(app) as client:
        resp = client.delete("/api/history/999999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 12. History API — pagination response shape
# ---------------------------------------------------------------------------

def test_history_api_pagination_response_shape():
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/api/history?page=1&page_size=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert isinstance(data["items"], list)
    assert data["page"] == 1
    assert data["page_size"] == 5


# ---------------------------------------------------------------------------
# 13. Stats API — empty returns valid structure
# ---------------------------------------------------------------------------

def test_stats_api_empty_database():
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_predictions" in data
    assert "category_distribution" in data
    assert "average_confidence" in data
    assert "recyclable_count" in data
    assert isinstance(data["category_distribution"], list)
    # Empty DB → all zeros
    assert data["total_predictions"] == 0
    assert data["recyclable_count"] == 0


# ---------------------------------------------------------------------------
# 14. Image storage — save and validate
# ---------------------------------------------------------------------------

def test_image_storage_save_upload(tmp_path):
    from app.services.image_storage import save_upload

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    safe_name, rel_path = save_upload(image_bytes, "my photo.jpg", tmp_path)
    assert rel_path.startswith("uploads/")
    assert (tmp_path / safe_name).exists()
    assert "my_photo" in safe_name or "my" in safe_name  # sanitised


def test_image_storage_validates_size():
    from app.services.image_storage import validate_upload
    from fastapi import HTTPException

    big_bytes = b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024)
    with pytest.raises(HTTPException) as exc_info:
        validate_upload(big_bytes, "big.jpg")
    assert exc_info.value.status_code == 400


def test_image_storage_validates_extension():
    from app.services.image_storage import validate_upload
    from fastapi import HTTPException

    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="JPEG")
    with pytest.raises(HTTPException) as exc_info:
        validate_upload(buf.getvalue(), "script.exe")
    assert exc_info.value.status_code == 400
