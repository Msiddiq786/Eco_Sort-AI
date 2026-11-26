"""
Async CRUD helpers for the Prediction table.

All public functions accept an AsyncSession and return plain Python
objects so callers remain ORM-agnostic.
"""
from __future__ import annotations

import json
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import DetectionRecord, FeedbackCorrection, Prediction

# Categories considered recyclable (not "trash")
RECYCLABLE_CATEGORIES: frozenset[str] = frozenset(
    {"cardboard", "glass", "metal", "paper", "plastic"}
)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

async def create_prediction(
    db: AsyncSession,
    *,
    filename: str,
    original_filename: str,
    image_path: str,
    predicted_category: str,
    confidence: float,
    vit_prediction: str,
    clip_prediction: str,
    class_probabilities: dict,
    model_version: str,
) -> Prediction:
    """Persist a new classification result and return the saved ORM record."""
    record = Prediction(
        filename=filename,
        original_filename=original_filename,
        image_path=image_path,
        predicted_category=predicted_category,
        confidence=confidence,
        vit_prediction=vit_prediction,
        clip_prediction=clip_prediction,
        class_probabilities=json.dumps(class_probabilities),
        model_version=model_version,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


# ---------------------------------------------------------------------------
# Read — single record
# ---------------------------------------------------------------------------

async def get_prediction_by_id(
    db: AsyncSession, prediction_id: int
) -> Optional[Prediction]:
    """Return a single Prediction by primary key, or None."""
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id)
    )
    return result.scalar_one_or_none()


# Keep the old name as an alias so existing callers don't break.
get_prediction = get_prediction_by_id


# ---------------------------------------------------------------------------
# Read — paginated list
# ---------------------------------------------------------------------------

async def get_predictions(
    db: AsyncSession,
    *,
    page: int = 1,
    page_size: int = 20,
    category_filter: Optional[str] = None,
    search: Optional[str] = None,
) -> tuple[List[Prediction], int]:
    """
    Return a (records, total_count) tuple.

    *page* is 1-based.
    *category_filter* matches predicted_category exactly (case-insensitive).
    *search* is a substring match against original_filename.
    """
    stmt = select(Prediction).order_by(Prediction.created_at.desc())

    if category_filter:
        stmt = stmt.where(
            Prediction.predicted_category == category_filter.lower()
        )
    if search:
        # Search on the human-readable original filename
        stmt = stmt.where(Prediction.original_filename.ilike(f"%{search}%"))

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = (await db.execute(count_stmt)).scalar_one()

    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(stmt)
    return list(result.scalars().all()), total


# Keep old name as alias
list_predictions = get_predictions


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

async def delete_prediction(db: AsyncSession, prediction_id: int) -> bool:
    """Delete a record by ID.  Returns True on success, False if not found."""
    record = await get_prediction_by_id(db, prediction_id)
    if record is None:
        return False
    await db.delete(record)
    await db.commit()
    return True


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

async def get_total_prediction_count(db: AsyncSession) -> int:
    """Return the total number of persisted predictions."""
    result = await db.execute(select(func.count()).select_from(Prediction))
    return result.scalar_one()


async def get_category_statistics(db: AsyncSession) -> List[dict]:
    """
    Return per-category stats derived from the database.

    Each entry::

        {
            "category": "cardboard",
            "count": 42,
            "percentage": 35.0,
            "is_recyclable": True,
        }

    Returns an empty list when the database is empty.
    """
    total = await get_total_prediction_count(db)
    if total == 0:
        return []

    dist_result = await db.execute(
        select(
            Prediction.predicted_category,
            func.count(Prediction.id).label("cnt"),
        )
        .group_by(Prediction.predicted_category)
        .order_by(func.count(Prediction.id).desc())
    )

    return [
        {
            "category": row.predicted_category,
            "count": row.cnt,
            "percentage": round(row.cnt / total * 100, 2),
            "is_recyclable": row.predicted_category in RECYCLABLE_CATEGORIES,
        }
        for row in dist_result.all()
    ]


async def get_stats(db: AsyncSession) -> dict:
    """
    Convenience aggregation used by GET /api/stats.

    Returns::

        {
            "total": int,
            "distribution": List[dict],   # from get_category_statistics()
            "avg_confidence": float,
            "recyclable_count": int,
        }
    """
    total = await get_total_prediction_count(db)
    distribution = await get_category_statistics(db)

    avg_result = await db.execute(select(func.avg(Prediction.confidence)))
    avg_confidence: float = avg_result.scalar_one() or 0.0

    # Count records whose predicted_category is recyclable
    recyclable_count = sum(
        entry["count"]
        for entry in distribution
        if entry["is_recyclable"]
    )

    return {
        "total": total,
        "distribution": distribution,
        "avg_confidence": round(avg_confidence, 4),
        "recyclable_count": recyclable_count,
    }


# ---------------------------------------------------------------------------
# Multi-object DetectionRecord helpers
# ---------------------------------------------------------------------------

async def create_detection_record(
    db: AsyncSession,
    *,
    filename: str,
    original_filename: str,
    image_path: str,
    total_objects: int,
    category_counts: dict,
    status_counts: dict,
    objects_summary: list,
    inference_time_ms: float,
    detection_model_status: str,
) -> DetectionRecord:
    """Persist a multi-object detection scan summary."""
    record = DetectionRecord(
        filename=filename,
        original_filename=original_filename,
        image_path=image_path,
        total_objects=total_objects,
        category_counts=json.dumps(category_counts),
        status_counts=json.dumps(status_counts),
        objects_summary=json.dumps(objects_summary),
        inference_time_ms=round(float(inference_time_ms), 2),
        detection_model_status=detection_model_status,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def get_detection_records(
    db: AsyncSession,
    *,
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
) -> tuple[List[DetectionRecord], int]:
    """Return paginated multi-object detection scan records ordered newest-first."""
    base_query = select(DetectionRecord)
    count_query = select(func.count()).select_from(DetectionRecord)

    if search:
        pattern = f"%{search}%"
        base_query = base_query.where(DetectionRecord.original_filename.ilike(pattern))
        count_query = count_query.where(DetectionRecord.original_filename.ilike(pattern))

    total_result = await db.execute(count_query)
    total: int = total_result.scalar_one()

    offset = (page - 1) * page_size
    rows_result = await db.execute(
        base_query.order_by(DetectionRecord.created_at.desc(), DetectionRecord.id.desc())
        .offset(offset)
        .limit(page_size)
    )
    return list(rows_result.scalars().all()), total


async def delete_detection_record(db: AsyncSession, detection_id: int) -> bool:
    """Delete a detection record by primary key."""
    result = await db.execute(select(DetectionRecord).where(DetectionRecord.id == detection_id))
    record = result.scalar_one_or_none()
    if record is None:
        return False
    await db.delete(record)
    await db.commit()
    return True


# ---------------------------------------------------------------------------
# User FeedbackCorrection helpers
# ---------------------------------------------------------------------------

async def create_feedback_correction(
    db: AsyncSession,
    *,
    predicted_category: str,
    corrected_category: str,
    prediction_id: Optional[int] = None,
    detection_record_id: Optional[int] = None,
    object_id: Optional[int] = None,
    source: str = "single",
    notes: Optional[str] = None,
    saved_crop_path: Optional[str] = None,
) -> FeedbackCorrection:
    """Persist a user-verified category correction for future model retraining."""
    record = FeedbackCorrection(
        prediction_id=prediction_id,
        detection_record_id=detection_record_id,
        object_id=object_id,
        predicted_category=predicted_category,
        corrected_category=corrected_category,
        source=source,
        notes=notes,
        saved_crop_path=saved_crop_path,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def get_feedback_summary(
    db: AsyncSession,
    *,
    limit: int = 50,
) -> dict:
    """Return total feedback count, breakdown by corrected category, and recent rows."""
    total_res = await db.execute(select(func.count()).select_from(FeedbackCorrection))
    total = int(total_res.scalar_one() or 0)

    dist_res = await db.execute(
        select(
            FeedbackCorrection.corrected_category,
            func.count(FeedbackCorrection.id).label("cnt"),
        ).group_by(FeedbackCorrection.corrected_category)
    )
    by_cat = {row.corrected_category: int(row.cnt) for row in dist_res.all()}

    recent_res = await db.execute(
        select(FeedbackCorrection)
        .order_by(FeedbackCorrection.created_at.desc(), FeedbackCorrection.id.desc())
        .limit(limit)
    )
    recent = list(recent_res.scalars().all())
    return {
        "total_corrections": total,
        "by_corrected_category": by_cat,
        "recent": recent,
    }
