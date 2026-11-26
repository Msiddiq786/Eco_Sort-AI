"""
GET /api/history  — paginated prediction history.
DELETE /api/history/{id}  — delete a record.
"""
import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.schemas.history import (
    DetectionHistoryItem,
    DetectionObjectSummary,
    HistoryItem,
    PaginatedDetectionHistory,
    PaginatedHistory,
)
from app.services.prediction_store import (
    delete_detection_record,
    delete_prediction,
    get_detection_records,
    get_predictions,
)

router = APIRouter()


@router.get("/history", response_model=PaginatedHistory)
async def get_history(
    page: int = Query(1, ge=1, description="1-based page number"),
    page_size: int = Query(20, ge=1, le=100, description="Records per page"),
    category: str | None = Query(None, description="Filter by predicted_category"),
    search: str | None = Query(None, description="Substring match on original_filename"),
    db: AsyncSession = Depends(get_db),
):
    records, total = await get_predictions(
        db, page=page, page_size=page_size, category_filter=category, search=search
    )
    items = [
        HistoryItem(
            id=r.id,
            filename=r.filename,
            original_filename=r.original_filename,
            image_path=r.image_path,
            predicted_category=r.predicted_category,
            confidence=r.confidence,
            vit_prediction=r.vit_prediction,
            clip_prediction=r.clip_prediction,
            class_probabilities=json.loads(r.class_probabilities),
            created_at=r.created_at,
        )
        for r in records
    ]
    return PaginatedHistory(items=items, total=total, page=page, page_size=page_size)


@router.delete("/history/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_history_item(
    prediction_id: int, db: AsyncSession = Depends(get_db)
):
    deleted = await delete_prediction(db, prediction_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found.",
        )


@router.get("/history/detections", response_model=PaginatedDetectionHistory)
async def get_detection_history(
    page: int = Query(1, ge=1, description="1-based page number"),
    page_size: int = Query(20, ge=1, le=100, description="Records per page"),
    search: str | None = Query(None, description="Substring match on original_filename"),
    db: AsyncSession = Depends(get_db),
):
    records, total = await get_detection_records(
        db, page=page, page_size=page_size, search=search
    )
    items = [
        DetectionHistoryItem(
            id=r.id,
            filename=r.filename,
            original_filename=r.original_filename,
            image_path=r.image_path,
            total_objects=r.total_objects,
            category_counts=json.loads(r.category_counts),
            status_counts=json.loads(r.status_counts),
            objects=[
                DetectionObjectSummary(**obj) for obj in json.loads(r.objects_summary)
            ],
            inference_time_ms=r.inference_time_ms,
            detection_model_status=r.detection_model_status,
            created_at=r.created_at,
        )
        for r in records
    ]
    return PaginatedDetectionHistory(items=items, total=total, page=page, page_size=page_size)


@router.delete("/history/detections/{detection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_detection_history_item(
    detection_id: int, db: AsyncSession = Depends(get_db)
):
    deleted = await delete_detection_record(db, detection_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection scan {detection_id} not found.",
        )
