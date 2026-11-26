"""
POST /api/feedback & GET /api/feedback — User correction loop for active learning.
"""
from __future__ import annotations

import base64
import io
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.connection import get_db
from app.database.models import Prediction
from app.schemas.feedback import (
    FeedbackListItem,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
)
from app.services.prediction_store import (
    create_feedback_correction,
    get_feedback_summary,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_MAX_FEEDBACK_IMAGE_BYTES = 8 * 1024 * 1024


def _save_base64_crop(image_base64: str, target_dir: Path) -> Optional[str]:
    """Decode a Base64/DataURL crop image safely and save it as a JPEG inside target_dir."""
    payload = image_base64.strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]

    raw = base64.b64decode(payload, validate=True)
    if len(raw) > _MAX_FEEDBACK_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feedback image payload exceeds 8 MB limit.",
        )

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"fb_{uuid.uuid4().hex[:12]}.jpg"
    img.save(out_file, format="JPEG", quality=92)
    return str(out_file)


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Record a human-corrected category label and save the corresponding crop/upload
    under FEEDBACK_DIR/<corrected_category>/ for future fine-tuning curation.
    """
    target_dir = Path(settings.FEEDBACK_DIR) / payload.corrected_category
    saved_path: Optional[str] = None

    if payload.image_base64:
        try:
            saved_path = _save_base64_crop(payload.image_base64, target_dir)
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Failed to decode feedback image_base64: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid feedback image_base64 payload: {exc}",
            )
    elif payload.prediction_id is not None:
        pred_res = await db.execute(
            select(Prediction).where(Prediction.id == payload.prediction_id)
        )
        pred = pred_res.scalar_one_or_none()
        if pred is not None:
            candidate_paths = [
                Path(pred.image_path),
                Path(settings.UPLOAD_DIR) / Path(pred.image_path).name,
            ]
            for src in candidate_paths:
                if src.is_file():
                    target_dir.mkdir(parents=True, exist_ok=True)
                    dest = target_dir / f"fb_{uuid.uuid4().hex[:12]}{src.suffix or '.jpg'}"
                    shutil.copy2(src, dest)
                    saved_path = str(dest)
                    break

    record = await create_feedback_correction(
        db,
        predicted_category=payload.predicted_category,
        corrected_category=payload.corrected_category,
        prediction_id=payload.prediction_id,
        detection_record_id=payload.detection_record_id,
        object_id=payload.object_id,
        source=payload.source,
        notes=payload.notes,
        saved_crop_path=saved_path,
    )

    msg = (
        f"Feedback saved: '{payload.predicted_category}' -> '{payload.corrected_category}'."
    )
    if saved_path:
        msg += " Sample archived for active-learning dataset curation."

    return FeedbackResponse(
        id=record.id,
        predicted_category=record.predicted_category,
        corrected_category=record.corrected_category,
        source=record.source,
        saved_crop_path=record.saved_crop_path,
        created_at=record.created_at,
        message=msg,
    )


@router.get("/feedback", response_model=FeedbackStatsResponse)
async def list_feedback(
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Return user correction counts and recent curated feedback records."""
    summary = await get_feedback_summary(db, limit=limit)
    return FeedbackStatsResponse(
        total_corrections=summary["total_corrections"],
        by_corrected_category=summary["by_corrected_category"],
        recent=[FeedbackListItem.model_validate(item) for item in summary["recent"]],
    )
