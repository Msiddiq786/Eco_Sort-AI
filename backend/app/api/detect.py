"""
POST /api/detect — Multi-Object Waste Detection and Material Classification.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.dependencies import get_model_registry
from app.database.connection import get_db
from app.ml.detector import detect_and_classify_objects
from app.ml.preprocessor import load_image_pil
from app.schemas.detect import DetectResponse
from app.services.image_storage import save_upload, validate_upload
from app.services.prediction_store import create_detection_record

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect", response_model=DetectResponse, status_code=status.HTTP_200_OK)
async def detect_multi_objects(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    registry=Depends(get_model_registry),
):
    """
    Detect multiple waste items in an image, generate bounding boxes,
    classify each item's material via ViT + CLIP ensemble, and provide
    per-object intelligent recycling recommendations.
    """
    # 1. Guard: classifier must be ready
    if not registry.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The ViT classifier is not loaded. "
                "Ensure model weights are available in backend/models/ and restart the server."
            ),
        )

    # 2. Read image bytes
    image_bytes = await file.read()
    original_filename = file.filename or "upload.jpg"

    # 3. Validate image format and size
    validate_upload(image_bytes, original_filename)

    # 4. Decode PIL Image
    try:
        pil_image = load_image_pil(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode image for multi-object detection: {exc}",
        )

    # 5. Run detection and classification pipeline
    try:
        response = detect_and_classify_objects(
            pil_image=pil_image,
            vit_model=registry.vit_model,
            vit_processor=registry.vit_processor,
            clip_model=registry.clip_model if registry.clip_loaded else None,
            clip_processor=registry.clip_processor if registry.clip_loaded else None,
        )
    except Exception as exc:
        logger.exception("Multi-object detection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection pipeline error: {exc}",
        )

    # 6. Persist multi-object scan summary to SQLite History when DB session is active
    if isinstance(db, AsyncSession):
        try:
            safe_name, image_path = save_upload(
                image_bytes, original_filename, settings.UPLOAD_DIR
            )
            objects_summary = [
                {
                    "object_id": obj.object_id,
                    "material_category": obj.material_category,
                    "calibrated_confidence": obj.calibrated_confidence,
                    "vit_prediction": obj.vit_prediction,
                    "clip_prediction": obj.clip_prediction,
                    "recycling_status": obj.recycling_status,
                    "model_agreement": obj.model_agreement,
                    "resin_code": obj.resin_info.resin_code if obj.resin_info else None,
                }
                for obj in response.objects
            ]
            record = await create_detection_record(
                db,
                filename=safe_name,
                original_filename=original_filename,
                image_path=image_path,
                total_objects=response.total_objects,
                category_counts=response.category_counts,
                status_counts=response.recycling_summary.status_counts.model_dump(),
                objects_summary=objects_summary,
                inference_time_ms=response.inference_time_ms,
                detection_model_status=response.detection_model_status,
            )
            response.scan_id = record.id
        except Exception as exc:
            logger.warning("Failed to persist multi-object detection record: %s", exc)

    return response
