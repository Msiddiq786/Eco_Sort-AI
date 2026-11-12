"""
POST /api/detect — Multi-Object Waste Detection and Material Classification.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.core.dependencies import get_model_registry
from app.ml.detector import detect_and_classify_objects
from app.ml.preprocessor import load_image_pil
from app.schemas.detect import DetectResponse
from app.services.image_storage import validate_upload

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect", response_model=DetectResponse, status_code=status.HTTP_200_OK)
async def detect_multi_objects(
    file: UploadFile,
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
        return response
    except Exception as exc:
        logger.exception("Multi-object detection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection pipeline error: {exc}",
        )
