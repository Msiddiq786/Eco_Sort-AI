"""
POST /api/predict — accept an image and return the ensemble classification.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.dependencies import get_model_registry
from app.database.connection import get_db
from app.ml.clip_model import predict_clip
from app.ml.ensemble import fuse
from app.ml.preprocessor import load_image_pil
from app.ml.vit_model import predict_vit
from app.schemas.predict import (
    ClassProbabilities,
    CLIPPredictResponse,
    PredictResponse,
    RecyclingDecisionSchema,
)
from app.services.image_storage import save_upload, validate_upload
from app.services.prediction_store import create_prediction
from app.services.recommendation import get_recommendation
from app.services.recycling_engine import get_recycling_decision

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_image(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    registry=Depends(get_model_registry),
):
    # ── 1. Guard: models must be loaded ──────────────────────────────────────
    if not registry.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The fine-tuned ViT model is not loaded. "
                "Please run backend/training/train_vit.py to train the model, "
                "place the checkpoint in backend/models/, and restart the server."
            ),
        )

    # ── 2. Read image bytes ───────────────────────────────────────────────────
    image_bytes = await file.read()
    original_filename = file.filename or "upload.jpg"

    # ── 3. Validate (extension + size + magic bytes) ─────────────────────────
    validate_upload(image_bytes, original_filename)

    # ── 4. Persist upload to disk ─────────────────────────────────────────────
    safe_name, image_path = save_upload(
        image_bytes, original_filename, settings.UPLOAD_DIR
    )

    # ── 5. Preprocess ─────────────────────────────────────────────────────────
    try:
        pil_image = load_image_pil(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode image: {exc}",
        )

    # ── 6. ViT inference ─────────────────────────────────────────────────────
    try:
        vit_probs = predict_vit(registry.vit_model, registry.vit_processor, pil_image)
    except Exception as exc:
        logger.exception("ViT inference failed")
        raise HTTPException(status_code=500, detail=f"ViT inference error: {exc}")

    # ── 7. CLIP inference (optional — degrade gracefully) ─────────────────────
    if registry.clip_loaded:
        try:
            clip_res = predict_clip(
                registry.clip_model, registry.clip_processor, pil_image
            )
            clip_probs = (
                clip_res["probabilities"]
                if isinstance(clip_res, dict) and "probabilities" in clip_res
                else clip_res
            )
        except Exception as exc:
            logger.warning("CLIP inference failed, falling back to ViT only: %s", exc)
            clip_probs = vit_probs
    else:
        clip_probs = vit_probs

    # ── 8. Ensemble fusion ───────────────────────────────────────────────────
    result = fuse(vit_probs, clip_probs)

    # ── 9. Recycling recommendation & decision engine ────────────────────────
    recommendation = get_recommendation(result.predicted_category)
    rec_decision = get_recycling_decision(
        category=result.predicted_category,
        confidence=result.confidence,
        is_ood=result.is_out_of_distribution,
        confidence_status=result.confidence_status,
    )

    # ── 10. Persist record ───────────────────────────────────────────────────
    record = await create_prediction(
        db,
        filename=safe_name,
        original_filename=original_filename,
        image_path=image_path,
        predicted_category=result.predicted_category,
        confidence=result.confidence,
        vit_prediction=result.vit_prediction,
        clip_prediction=result.clip_prediction,
        class_probabilities=result.class_probabilities,
        model_version=settings.MODEL_VERSION,
    )

    vit_prob_obj = ClassProbabilities(**result.vit_probabilities) if result.vit_probabilities else None
    clip_prob_obj = ClassProbabilities(**result.clip_probabilities) if result.clip_probabilities else None
    cal_prob_obj = ClassProbabilities(**result.calibrated_probabilities) if result.calibrated_probabilities else None

    return PredictResponse(
        id=record.id,
        filename=safe_name,
        original_filename=original_filename,
        image_path=image_path,
        predicted_category=result.predicted_category,
        confidence=result.confidence,
        calibrated_confidence=result.calibrated_confidence,
        vit_prediction=result.vit_prediction,
        clip_prediction=result.clip_prediction,
        class_probabilities=ClassProbabilities(**result.class_probabilities),
        calibrated_probabilities=cal_prob_obj,
        vit_probabilities=vit_prob_obj,
        clip_probabilities=clip_prob_obj,
        is_low_confidence=result.is_low_confidence,
        is_ambiguous=result.is_ambiguous,
        is_out_of_distribution=result.is_out_of_distribution,
        confidence_status=result.confidence_status,
        ood_warning=result.ood_warning,
        recycling_recommendation=recommendation,
        recycling_decision=RecyclingDecisionSchema(**rec_decision.to_dict()),
        model_version=settings.MODEL_VERSION,
        created_at=record.created_at,
    )


@router.post(
    "/predict/clip",
    response_model=CLIPPredictResponse,
    status_code=status.HTTP_200_OK,
)
async def predict_clip_only(
    file: UploadFile,
    registry=Depends(get_model_registry),
):
    """
    Direct zero-shot classification using the CLIP foundation model.

    Phase 5 standalone inference endpoint (does not persist predictions
    or apply ensemble fusion).
    """
    if (
        not registry.clip_loaded
        or registry.clip_model is None
        or registry.clip_processor is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "CLIP foundation model is not loaded or is unavailable for inference. "
                "Ensure CLIP weights are available and restart the server."
            ),
        )

    image_bytes = await file.read()
    original_filename = file.filename or "upload.jpg"

    validate_upload(image_bytes, original_filename)

    try:
        pil_image = load_image_pil(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode image: {exc}",
        )

    try:
        clip_res = predict_clip(
            registry.clip_model, registry.clip_processor, pil_image
        )
    except Exception as exc:
        logger.exception("CLIP standalone inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CLIP inference error: {exc}",
        )

    return CLIPPredictResponse(
        predicted_category=clip_res.predicted_category,
        confidence=clip_res.confidence,
        probabilities=ClassProbabilities(**clip_res.probabilities),
    )
