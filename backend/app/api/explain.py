"""
XAI endpoints for EcoSort AI.

GET  /api/explain/{prediction_id}
     Re-run XAI on an already-stored prediction (loads image from DB path).

POST /api/explain
     Run XAI inline on a freshly-uploaded image (no DB lookup).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_model_registry
from app.database.connection import get_db
from app.ml.clip_model_v2 import predict_clip_v2
from app.ml.preprocessor import load_image_pil
from app.ml.vit_model import predict_vit
from app.ml.xai import build_xai_explanation, generate_attention_heatmap
from app.schemas.predict import XAIExplanation
from app.services.ensemble_optimizer import load_ensemble_config
from app.services.image_storage import validate_upload

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_xai(
    registry,
    pil_image,
    vit_probs: Dict[str, float],
    clip_probs: Dict[str, float],
    ensemble_probs: Dict[str, float],
    predicted_category: str,
    confidence: float,
    include_heatmap: bool = True,
    vit_weight: float = 0.70,
    clip_weight: float = 0.30,
) -> Dict[str, Any]:
    """
    Core XAI computation — shared by both endpoints.
    Returns the raw dict (not Pydantic) for maximum flexibility.
    """
    heatmap_b64: Optional[str] = None

    if include_heatmap and registry.ready:
        try:
            heatmap_b64, _ = generate_attention_heatmap(
                model=registry.vit_model,
                processor=registry.vit_processor,
                pil_image=pil_image,
            )
        except Exception as exc:
            logger.warning("Attention heatmap generation failed (non-fatal): %s", exc)

    return build_xai_explanation(
        predicted_category=predicted_category,
        confidence=confidence,
        vit_probs=vit_probs,
        clip_probs=clip_probs,
        ensemble_probs=ensemble_probs,
        attention_heatmap_b64=heatmap_b64,
        vit_weight=vit_weight,
        clip_weight=clip_weight,
    )


# ---------------------------------------------------------------------------
# GET /api/explain/{prediction_id}
# ---------------------------------------------------------------------------

@router.get(
    "/explain/{prediction_id}",
    response_model=XAIExplanation,
    summary="Explain a stored prediction via ViT Attention Rollout + CLIP scores",
)
async def explain_stored_prediction(
    prediction_id: int,
    include_heatmap: bool = Query(True, description="Set false to skip the attention heatmap (faster)."),
    db: AsyncSession = Depends(get_db),
    registry=Depends(get_model_registry),
):
    """
    Re-runs XAI inference on a stored prediction using the original image path.
    Returns a structured explanation including:
    - ViT Attention Rollout heatmap (base64 PNG, optional)
    - CLIP similarity scores ranked per class
    - Ensemble reasoning text
    """
    if not registry.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ViT model not loaded; cannot produce attention maps.",
        )

    # Load prediction record from DB
    from sqlalchemy import select
    from app.database.models import Prediction

    result = await db.execute(select(Prediction).where(Prediction.id == prediction_id))
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found.",
        )

    # Load original image from stored path
    image_path = Path(record.image_path)
    if not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original image not found at {image_path}.",
        )

    try:
        pil_image = load_image_pil(image_path.read_bytes())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode stored image: {exc}",
        )

    # Re-run inference to get fresh probability distributions
    try:
        vit_probs = predict_vit(registry.vit_model, registry.vit_processor, pil_image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ViT re-inference failed: {exc}")

    clip_probs: Dict[str, float] = {}
    if registry.clip_loaded:
        try:
            clip_res = predict_clip_v2(registry.clip_model, registry.clip_processor, pil_image)
            clip_probs = clip_res.probabilities
        except Exception as exc:
            logger.warning("CLIP re-inference failed: %s", exc)
            clip_probs = {k: vit_probs.get(k, 0.0) for k in vit_probs}
    else:
        clip_probs = {k: vit_probs.get(k, 0.0) for k in vit_probs}

    # Build ensemble probabilities using active weights
    cfg = load_ensemble_config()
    vit_w = float(cfg.get("vit_weight", 0.70))
    clip_w = float(cfg.get("clip_weight", 0.30))

    all_keys = set(list(vit_probs.keys()) + list(clip_probs.keys()))
    ensemble_probs = {
        k: round(vit_w * vit_probs.get(k, 0.0) + clip_w * clip_probs.get(k, 0.0), 4)
        for k in all_keys
    }

    xai_dict = _run_xai(
        registry=registry,
        pil_image=pil_image,
        vit_probs=vit_probs,
        clip_probs=clip_probs,
        ensemble_probs=ensemble_probs,
        predicted_category=record.predicted_category,
        confidence=record.confidence,
        include_heatmap=include_heatmap,
        vit_weight=vit_w,
        clip_weight=clip_w,
    )

    return XAIExplanation(**xai_dict)


# ---------------------------------------------------------------------------
# POST /api/explain  (inline, no DB lookup)
# ---------------------------------------------------------------------------

@router.post(
    "/explain",
    response_model=XAIExplanation,
    summary="Explain a freshly uploaded image (no DB storage)",
)
async def explain_uploaded_image(
    file: UploadFile,
    include_heatmap: bool = Query(True, description="Set false to skip the attention heatmap (faster)."),
    registry=Depends(get_model_registry),
):
    """
    Accepts an image upload, runs ViT + CLIP inference, and returns
    a full XAI explanation without storing to the database.
    """
    if not registry.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ViT model not loaded; cannot produce attention maps.",
        )

    image_bytes = await file.read()
    validate_upload(image_bytes, file.filename or "upload.jpg")

    try:
        pil_image = load_image_pil(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not decode image: {exc}",
        )

    try:
        vit_probs = predict_vit(registry.vit_model, registry.vit_processor, pil_image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ViT inference failed: {exc}")

    clip_probs: Dict[str, float] = {}
    if registry.clip_loaded:
        try:
            clip_res = predict_clip_v2(registry.clip_model, registry.clip_processor, pil_image)
            clip_probs = clip_res.probabilities
        except Exception as exc:
            logger.warning("CLIP inference failed: %s", exc)
            clip_probs = {k: vit_probs.get(k, 0.0) for k in vit_probs}
    else:
        clip_probs = {k: vit_probs.get(k, 0.0) for k in vit_probs}

    cfg = load_ensemble_config()
    vit_w = float(cfg.get("vit_weight", 0.70))
    clip_w = float(cfg.get("clip_weight", 0.30))

    all_keys = set(list(vit_probs.keys()) + list(clip_probs.keys()))
    ensemble_probs = {
        k: round(vit_w * vit_probs.get(k, 0.0) + clip_w * clip_probs.get(k, 0.0), 4)
        for k in all_keys
    }

    predicted_category = max(ensemble_probs, key=ensemble_probs.get)
    confidence = round(ensemble_probs[predicted_category], 4)

    xai_dict = _run_xai(
        registry=registry,
        pil_image=pil_image,
        vit_probs=vit_probs,
        clip_probs=clip_probs,
        ensemble_probs=ensemble_probs,
        predicted_category=predicted_category,
        confidence=confidence,
        include_heatmap=include_heatmap,
        vit_weight=vit_w,
        clip_weight=clip_w,
    )

    return XAIExplanation(**xai_dict)
