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
from app.ml.clip_model_v2 import V2_WASTE_CLASSES, predict_clip_v2
from app.ml.ensemble import fuse
from app.ml.preprocessor import load_image_pil
from app.ml.vit_model import predict_vit
from app.schemas.predict import (
    ClassProbabilities,
    CLIPPredictResponse,
    ModelCoverage,
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

    # ── 6. ViT inference (6-class V1 or 7-class V2) ───────────────────────────
    vit_num_classes = 6
    if hasattr(registry.vit_model, "config") and hasattr(registry.vit_model.config, "num_labels"):
        val = getattr(registry.vit_model.config, "num_labels", 6)
        if isinstance(val, int):
            vit_num_classes = val

    try:
        vit_probs = predict_vit(registry.vit_model, registry.vit_processor, pil_image)
    except Exception as exc:
        logger.exception("ViT inference failed")
        raise HTTPException(status_code=500, detail=f"ViT inference error: {exc}")

    # ── 7. CLIP V2 inference (7-class aware) ──────────────────────────────────
    clip_probs = {}
    clip_pred = ""
    if registry.clip_loaded:
        try:
            clip_v2_res = predict_clip_v2(registry.clip_model, registry.clip_processor, pil_image)
            clip_probs = clip_v2_res.probabilities
            clip_pred = clip_v2_res.predicted_category
        except Exception as exc:
            logger.warning("CLIP V2 inference failed: %s, falling back to 6-class: %s", exc)
            try:
                from app.ml.clip_model import predict_clip
                c_res = predict_clip(registry.clip_model, registry.clip_processor, pil_image)
                clip_probs = c_res.probabilities if hasattr(c_res, "probabilities") else c_res
                clip_pred = c_res.predicted_category if hasattr(c_res, "predicted_category") else "trash"
            except Exception:
                clip_probs = {c: vit_probs.get(c, 0.0) for c in V2_WASTE_CLASSES}
                clip_pred = max(vit_probs, key=vit_probs.get)
    else:
        clip_probs = {c: vit_probs.get(c, 0.0) for c in V2_WASTE_CLASSES}
        clip_pred = max(vit_probs, key=vit_probs.get)

    # ── 8. Dynamic Model Coverage, Disagreement, & Organic Policy ────────────
    vit_top = max(vit_probs, key=vit_probs.get)
    vit_top_prob = vit_probs[vit_top]

    clip_organic_prob = clip_probs.get("organic", 0.0)
    second_clip_prob = max([v for k, v in clip_probs.items() if k != "organic"], default=0.0)
    clip_organic_margin = clip_organic_prob - second_clip_prob

    # Strict 5-point Organic Bypass Policy
    is_organic_v1_bypass = (
        vit_num_classes == 6
        and clip_pred == "organic"
        and clip_organic_prob >= 0.40
        and clip_organic_margin >= 0.10
    )

    if is_organic_v1_bypass:
        final_cat = "organic"
        final_conf = min(0.985, float(clip_organic_prob))
        conf_status = "normal" if final_conf >= 0.70 else "ambiguous"
        is_ood = False
        models_agree = False
        vit_prediction_label = "Not supported (V1 6-class)"
        clip_prediction_label = "organic"
        compat_decision = "organic_v1_clip_bypass"
        coverage_type = "partial"
        vit_supported = False
        clip_supported = True
        ensemble_supported = False
        unsupported_reason = "ViT V1 is trained on 6 classes and does not support Organic waste."

        class_probs = {
            "cardboard": round(vit_probs.get("cardboard", 0.0) * 0.7 + clip_probs.get("cardboard", 0.0) * 0.3, 4),
            "glass": round(vit_probs.get("glass", 0.0) * 0.7 + clip_probs.get("glass", 0.0) * 0.3, 4),
            "metal": round(vit_probs.get("metal", 0.0) * 0.7 + clip_probs.get("metal", 0.0) * 0.3, 4),
            "paper": round(vit_probs.get("paper", 0.0) * 0.7 + clip_probs.get("paper", 0.0) * 0.3, 4),
            "plastic": round(vit_probs.get("plastic", 0.0) * 0.7 + clip_probs.get("plastic", 0.0) * 0.3, 4),
            "organic": round(clip_organic_prob, 4),
            "trash": round(vit_probs.get("trash", 0.0) * 0.35 + clip_probs.get("trash", 0.0) * 0.3, 4),
        }
        tot = sum(class_probs.values())
        class_probs = {k: round(v / tot, 4) for k, v in class_probs.items()}

    elif clip_pred == "organic" and vit_num_classes == 6:
        # Organic predicted but confidence or margin is below threshold -> requires verification
        final_cat = "unknown"
        final_conf = min(0.68, float(clip_organic_prob))
        conf_status = "low_confidence"
        is_ood = False
        models_agree = False
        vit_prediction_label = "Not supported (V1 6-class)"
        clip_prediction_label = "organic"
        compat_decision = "organic_v1_weak_confidence"
        coverage_type = "partial"
        vit_supported = False
        clip_supported = True
        ensemble_supported = False
        unsupported_reason = "ViT V1 lacks Organic taxonomy; CLIP organic confidence or margin is below threshold."

        class_probs = {
            "cardboard": round(vit_probs.get("cardboard", 0.0) * 0.7 + clip_probs.get("cardboard", 0.0) * 0.3, 4),
            "glass": round(vit_probs.get("glass", 0.0) * 0.7 + clip_probs.get("glass", 0.0) * 0.3, 4),
            "metal": round(vit_probs.get("metal", 0.0) * 0.7 + clip_probs.get("metal", 0.0) * 0.3, 4),
            "paper": round(vit_probs.get("paper", 0.0) * 0.7 + clip_probs.get("paper", 0.0) * 0.3, 4),
            "plastic": round(vit_probs.get("plastic", 0.0) * 0.7 + clip_probs.get("plastic", 0.0) * 0.3, 4),
            "organic": round(clip_organic_prob, 4),
            "trash": round(vit_probs.get("trash", 0.0) * 0.35 + clip_probs.get("trash", 0.0) * 0.3, 4),
        }
        tot = sum(class_probs.values())
        class_probs = {k: round(v / tot, 4) for k, v in class_probs.items()}

    elif vit_num_classes >= 7 and (clip_pred == "organic" or vit_top == "organic"):
        # Dual 7-class support
        fused = fuse(vit_probs, clip_probs)
        final_cat = fused.predicted_category
        final_conf = min(0.985, fused.confidence)
        vit_prediction_label = fused.vit_prediction
        clip_prediction_label = fused.clip_prediction
        models_agree = (vit_prediction_label.lower() == clip_prediction_label.lower())
        coverage_type = "full"
        vit_supported = True
        clip_supported = True
        ensemble_supported = True
        unsupported_reason = None
        compat_decision = "organic_v2_dual_consensus" if models_agree else "organic_v2_dual_disagreement"

        if not models_agree:
            conf_status = "ambiguous"
            final_conf = min(final_conf, 0.68)
        else:
            conf_status = fused.confidence_status
        is_ood = fused.is_out_of_distribution
        class_probs = {k: round(v, 4) for k, v in fused.class_probabilities.items()}

    else:
        # Standard 6-class shared evaluation
        fused = fuse(vit_probs, {k: clip_probs.get(k, 0.0) for k in vit_probs})
        final_cat = fused.predicted_category
        final_conf = min(0.985, fused.confidence)
        vit_prediction_label = fused.vit_prediction
        clip_prediction_label = clip_pred
        models_agree = (vit_prediction_label.lower() == clip_prediction_label.lower())
        coverage_type = "full"
        vit_supported = True
        clip_supported = True
        ensemble_supported = True
        unsupported_reason = None

        if not models_agree:
            conf_status = "ambiguous"
            final_conf = min(final_conf, 0.68)
            compat_decision = "model_disagreement"
        else:
            conf_status = fused.confidence_status
            compat_decision = "full_consensus"

        is_ood = fused.is_out_of_distribution
        class_probs = {k: round(v, 4) for k, v in fused.class_probabilities.items()}
        class_probs["organic"] = round(clip_organic_prob, 4)
        tot = sum(class_probs.values())
        class_probs = {k: round(v / tot, 4) for k, v in class_probs.items()}

    # ── 9. Recycling recommendation & intelligent decision engine ───────────
    recommendation = get_recommendation(final_cat)
    rec_decision = get_recycling_decision(
        category=final_cat,
        confidence=final_conf,
        is_ood=is_ood,
        confidence_status=conf_status,
        model_agreement=models_agree,
        is_organic_v1_bypass=is_organic_v1_bypass,
    )

    # ── 10. Persist record to database ───────────────────────────────────────
    record = await create_prediction(
        db,
        filename=safe_name,
        original_filename=original_filename,
        image_path=image_path,
        predicted_category=final_cat,
        confidence=final_conf,
        vit_prediction=vit_prediction_label,
        clip_prediction=clip_prediction_label,
        class_probabilities=class_probs,
        model_version=settings.MODEL_VERSION,
    )

    coverage_obj = ModelCoverage(
        vit_supported=vit_supported,
        clip_supported=clip_supported,
        ensemble_supported=ensemble_supported,
        unsupported_reason=unsupported_reason,
        compatibility_decision=compat_decision,
        coverage_type=coverage_type,
    )

    vit_prob_obj = ClassProbabilities(**vit_probs) if vit_supported else None
    clip_prob_obj = ClassProbabilities(**clip_probs) if clip_probs else None
    cal_prob_obj = ClassProbabilities(**class_probs)

    ood_warning = None
    if is_ood:
        ood_warning = "Caution: Possible unsupported or out-of-distribution item detected."
    elif not models_agree and not is_organic_v1_bypass:
        ood_warning = f"Caution: Model disagreement between {vit_prediction_label} and {clip_prediction_label}."

    return PredictResponse(
        id=record.id,
        filename=safe_name,
        original_filename=original_filename,
        image_path=image_path,
        predicted_category=final_cat,
        confidence=final_conf,
        calibrated_confidence=final_conf,
        vit_prediction=vit_prediction_label,
        clip_prediction=clip_prediction_label,
        class_probabilities=ClassProbabilities(**class_probs),
        calibrated_probabilities=cal_prob_obj,
        vit_probabilities=vit_prob_obj,
        clip_probabilities=clip_prob_obj,
        is_low_confidence=(final_conf < 0.40),
        is_ambiguous=(conf_status == "ambiguous"),
        is_out_of_distribution=is_ood,
        confidence_status=conf_status,
        ood_warning=ood_warning,
        recycling_recommendation=recommendation,
        recycling_decision=RecyclingDecisionSchema(**rec_decision.to_dict()),
        model_agreement=models_agree,
        model_coverage=coverage_obj,
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
