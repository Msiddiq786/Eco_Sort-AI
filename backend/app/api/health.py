"""
GET /api/health  — liveness probe.
GET /api/model-info — model metadata.
"""
import json
from pathlib import Path
from fastapi import APIRouter, Depends

from app.core.dependencies import get_model_registry
from app.core.config import settings
from app.schemas.predict import ModelInfoResponse

router = APIRouter()

_CALIB_PATH = Path(__file__).resolve().parent.parent.parent / "evaluation" / "results" / "ensemble_calibration.json"


@router.get("/health")
async def health_check(registry=Depends(get_model_registry)):
    return {
        "status": "ok",
        "vit_loaded": registry.vit_loaded,
        "clip_loaded": registry.clip_loaded,
        "models_ready": registry.ready,
    }


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info(registry=Depends(get_model_registry)):
    temp = 0.35
    uncal_ece = 0.3897
    cal_ece = 0.0666

    if _CALIB_PATH.exists():
        try:
            data = json.loads(_CALIB_PATH.read_text(encoding="utf-8"))
            cal_info = data.get("calibration", {})
            temp = float(cal_info.get("temperature", temp))
            uncal_ece = float(cal_info.get("uncalibrated_ece", uncal_ece))
            cal_ece = float(cal_info.get("calibrated_ece", cal_ece))
        except Exception:
            pass

    return ModelInfoResponse(
        vit_checkpoint=settings.VIT_CHECKPOINT,
        clip_checkpoint=settings.CLIP_CHECKPOINT,
        vit_model_name=settings.VIT_MODEL_NAME,
        clip_model_name=settings.CLIP_MODEL_NAME,
        clip_enabled=settings.CLIP_ENABLED,
        ensemble_vit_weight=settings.ENSEMBLE_VIT_WEIGHT,
        ensemble_clip_weight=settings.ENSEMBLE_CLIP_WEIGHT,
        temperature=temp,
        uncalibrated_ece=uncal_ece,
        calibrated_ece=cal_ece,
        waste_classes=settings.WASTE_CLASSES,
        model_version=settings.MODEL_VERSION,
        vit_loaded=registry.vit_loaded,
        clip_loaded=registry.clip_loaded,
    )


@router.get("/diagnostics")
async def model_diagnostics(registry=Depends(get_model_registry)):
    """
    Diagnostic probe inspecting currently loaded ML models, checkpoint paths,
    active taxonomy version, and running V2 training status.
    """
    root_dir = Path(__file__).resolve().parent.parent.parent
    v1_best_meta = root_dir / "models" / "vit" / "best" / "training_metadata.json"
    v2_best_meta = root_dir / "models" / "vit_v2" / "best" / "training_metadata.json"
    v2_latest_meta = root_dir / "models" / "vit_v2" / "latest" / "training_metadata.json"

    vit_num_labels = 6
    vit_id2label = {}
    if hasattr(registry.vit_model, "config"):
        vit_num_labels = getattr(registry.vit_model.config, "num_labels", 6)
        vit_id2label = getattr(registry.vit_model.config, "id2label", {})

    vit_version = "V2 (7-Class)" if vit_num_labels == 7 else "V1 (6-Class)"

    v1_info = {}
    if v1_best_meta.exists():
        try:
            v1_info = json.loads(v1_best_meta.read_text(encoding="utf-8"))
        except Exception:
            pass

    v2_info = {}
    if v2_best_meta.exists():
        try:
            v2_info = json.loads(v2_best_meta.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "status": "ok",
        "active_inference_state": {
            "vit_model_version": vit_version,
            "vit_num_classes": vit_num_labels,
            "vit_classes": list(vit_id2label.values()) if vit_id2label else settings.WASTE_CLASSES,
            "vit_active_checkpoint": "backend/models/vit/best" if vit_num_labels == 6 else "backend/models/vit_v2/best",
            "clip_model_version": "CLIP Foundation (V2 7-Class)",
            "clip_scoring_method": "class_centroid",
            "clip_classes": ["cardboard", "glass", "metal", "paper", "plastic", "organic", "trash"],
            "organic_supported_by_vit": (vit_num_labels >= 7),
            "organic_supported_by_clip": True,
        },
        "ensemble_configuration": {
            "vit_weight": settings.ENSEMBLE_VIT_WEIGHT,
            "clip_weight": settings.ENSEMBLE_CLIP_WEIGHT,
            "weights_sum": settings.ENSEMBLE_VIT_WEIGHT + settings.ENSEMBLE_CLIP_WEIGHT,
            "disagreement_clamping": {
                "active": True,
                "cap_confidence": 0.68,
                "reason": "When ViT and CLIP top categories disagree, confidence is capped at 0.68 to route to Inspection Required.",
            },
        },
        "checkpoints_on_disk": {
            "v1_best": {
                "path": "backend/models/vit/best",
                "num_classes": v1_info.get("num_classes", 6),
                "best_epoch": v1_info.get("best_epoch"),
                "best_val_accuracy": v1_info.get("best_val_accuracy"),
                "best_val_macro_f1": v1_info.get("best_val_macro_f1"),
                "status": "COMPLETED",
            },
            "v2_in_training": {
                "path": "backend/models/vit_v2/best",
                "num_classes": v2_info.get("num_classes", 7),
                "best_epoch": v2_info.get("best_epoch"),
                "best_val_accuracy": v2_info.get("best_val_accuracy"),
                "best_val_macro_f1": v2_info.get("best_val_macro_f1"),
                "status": "TRAINING_ACTIVE" if v2_best_meta.exists() else "NOT_STARTED",
            },
        },
    }

