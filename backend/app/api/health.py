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
