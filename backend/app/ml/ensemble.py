"""
Ensemble prediction service.

Combines ViT and CLIP probability distributions using configurable weights.
Default: 70% ViT + 30% CLIP.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

from app.core.config import settings
from app.ml.calibration import TemperatureCalibrator, detect_out_of_distribution

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    predicted_category: str
    confidence: float
    class_probabilities: Dict[str, float]
    vit_prediction: str
    clip_prediction: str
    is_low_confidence: bool
    is_ambiguous: bool
    confidence_status: str = "normal"
    calibrated_confidence: float = 0.0
    calibrated_probabilities: Optional[Dict[str, float]] = None
    is_out_of_distribution: bool = False
    ood_warning: Optional[str] = None
    vit_probabilities: Optional[Dict[str, float]] = None
    clip_probabilities: Optional[Dict[str, float]] = None


# Threshold below which a prediction is considered low-confidence (default: 0.40)
LOW_CONFIDENCE_THRESHOLD = float(getattr(settings, "LOW_CONFIDENCE_THRESHOLD", 0.40))

# Difference between top-2 probabilities below which a prediction is ambiguous.
AMBIGUITY_THRESHOLD = 0.10


def _validate_and_normalize_probs(probs: Dict[str, float], model_name: str) -> Dict[str, float]:
    """Validate probability dictionary, ensure all classes exist, and sum to 1.0."""
    if not isinstance(probs, dict) or not probs:
        raise ValueError(f"{model_name} probabilities must be a non-empty dictionary.")

    classes = settings.WASTE_CLASSES
    normalized: Dict[str, float] = {}

    for cls in classes:
        val = probs.get(cls, 0.0)
        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val) or val < 0.0:
            raise ValueError(f"Invalid probability value for '{cls}' in {model_name}: {val}")
        normalized[cls] = float(val)

    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError(f"{model_name} probability sum must be positive, got {total}.")

    return {k: round(v / total, 6) for k, v in normalized.items()}


def fuse(
    vit_probs: Dict[str, float],
    clip_probs: Dict[str, float],
    vit_weight: float | None = None,
    clip_weight: float | None = None,
    temperature: float | None = None,
) -> EnsembleResult:
    """
    Fuse ViT and CLIP probability distributions via weighted average.
    Applies validation-fitted temperature scaling and checks for Out-of-Distribution items.

    Args:
        vit_probs:  {class: probability} from ViT softmax.
        clip_probs: {class: probability} from CLIP softmax.
        vit_weight: weight for ViT (defaults to settings.ENSEMBLE_VIT_WEIGHT).
        clip_weight: weight for CLIP (defaults to settings.ENSEMBLE_CLIP_WEIGHT).
        temperature: optional temperature scalar for probability calibration.

    Returns:
        EnsembleResult with raw and calibrated probabilities, predicted class, and diagnostic flags.
    """
    v_norm = _validate_and_normalize_probs(vit_probs, "ViT")
    c_norm = _validate_and_normalize_probs(clip_probs, "CLIP")

    w_vit = vit_weight if vit_weight is not None else settings.ENSEMBLE_VIT_WEIGHT
    w_clip = clip_weight if clip_weight is not None else settings.ENSEMBLE_CLIP_WEIGHT

    # Normalise weights in case they don't sum to 1.0
    total_w = w_vit + w_clip
    if total_w <= 0:
        w_vit, w_clip = 0.70, 0.30
        total_w = 1.0
    w_vit /= total_w
    w_clip /= total_w

    classes = settings.WASTE_CLASSES
    ensemble_probs: Dict[str, float] = {}
    for cls in classes:
        p_vit = v_norm[cls]
        p_clip = c_norm[cls]
        ensemble_probs[cls] = round(w_vit * p_vit + w_clip * p_clip, 6)

    # Re-normalise to handle floating-point drift
    total = sum(ensemble_probs.values())
    if total > 0:
        ensemble_probs = {k: round(v / total, 6) for k, v in ensemble_probs.items()}

    sorted_probs = sorted(ensemble_probs.items(), key=lambda x: x[1], reverse=True)
    best_class, best_prob = sorted_probs[0]
    second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0

    vit_best = max(v_norm, key=v_norm.get)  # type: ignore[arg-type]
    clip_best = max(c_norm, key=c_norm.get)  # type: ignore[arg-type]

    is_low = best_prob < LOW_CONFIDENCE_THRESHOLD
    is_ambiguous = (best_prob - second_prob) < AMBIGUITY_THRESHOLD

    # Confidence Calibration via Temperature Scaling
    cal_temp = temperature if temperature is not None else float(getattr(settings, "TEMPERATURE_SCALING", 0.35))
    calibrator = TemperatureCalibrator(temperature=cal_temp)
    cal_probs = calibrator.calibrate(ensemble_probs)
    cal_sorted = sorted(cal_probs.items(), key=lambda x: x[1], reverse=True)
    cal_conf = cal_sorted[0][1]

    # Out-of-Distribution / Unknown Item Detection (evaluated on calibrated distribution)
    is_ood, ood_warning, ood_diag = detect_out_of_distribution(
        cal_probs, vit_pred=vit_best, clip_pred=clip_best
    )

    confidence_status = "normal"
    if is_low:
        confidence_status = "low_confidence"
        logger.warning(
            "Low-confidence prediction: %.2f%% for '%s' (< %.2f%% threshold).",
            best_prob * 100,
            best_class,
            LOW_CONFIDENCE_THRESHOLD * 100,
        )
    elif is_ambiguous:
        confidence_status = "ambiguous"
        logger.warning(
            "Ambiguous prediction: top-2 gap = %.2f%% (< %.2f%% threshold).",
            (best_prob - second_prob) * 100,
            AMBIGUITY_THRESHOLD * 100,
        )
    elif is_ood:
        confidence_status = "out_of_distribution"
        logger.warning("Out-of-distribution item flagged: %s", ood_diag)

    return EnsembleResult(
        predicted_category=best_class,
        confidence=best_prob,
        class_probabilities=ensemble_probs,
        vit_prediction=vit_best,
        clip_prediction=clip_best,
        is_low_confidence=is_low,
        is_ambiguous=is_ambiguous,
        confidence_status=confidence_status,
        calibrated_confidence=cal_conf,
        calibrated_probabilities=cal_probs,
        is_out_of_distribution=is_ood,
        ood_warning=ood_warning,
        vit_probabilities=v_norm,
        clip_probabilities=c_norm,
    )
