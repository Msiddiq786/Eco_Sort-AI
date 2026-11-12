"""
Confidence Calibration & Out-of-Distribution (OOD) Detection for EcoSort AI.

Implements:
  1. Temperature Scaling (Guo et al., ICML 2017)
  2. Expected Calibration Error (ECE) measurement
  3. Entropy-based & margin-based Out-of-Distribution / Unknown waste detection
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Max theoretical entropy for 6 classes: ln(6) ~= 1.791759
MAX_6_CLASS_ENTROPY = math.log(6.0)

# Thresholds for OOD / Unknown item detection (applied on calibrated distribution)
OOD_CONFIDENCE_THRESHOLD = 0.45
OOD_ENTROPY_THRESHOLD = 1.35  # ~75% of max entropy (1.7917)
OOD_MARGIN_THRESHOLD = 0.15   # Top-1 vs Top-2 gap


class TemperatureCalibrator:
    """
    Post-hoc temperature scaling to calibrate ensemble probability distributions.
    Learned strictly on validation data to minimize Negative Log-Likelihood (NLL).
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(0.1, float(temperature))

    def fit(
        self,
        val_probs: Sequence[Sequence[float]],
        val_targets: Sequence[int],
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Fit scalar temperature T on validation probability predictions.
        Uses grid search + gradient refinement to find optimal T that minimizes NLL.
        """
        if len(val_probs) == 0:
            return self.temperature

        probs_arr = np.clip(np.array(val_probs, dtype=np.float64), 1e-7, 1.0)
        # Normalize rows
        probs_arr = probs_arr / probs_arr.sum(axis=-1, keepdims=True)
        # Convert to unscaled log-odds
        logits = np.log(probs_arr)
        targets = np.array(val_targets, dtype=np.int64)

        best_t = 1.0
        best_nll = float("inf")

        # Grid search over reasonable range [0.3, 3.0]
        for t_candidate in np.linspace(0.3, 3.0, 55):
            scaled_logits = logits / t_candidate
            # Stable log-sum-exp
            max_l = np.max(scaled_logits, axis=-1, keepdims=True)
            log_sum_exp = max_l + np.log(np.sum(np.exp(scaled_logits - max_l), axis=-1, keepdims=True))
            log_probs = scaled_logits - log_sum_exp
            nll = -float(np.mean(log_probs[np.arange(len(targets)), targets]))

            if nll < best_nll:
                best_nll = nll
                best_t = float(t_candidate)

        self.temperature = round(best_t, 4)
        logger.info("Fitted temperature scaling: T = %.4f (Val NLL = %.4f)", self.temperature, best_nll)
        return self.temperature

    def calibrate(self, probs: Mapping[str, float]) -> Dict[str, float]:
        """
        Scale probabilities by learned temperature T and re-apply softmax.
        When T < 1.0, under-confident models are sharpened.
        When T > 1.0, over-confident models are softened.
        """
        if abs(self.temperature - 1.0) < 1e-4:
            return {k: round(float(v), 6) for k, v in probs.items()}

        keys = list(probs.keys())
        raw_vals = np.array([max(1e-7, float(probs[k])) for k in keys], dtype=np.float64)
        raw_vals = raw_vals / raw_vals.sum()

        logits = np.log(raw_vals) / self.temperature
        # Numerically stable softmax
        logits -= np.max(logits)
        exp_l = np.exp(logits)
        calibrated_vals = exp_l / np.sum(exp_l)

        return {k: round(float(v), 6) for k, v in zip(keys, calibrated_vals)}


def compute_entropy(probs: Mapping[str, float] | Sequence[float]) -> float:
    """Compute Shannon entropy H(p) = -sum(p * log(p))."""
    if isinstance(probs, dict):
        vals = [max(1e-7, float(v)) for v in probs.values()]
    else:
        vals = [max(1e-7, float(v)) for v in probs]
    total = sum(vals)
    norm = [v / total for v in vals]
    return -sum(p * math.log(p) for p in norm)


def compute_ece(
    probs: Sequence[Sequence[float]],
    targets: Sequence[int],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width confidence bins.
    """
    if len(probs) == 0:
        return 0.0

    probs_arr = np.array(probs, dtype=np.float64)
    confidences = np.max(probs_arr, axis=-1)
    predictions = np.argmax(probs_arr, axis=-1)
    accuracies = (predictions == np.array(targets)).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = float(len(targets))

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_total) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


def detect_out_of_distribution(
    probabilities: Mapping[str, float],
    vit_pred: str | None = None,
    clip_pred: str | None = None,
    conf_threshold: float = OOD_CONFIDENCE_THRESHOLD,
    entropy_threshold: float = OOD_ENTROPY_THRESHOLD,
    margin_threshold: float = OOD_MARGIN_THRESHOLD,
) -> Tuple[bool, str | None, Dict[str, Any]]:
    """
    Detect whether an input image is likely out-of-distribution (OOD) or unknown
    (e.g., vegetables, food waste, electronics/batteries, or unidentifiable clutter).

    Returns:
        (is_ood, warning_message, diagnostics_dict)
    """
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top1_class, top1_prob = sorted_items[0]
    top2_prob = sorted_items[1][1] if len(sorted_items) > 1 else 0.0

    margin = top1_prob - top2_prob
    entropy = compute_entropy(probabilities)
    models_disagree = (vit_pred is not None and clip_pred is not None and vit_pred != clip_pred)

    is_low_conf = top1_prob < conf_threshold
    is_high_entropy = entropy > entropy_threshold
    is_narrow_margin = margin < margin_threshold

    # OOD condition: either low confidence, high entropy, or severe disagreement with narrow margin
    is_ood = is_low_conf or (is_high_entropy and (models_disagree or is_narrow_margin))

    warning_msg: str | None = None
    if is_ood:
        warning_msg = (
            "Caution: Possible unsupported or out-of-distribution item detected. "
            "The image may contain organic food waste, hazardous material, or mixed refuse "
            "outside the six standard recycling categories. Manual inspection recommended."
        )

    diagnostics = {
        "entropy": round(entropy, 4),
        "normalized_entropy": round(entropy / MAX_6_CLASS_ENTROPY, 4),
        "top1_prob": round(top1_prob, 4),
        "top2_prob": round(top2_prob, 4),
        "margin": round(margin, 4),
        "models_disagree": models_disagree,
        "is_ood": is_ood,
    }

    return is_ood, warning_msg, diagnostics
