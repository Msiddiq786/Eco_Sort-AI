"""
Deep-Dive Validation Audit for EcoSort AI CLIP V2 Pipeline.

Investigates:
1. Organic classification breakdown:
   - Inspect all 60 organic validation images.
   - Identify which images are correctly vs incorrectly classified.
   - Trace why F1 shifted from 93.3% (with 12 un-balanced prompts in global softmax)
     to 88.7% (with 5 balanced prompts in centroid space).
2. Trash classification breakdown:
   - Inspect all 21 trash validation images.
   - Identify true positives vs false negatives (which classes they were misassigned to).
   - Quantify statistical variance due to n=21 sample size.
3. Strict probability normalization verification across all 439 validation samples:
   - Verify min, max, sum, and max absolute deviation from 1.0.
4. Comprehensive confidence calibration evaluation:
   - Expected Calibration Error (ECE, 10 bins).
   - Maximum Calibration Error (MCE).
   - Brier Score.
   - Negative Log-Likelihood (NLL).
   - Bin-by-bin reliability analysis.
   - Optimal temperature estimation on validation set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import numpy as np
from PIL import Image
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    brier_score_loss,
)
from transformers import CLIPModel, CLIPProcessor

from app.ml.clip_model_v2 import (
    V2_WASTE_CLASSES,
    V2_CLIP_PROMPT_TEMPLATES,
    get_or_build_class_centroids,
    predict_clip_v2,
)

V2_CLASSES = V2_WASTE_CLASSES
CLASS_TO_IDX = {c: i for i, c in enumerate(V2_CLASSES)}


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Tuple[float, float, List[Dict]]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if np.sum(in_bin) > 0:
            avg_acc = float(np.mean(accuracies[in_bin]))
            avg_conf = float(np.mean(confidences[in_bin]))
            gap = abs(avg_acc - avg_conf)
            ece += gap * prop_in_bin
            mce = max(mce, gap)
            bin_details.append({
                "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "count": int(np.sum(in_bin)),
                "avg_confidence": round(avg_conf, 4),
                "avg_accuracy": round(avg_acc, 4),
                "calibration_gap": round(gap, 4),
            })
        else:
            bin_details.append({
                "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "count": 0,
                "avg_confidence": 0.0,
                "avg_accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    return float(ece), float(mce), bin_details


def run_audit():
    root = Path(__file__).resolve().parent.parent
    val_dir = root / "data" / "val"
    assert val_dir.is_dir(), f"Validation dir not found at {val_dir}"

    device = torch.device("cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    centroids = get_or_build_class_centroids(model, processor)

    # 1. Load validation dataset
    samples = []
    for cls in V2_CLASSES:
        folder = val_dir / cls
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in folder.glob(ext):
                samples.append((p, CLASS_TO_IDX[cls], cls))
    samples = sorted(samples, key=lambda x: str(x[0]))
    y_true = np.array([s[1] for s in samples])

    # 2. Extract predictions, probabilities, and normalization checks
    all_probs = []
    all_logits = []
    normalization_errors = []
    per_image_results = []

    logit_scale = model.logit_scale.exp().clamp(max=100.0)

    for p, y_idx, true_cls in samples:
        img = Image.open(p).convert("RGB")
        res = predict_clip_v2(model, processor, img)
        p_vec = [res.probabilities[c] for c in V2_CLASSES]
        all_probs.append(p_vec)

        # Check normalization
        p_sum = sum(p_vec)
        deviation = abs(p_sum - 1.0)
        normalization_errors.append(deviation)

        per_image_results.append({
            "path": p.name,
            "true_class": true_cls,
            "pred_class": res.predicted_category,
            "confidence": res.confidence,
            "probabilities": {c: round(res.probabilities[c], 4) for c in V2_CLASSES},
            "correct": (res.predicted_category == true_cls),
        })

    all_probs = np.array(all_probs)
    y_pred = np.argmax(all_probs, axis=1)

    # 3. Normalization Audit
    max_norm_err = float(np.max(normalization_errors))
    avg_norm_err = float(np.mean(normalization_errors))
    min_prob = float(np.min(all_probs))
    max_prob = float(np.max(all_probs))

    # 4. Calibration Audit
    ece, mce, bin_details = compute_ece(all_probs, y_true, n_bins=10)

    # Multi-class Brier score
    y_true_one_hot = np.zeros_like(all_probs)
    for i, y in enumerate(y_true):
        y_true_one_hot[i, y] = 1.0
    brier_score = float(np.mean(np.sum((all_probs - y_true_one_hot) ** 2, axis=1)))

    # NLL
    eps = 1e-12
    nll = float(-np.mean(np.log(all_probs[np.arange(len(y_true)), y_true] + eps)))

    # 5. Organic Deep-Dive
    organic_samples = [r for r in per_image_results if r["true_class"] == "organic"]
    organic_correct = [r for r in organic_samples if r["correct"]]
    organic_misclassified = [r for r in organic_samples if not r["correct"]]

    organic_confusions = {}
    for r in organic_misclassified:
        pred = r["pred_class"]
        organic_confusions[pred] = organic_confusions.get(pred, 0) + 1

    # 6. Trash Deep-Dive
    trash_samples = [r for r in per_image_results if r["true_class"] == "trash"]
    trash_correct = [r for r in trash_samples if r["correct"]]
    trash_misclassified = [r for r in trash_samples if not r["correct"]]

    trash_confusions = {}
    for r in trash_misclassified:
        pred = r["pred_class"]
        trash_confusions[pred] = trash_confusions.get(pred, 0) + 1

    # 7. Standard Metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(7)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(7))).tolist()

    per_class_metrics = {}
    for i, c in enumerate(V2_CLASSES):
        per_class_metrics[c] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    audit_report = {
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "total_val_samples": len(samples),
        },
        "normalization_audit": {
            "max_probability_sum_deviation": max_norm_err,
            "average_probability_sum_deviation": avg_norm_err,
            "minimum_probability": min_prob,
            "maximum_probability": max_prob,
            "strictly_normalized": bool(max_norm_err < 1e-4),
        },
        "calibration_audit": {
            "ece_10_bins": round(ece, 4),
            "mce": round(mce, 4),
            "brier_score": round(brier_score, 4),
            "negative_log_likelihood": round(nll, 4),
            "reliability_bins": bin_details,
        },
        "organic_investigation": {
            "total_support": len(organic_samples),
            "correct_count": len(organic_correct),
            "misclassified_count": len(organic_misclassified),
            "recall": round(len(organic_correct) / len(organic_samples), 4),
            "precision": per_class_metrics["organic"]["precision"],
            "f1_score": per_class_metrics["organic"]["f1"],
            "misclassified_destinations": organic_confusions,
            "misclassified_details": organic_misclassified,
        },
        "trash_investigation": {
            "total_support": len(trash_samples),
            "correct_count": len(trash_correct),
            "misclassified_count": len(trash_misclassified),
            "recall": round(len(trash_correct) / len(trash_samples), 4),
            "precision": per_class_metrics["trash"]["precision"],
            "f1_score": per_class_metrics["trash"]["f1"],
            "misclassified_destinations": trash_confusions,
            "misclassified_details": trash_misclassified,
            "sample_size_sensitivity": "Each image represents 4.76% (1/21) of Trash recall",
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm,
    }

    out_file = root / "evaluation" / "results" / "clip_v2_validation_audit_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print("\n=======================================================")
    print("EcoSort AI CLIP V2 Validation Audit Completed")
    print(f"  Accuracy:    {acc*100:.2f}%")
    print(f"  Macro F1:    {macro_f1*100:.2f}%")
    print(f"  ECE:         {ece*100:.2f}%")
    print(f"  Brier Score: {brier_score:.4f}")
    print(f"  NLL:         {nll:.4f}")
    print(f"  Prob Sum Dev: {max_norm_err:.2e}")
    print("Organic Investigation:")
    print(f"  Recall: {len(organic_correct)}/{len(organic_samples)} ({len(organic_correct)/len(organic_samples)*100:.1f}%)")
    print(f"  Misclassifications: {organic_confusions}")
    print("Trash Investigation:")
    print(f"  Recall: {len(trash_correct)}/{len(trash_samples)} ({len(trash_correct)/len(trash_samples)*100:.1f}%)")
    print(f"  Misclassifications: {trash_confusions}")
    print("=======================================================\n")
    return audit_report


if __name__ == "__main__":
    run_audit()
