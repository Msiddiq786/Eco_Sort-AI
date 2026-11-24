"""
EcoSort AI V2 — Final Test Split Evaluation (ViT=0.90 / CLIP=0.10).

Evaluates on the UNTOUCHED test split: backend/data/test (N=440).
Compares:
1. ViT V2 only (1.00 / 0.00)
2. CLIP V2 only (0.00 / 1.00)
3. Legacy 70/30 Ensemble (0.70 / 0.30) [for exact comparison]
4. Selected 90/10 Ensemble (0.90 / 0.10)

Calibration:
Temperature scaling parameter T is fitted strictly on validation data (backend/data/val)
to avoid any test label leakage. Raw probabilities are preserved.

Error Analysis:
- Most frequently confused classes
- Cardboard vs Paper errors
- Glass vs Plastic errors
- Organic vs Trash errors
- Trash classification performance

Saves report to:
backend/evaluation/results_v2/final_ensemble_90_10_test.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import CLIPModel, CLIPProcessor, ViTForImageClassification, ViTImageProcessor

from app.ml.clip_model_v2 import (
    V2_WASTE_CLASSES,
    predict_clip_v2,
)

V2_CLASSES = V2_WASTE_CLASSES
CLASS_TO_IDX = {c: i for i, c in enumerate(V2_CLASSES)}


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Tuple[float, float, List[Dict]]:
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
        in_bin = (
            (confidences > bin_lower) & (confidences <= bin_upper)
            if i > 0
            else (confidences >= bin_lower) & (confidences <= bin_upper)
        )
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
                "gap": round(gap, 4),
            })

    return float(ece), float(mce), bin_details


def fit_temperature_on_val(val_probs: np.ndarray, val_labels: np.ndarray) -> float:
    """
    Fit temperature scaling T strictly on validation probabilities.
    Uses bounded scalar minimization of NLL on validation set.
    """
    eps = 1e-12
    pseudo_logits = np.log(np.clip(val_probs, eps, 1.0))

    def nll_loss(T: float) -> float:
        scaled_logits = pseudo_logits / T
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        cal_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # NLL
        correct_probs = cal_probs[np.arange(len(val_labels)), val_labels]
        return float(-np.mean(np.log(np.clip(correct_probs, eps, 1.0))))

    res = minimize_scalar(nll_loss, bounds=(0.1, 5.0), method="bounded")
    optimal_T = float(res.x)
    return optimal_T


def apply_temperature_scaling(probs: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-12
    pseudo_logits = np.log(np.clip(probs, eps, 1.0))
    scaled_logits = pseudo_logits / T
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    cal_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return cal_probs


def load_dataset_split(split_dir: Path) -> Tuple[List[Tuple[Path, int, str]], np.ndarray]:
    samples = []
    for cls in V2_CLASSES:
        folder = split_dir / cls
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in folder.glob(ext):
                samples.append((p, CLASS_TO_IDX[cls], cls))
    samples = sorted(samples, key=lambda x: str(x[0]))
    labels = np.array([s[1] for s in samples])
    return samples, labels


def extract_split_probabilities(
    samples: List[Tuple[Path, int, str]],
    vit_model: Any,
    vit_processor: Any,
    clip_model: Any,
    clip_processor: Any,
    device: torch.device,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    vit_probs_list = []
    clip_probs_list = []
    num_samples = len(samples)

    start_time = time.time()
    for idx, (p, y_idx, true_cls) in enumerate(samples, start=1):
        img = Image.open(p).convert("RGB")

        # ViT Inference
        vit_inputs = vit_processor(images=img, return_tensors="pt")
        vit_inputs = {k: v.to(device) for k, v in vit_inputs.items()}
        with torch.no_grad():
            vit_logits = vit_model(**vit_inputs).logits
            v_p = torch.softmax(vit_logits, dim=-1)[0].cpu().numpy()
        vit_probs_list.append(v_p)

        # CLIP Inference (Class-Centroid scoring)
        c_res = predict_clip_v2(clip_model, clip_processor, img)
        c_p = np.array([c_res.probabilities[c] for c in V2_CLASSES])
        clip_probs_list.append(c_p)

        if idx % 50 == 0 or idx == num_samples:
            elapsed = time.time() - start_time
            print(f"  [{split_name}] Processed {idx}/{num_samples} images ({elapsed:.1f}s, {idx/elapsed:.2f} img/s)")

    total_time = time.time() - start_time
    return np.array(vit_probs_list), np.array(clip_probs_list), total_time


def analyze_configuration(
    name: str,
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    val_fitted_T: float,
    total_time: float,
) -> Dict[str, Any]:
    preds = np.argmax(raw_probs, axis=1)
    num_samples = len(y_true)
    correct_count = int(np.sum(preds == y_true))
    incorrect_count = num_samples - correct_count

    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, preds, labels=list(range(7)), zero_division=0
    )
    cm = confusion_matrix(y_true, preds, labels=list(range(7)))

    # Raw calibration metrics
    ece_raw, mce_raw, bins_raw = compute_ece(raw_probs, y_true, n_bins=10)

    # Calibrated probabilities (using validation-fitted T)
    cal_probs = apply_temperature_scaling(raw_probs, val_fitted_T)
    ece_cal, mce_cal, bins_cal = compute_ece(cal_probs, y_true, n_bins=10)

    # Multi-class Brier score on raw probabilities
    y_one_hot = np.zeros_like(raw_probs)
    for i, y in enumerate(y_true):
        y_one_hot[i, y] = 1.0
    brier_raw = float(np.mean(np.sum((raw_probs - y_one_hot) ** 2, axis=1)))
    brier_cal = float(np.mean(np.sum((cal_probs - y_one_hot) ** 2, axis=1)))

    per_class = {
        V2_CLASSES[i]: {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }
        for i in range(7)
    }

    # Error analysis specific pairs
    c_card = CLASS_TO_IDX["cardboard"]
    c_pap = CLASS_TO_IDX["paper"]
    c_gla = CLASS_TO_IDX["glass"]
    c_plas = CLASS_TO_IDX["plastic"]
    c_org = CLASS_TO_IDX["organic"]
    c_trash = CLASS_TO_IDX["trash"]

    # Most frequently confused pairs (off-diagonal)
    confused_pairs = []
    for r in range(7):
        for c in range(7):
            if r != c and cm[r, c] > 0:
                confused_pairs.append({
                    "true_class": V2_CLASSES[r],
                    "predicted_class": V2_CLASSES[c],
                    "count": int(cm[r, c]),
                })
    confused_pairs = sorted(confused_pairs, key=lambda x: x["count"], reverse=True)

    return {
        "name": name,
        "sample_count": num_samples,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "calibration": {
            "uncalibrated_ece": round(ece_raw, 4),
            "uncalibrated_mce": round(mce_raw, 4),
            "uncalibrated_brier_score": round(brier_raw, 4),
            "validation_fitted_temperature": round(val_fitted_T, 4),
            "calibrated_ece": round(ece_cal, 4),
            "calibrated_mce": round(mce_cal, 4),
            "calibrated_brier_score": round(brier_cal, 4),
            "calibration_method": "Temperature Scaling (T fitted on validation set NLL)",
        },
        "performance_speed": {
            "total_inference_time_seconds": round(total_time, 2),
            "average_time_per_image_seconds": round(total_time / num_samples, 4),
            "samples_per_second": round(num_samples / total_time, 2),
        },
        "error_analysis": {
            "cardboard_pred_paper": int(cm[c_card, c_pap]),
            "paper_pred_cardboard": int(cm[c_pap, c_card]),
            "glass_pred_plastic": int(cm[c_gla, c_plas]),
            "plastic_pred_glass": int(cm[c_plas, c_gla]),
            "organic_pred_trash": int(cm[c_org, c_trash]),
            "trash_pred_organic": int(cm[c_trash, c_org]),
            "trash_performance": {
                "precision": round(float(prec[c_trash]), 4),
                "recall": round(float(rec[c_trash]), 4),
                "f1": round(float(f1[c_trash]), 4),
                "support": int(sup[c_trash]),
                "correct_count": int(cm[c_trash, c_trash]),
                "misclassifications": {
                    V2_CLASSES[c]: int(cm[c_trash, c]) for c in range(7) if c != c_trash and cm[c_trash, c] > 0
                },
            },
            "top_confused_pairs": confused_pairs[:6],
        },
    }


def run_evaluation():
    root = _BACKEND_DIR
    val_dir = root / "data" / "val"
    test_dir = root / "data" / "test"
    v2_best_dir = root / "models" / "vit_v2" / "best"
    out_file = root / "evaluation" / "results_v2" / "final_ensemble_90_10_test.json"

    device = torch.device("cpu")

    print("=========================================================================")
    print("EcoSort AI V2 — Final Test Split Evaluation (ViT=0.90 / CLIP=0.10)")
    print("=========================================================================")
    print(f"ViT Checkpoint: {v2_best_dir.resolve()}")
    print(f"Test Split:     {test_dir.resolve()}")
    print("=========================================================================\n")

    # Load models
    vit_model = ViTForImageClassification.from_pretrained(str(v2_best_dir)).to(device)
    vit_processor = ViTImageProcessor.from_pretrained(str(v2_best_dir))
    vit_model.eval()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # Step 1: Fit temperature strictly on validation set (NO test labels used)
    print("Step 1: Extracting validation set probabilities to fit calibration temperature...")
    val_samples, val_labels = load_dataset_split(val_dir)
    v_val_vit, v_val_clip, _ = extract_split_probabilities(
        val_samples, vit_model, vit_processor, clip_model, clip_processor, device, "Validation"
    )

    # 90/10 ensemble on validation
    ens_val_90_10 = 0.90 * v_val_vit + 0.10 * v_val_clip
    ens_val_90_10 = ens_val_90_10 / np.sum(ens_val_90_10, axis=1, keepdims=True)
    T_val_90_10 = fit_temperature_on_val(ens_val_90_10, val_labels)

    ens_val_70_30 = 0.70 * v_val_vit + 0.30 * v_val_clip
    ens_val_70_30 = ens_val_70_30 / np.sum(ens_val_70_30, axis=1, keepdims=True)
    T_val_70_30 = fit_temperature_on_val(ens_val_70_30, val_labels)

    T_val_vit = fit_temperature_on_val(v_val_vit, val_labels)
    T_val_clip = fit_temperature_on_val(v_val_clip, val_labels)

    print(f"  Fitted Validation Temperatures: ViT={T_val_vit:.3f}, CLIP={T_val_clip:.3f}, Ens(70/30)={T_val_70_30:.3f}, Ens(90/10)={T_val_90_10:.3f}\n")

    # Step 2: Extract test set probabilities
    print("Step 2: Evaluating untouched test split (backend/data/test)...")
    test_samples, test_labels = load_dataset_split(test_dir)
    v_test_vit, v_test_clip, total_test_time = extract_split_probabilities(
        test_samples, vit_model, vit_processor, clip_model, clip_processor, device, "Test"
    )
    num_test = len(test_samples)
    print(f"Test evaluation completed in {total_test_time:.2f}s ({num_test/total_test_time:.2f} images/s).\n")

    # Ensembles on test
    test_ens_90_10 = 0.90 * v_test_vit + 0.10 * v_test_clip
    test_ens_90_10 = test_ens_90_10 / np.sum(test_ens_90_10, axis=1, keepdims=True)

    test_ens_70_30 = 0.70 * v_test_vit + 0.30 * v_test_clip
    test_ens_70_30 = test_ens_70_30 / np.sum(test_ens_70_30, axis=1, keepdims=True)

    # Step 3: Analyze all 4 configurations
    print("Step 3: Calculating metrics and error analysis for all configurations...")
    vit_res = analyze_configuration("ViT_V2_Only (1.00 / 0.00)", v_test_vit, test_labels, T_val_vit, total_test_time)
    clip_res = analyze_configuration("CLIP_V2_Only (0.00 / 1.00)", v_test_clip, test_labels, T_val_clip, total_test_time)
    ens_70_30_res = analyze_configuration("Legacy_Ensemble (0.70 / 0.30)", test_ens_70_30, test_labels, T_val_70_30, total_test_time)
    ens_90_10_res = analyze_configuration("Selected_Ensemble (0.90 / 0.10)", test_ens_90_10, test_labels, T_val_90_10, total_test_time)

    # Check whether 90/10 outperforms ViT-only
    macro_f1_gain = ens_90_10_res["macro_f1"] - vit_res["macro_f1"]
    acc_gain = ens_90_10_res["accuracy"] - vit_res["accuracy"]
    outperforms_vit = (macro_f1_gain > 0) or (macro_f1_gain == 0 and acc_gain > 0)

    # Assemble final output
    final_report = {
        "dataset": "backend/data/test",
        "total_test_samples": num_test,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_fitted_temperatures": {
            "vit_only": round(T_val_vit, 4),
            "clip_only": round(T_val_clip, 4),
            "ensemble_70_30": round(T_val_70_30, 4),
            "ensemble_90_10": round(T_val_90_10, 4),
        },
        "summary_comparison": {
            "vit_v2_only": {
                "weights": "1.00 ViT / 0.00 CLIP",
                "accuracy": vit_res["accuracy"],
                "macro_f1": vit_res["macro_f1"],
                "weighted_f1": vit_res["weighted_f1"],
                "ece_raw": vit_res["calibration"]["uncalibrated_ece"],
                "ece_calibrated": vit_res["calibration"]["calibrated_ece"],
                "brier_score": vit_res["calibration"]["uncalibrated_brier_score"],
            },
            "clip_v2_only": {
                "weights": "0.00 ViT / 1.00 CLIP",
                "accuracy": clip_res["accuracy"],
                "macro_f1": clip_res["macro_f1"],
                "weighted_f1": clip_res["weighted_f1"],
                "ece_raw": clip_res["calibration"]["uncalibrated_ece"],
                "ece_calibrated": clip_res["calibration"]["calibrated_ece"],
                "brier_score": clip_res["calibration"]["uncalibrated_brier_score"],
            },
            "legacy_ensemble_70_30": {
                "weights": "0.70 ViT / 0.30 CLIP",
                "accuracy": ens_70_30_res["accuracy"],
                "macro_f1": ens_70_30_res["macro_f1"],
                "weighted_f1": ens_70_30_res["weighted_f1"],
                "ece_raw": ens_70_30_res["calibration"]["uncalibrated_ece"],
                "ece_calibrated": ens_70_30_res["calibration"]["calibrated_ece"],
                "brier_score": ens_70_30_res["calibration"]["uncalibrated_brier_score"],
            },
            "selected_ensemble_90_10": {
                "weights": "0.90 ViT / 0.10 CLIP",
                "accuracy": ens_90_10_res["accuracy"],
                "macro_f1": ens_90_10_res["macro_f1"],
                "weighted_f1": ens_90_10_res["weighted_f1"],
                "ece_raw": ens_90_10_res["calibration"]["uncalibrated_ece"],
                "ece_calibrated": ens_90_10_res["calibration"]["calibrated_ece"],
                "brier_score": ens_90_10_res["calibration"]["uncalibrated_brier_score"],
            },
        },
        "verdict": {
            "outperforms_vit_only": bool(outperforms_vit),
            "macro_f1_delta": round(macro_f1_gain, 4),
            "accuracy_delta": round(acc_gain, 4),
            "accuracy_gain_correct_samples": int(round(acc_gain * num_test)),
            "outperforms_legacy_70_30": bool(ens_90_10_res["macro_f1"] > ens_70_30_res["macro_f1"]),
        },
        "detailed_results": {
            "vit_v2_only": vit_res,
            "clip_v2_only": clip_res,
            "legacy_ensemble_70_30": ens_70_30_res,
            "selected_ensemble_90_10": ens_90_10_res,
        },
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # Step 4: Print clear comparison table
    print("\n" + "=" * 90)
    print("FINAL TEST EVALUATION RESULTS: 90/10 ENSEMBLE vs PREVIOUS CONFIGURATIONS (N=440)")
    print("=" * 90)
    row_fmt = "{:<26} | {:<10} | {:<10} | {:<12} | {:<9} | {:<10} | {:<8}"
    print(row_fmt.format("Configuration", "Accuracy", "Macro F1", "Weighted F1", "ECE (Raw)", "ECE (Cal)", "Brier"))
    print("-" * 90)
    print(row_fmt.format(
        "ViT V2 Only (1.00/0.00)",
        f"{vit_res['accuracy']*100:.2f}%",
        f"{vit_res['macro_f1']*100:.2f}%",
        f"{vit_res['weighted_f1']*100:.2f}%",
        f"{vit_res['calibration']['uncalibrated_ece']*100:.2f}%",
        f"{vit_res['calibration']['calibrated_ece']*100:.2f}%",
        f"{vit_res['calibration']['uncalibrated_brier_score']:.4f}",
    ))
    print(row_fmt.format(
        "CLIP V2 Only (0.00/1.00)",
        f"{clip_res['accuracy']*100:.2f}%",
        f"{clip_res['macro_f1']*100:.2f}%",
        f"{clip_res['weighted_f1']*100:.2f}%",
        f"{clip_res['calibration']['uncalibrated_ece']*100:.2f}%",
        f"{clip_res['calibration']['calibrated_ece']*100:.2f}%",
        f"{clip_res['calibration']['uncalibrated_brier_score']:.4f}",
    ))
    print(row_fmt.format(
        "Legacy 70/30 Ensemble",
        f"{ens_70_30_res['accuracy']*100:.2f}%",
        f"{ens_70_30_res['macro_f1']*100:.2f}%",
        f"{ens_70_30_res['weighted_f1']*100:.2f}%",
        f"{ens_70_30_res['calibration']['uncalibrated_ece']*100:.2f}%",
        f"{ens_70_30_res['calibration']['calibrated_ece']*100:.2f}%",
        f"{ens_70_30_res['calibration']['uncalibrated_brier_score']:.4f}",
    ))
    print(row_fmt.format(
        "Selected 90/10 Ensemble",
        f"{ens_90_10_res['accuracy']*100:.2f}%",
        f"{ens_90_10_res['macro_f1']*100:.2f}%",
        f"{ens_90_10_res['weighted_f1']*100:.2f}%",
        f"{ens_90_10_res['calibration']['uncalibrated_ece']*100:.2f}%",
        f"{ens_90_10_res['calibration']['calibrated_ece']*100:.2f}%",
        f"{ens_90_10_res['calibration']['uncalibrated_brier_score']:.4f}",
    ))
    print("=" * 90)

    print("\nPER-CLASS F1 BREAKDOWN ON TEST SET:")
    print(f"{'Class':<12} | {'ViT Only':<10} | {'CLIP Only':<10} | {'70/30 Ens':<10} | {'90/10 Ens':<10} | {'Support'}")
    print("-" * 75)
    for c in V2_CLASSES:
        print(
            f"{c:<12} | "
            f"{vit_res['per_class'][c]['f1']*100:>7.2f}%  | "
            f"{clip_res['per_class'][c]['f1']*100:>7.2f}%  | "
            f"{ens_70_30_res['per_class'][c]['f1']*100:>7.2f}%  | "
            f"{ens_90_10_res['per_class'][c]['f1']*100:>7.2f}%  | "
            f"{vit_res['per_class'][c]['support']:>5}"
        )
    print("-" * 75)

    print(f"\nSaved full results to: {out_file.resolve()}\n")
    return final_report


if __name__ == "__main__":
    run_evaluation()
