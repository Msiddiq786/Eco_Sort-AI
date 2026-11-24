"""
EcoSort AI V2 — Standalone 7-Class Test Split Ensemble Evaluation Script.

This script evaluates the completed V2 7-class Vision Transformer (from backend/models/vit_v2/best)
and the frozen CLIP Foundation V2 model on the UNTOUCHED test split (backend/data/test).

Computes:
1. ViT V2 Test Performance (Accuracy, Macro F1, Weighted F1, Per-Class Metrics, Confusion Matrix, ECE, Brier Score)
2. CLIP V2 Test Performance (Accuracy, Macro F1, Weighted F1, Per-Class Metrics, Confusion Matrix, ECE, Brier Score)
3. 70/30 ViT+CLIP Ensemble Performance (Accuracy, Macro F1, Weighted F1, Per-Class Metrics, Confusion Matrix, ECE, Brier Score)
4. Specific confusion breakdowns (Cardboard vs Paper, Glass vs Plastic, Organic, Trash)
5. Strict probability normalization audit across all test samples.
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


def run_test_evaluation():
    root = _BACKEND_DIR
    test_dir = root / "data" / "test"
    v2_best_dir = root / "models" / "vit_v2" / "best"

    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test split not found at {test_dir.resolve()}")
    if not (v2_best_dir / "config.json").exists():
        raise FileNotFoundError(f"V2 best checkpoint not found at {v2_best_dir.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "CUDA" if device.type == "cuda" else "CPU"

    # 1. Load ViT V2 Model & Processor
    print(f"Loading V2 ViT model from {v2_best_dir}...")
    vit_model = ViTForImageClassification.from_pretrained(str(v2_best_dir)).to(device)
    vit_processor = ViTImageProcessor.from_pretrained(str(v2_best_dir))
    vit_model.eval()

    # 2. Load CLIP Model & Processor
    print("Loading CLIP Foundation Model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # 3. Load Test Split (strictly backend/data/test)
    samples = []
    for cls in V2_CLASSES:
        folder = test_dir / cls
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in folder.glob(ext):
                samples.append((p, CLASS_TO_IDX[cls], cls))
    samples = sorted(samples, key=lambda x: str(x[0]))
    y_true = np.array([s[1] for s in samples])
    num_samples = len(samples)

    # 4. Print pre-evaluation details (Requirement 1)
    print("\n=======================================================")
    print("EcoSort AI V2 — 7-Class Test Split Ensemble Evaluation")
    print("=======================================================")
    print(f"V2 ViT Checkpoint:  {v2_best_dir.resolve()}")
    print(f"Test Dataset Path:  {test_dir.resolve()}")
    print(f"Total Test Samples: {num_samples}")
    print(f"Compute Device:     {device_name}")
    print("=======================================================\n")

    vit_probs_list: List[np.ndarray] = []
    clip_probs_list: List[np.ndarray] = []
    ensemble_probs_list: List[np.ndarray] = []

    start_time = time.time()

    # 5. Sample evaluation loop with progress logging (Requirements 2 & 3)
    for idx, (p, y_idx, true_cls) in enumerate(samples, start=1):
        img = Image.open(p).convert("RGB")

        # ViT Inference
        vit_inputs = vit_processor(images=img, return_tensors="pt")
        vit_inputs = {k: v.to(device) for k, v in vit_inputs.items()}
        with torch.no_grad():
            vit_logits = vit_model(**vit_inputs).logits
            v_p = torch.softmax(vit_logits, dim=-1)[0].cpu().numpy()
        vit_probs_list.append(v_p)

        # CLIP Inference (frozen Class-Centroid scoring)
        c_res = predict_clip_v2(clip_model, clip_processor, img)
        c_p = np.array([c_res.probabilities[c] for c in V2_CLASSES])
        clip_probs_list.append(c_p)

        # 70/30 Ensemble
        ens_p = 0.70 * v_p + 0.30 * c_p
        ens_p = ens_p / ens_p.sum()
        ensemble_probs_list.append(ens_p)

        # Progress reporting every 10 samples and at the final sample
        if idx % 10 == 0 or idx == num_samples:
            elapsed = time.time() - start_time
            speed = idx / elapsed if elapsed > 0 else 0.0
            print(
                f"Evaluating test sample {idx}/{num_samples} "
                f"(Elapsed: {elapsed:.1f}s, Speed: {speed:.2f} samples/s)"
            )

    vit_probs = np.array(vit_probs_list)
    clip_probs = np.array(clip_probs_list)
    ens_probs = np.array(ensemble_probs_list)

    total_eval_time = time.time() - start_time

    # 6. Prediction analysis function (Requirement 7)
    def analyze_predictions(name: str, probs: np.ndarray) -> Dict[str, Any]:
        preds = np.argmax(probs, axis=1)
        acc = float(accuracy_score(y_true, preds))
        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, preds, labels=list(range(7)), zero_division=0
        )
        cm = confusion_matrix(y_true, preds, labels=list(range(7)))
        ece, mce, bins = compute_ece(probs, y_true, n_bins=10)

        # Multi-class Brier score
        y_one_hot = np.zeros_like(probs)
        for i, y in enumerate(y_true):
            y_one_hot[i, y] = 1.0
        brier = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))

        # Probability normalization audit
        norm_sums = np.sum(probs, axis=1)
        max_norm_dev = float(np.max(np.abs(norm_sums - 1.0)))
        avg_norm_dev = float(np.mean(np.abs(norm_sums - 1.0)))

        per_class = {
            V2_CLASSES[i]: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(sup[i]),
            }
            for i in range(7)
        }

        c_card = CLASS_TO_IDX["cardboard"]
        c_pap = CLASS_TO_IDX["paper"]
        c_gla = CLASS_TO_IDX["glass"]
        c_plas = CLASS_TO_IDX["plastic"]
        c_org = CLASS_TO_IDX["organic"]
        c_trash = CLASS_TO_IDX["trash"]

        return {
            "name": name,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "ece": round(ece, 4),
            "mce": round(mce, 4),
            "brier_score": round(brier, 4),
            "normalization_audit": {
                "max_deviation_from_one": max_norm_dev,
                "average_deviation_from_one": avg_norm_dev,
                "strictly_normalized": bool(max_norm_dev < 1e-4),
            },
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "cardboard_pred_paper": int(cm[c_card, c_pap]),
            "paper_pred_cardboard": int(cm[c_pap, c_card]),
            "glass_pred_plastic": int(cm[c_gla, c_plas]),
            "plastic_pred_glass": int(cm[c_plas, c_gla]),
            "organic_performance": {
                "precision": round(float(prec[c_org]), 4),
                "recall": round(float(rec[c_org]), 4),
                "f1": round(float(f1[c_org]), 4),
                "support": int(sup[c_org]),
            },
            "trash_performance": {
                "precision": round(float(prec[c_trash]), 4),
                "recall": round(float(rec[c_trash]), 4),
                "f1": round(float(f1[c_trash]), 4),
                "support": int(sup[c_trash]),
            },
        }

    results = {
        "dataset": "backend/data/test",
        "sample_count": num_samples,
        "evaluation_time_seconds": round(total_eval_time, 2),
        "device": device_name,
        "vit_v2": analyze_predictions("ViT_V2_7Class", vit_probs),
        "clip_v2": analyze_predictions("CLIP_V2_Centroid", clip_probs),
        "ensemble_70_30": analyze_predictions("Ensemble_70ViT_30CLIP", ens_probs),
    }

    out_file = root / "evaluation" / "results_v2" / "final_ensemble_test_evaluation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 7. Print final summary and exit cleanly (Requirement 8)
    ens = results["ensemble_70_30"]
    vit = results["vit_v2"]
    clip = results["clip_v2"]

    print("\n" + "=" * 65)
    print("FINAL 7-CLASS ENSEMBLE TEST SET RESULTS")
    print("=" * 65)
    print(f"Evaluated Samples:  {num_samples} images from {test_dir.resolve()}")
    print(f"Total Time:         {total_eval_time:.1f}s ({num_samples / total_eval_time:.2f} samples/s)")
    print("-" * 65)
    print(f"ViT V2:             Acc={vit['accuracy']*100:.2f}%, Macro F1={vit['macro_f1']*100:.2f}%, ECE={vit['ece']*100:.2f}%")
    print(f"CLIP V2:            Acc={clip['accuracy']*100:.2f}%, Macro F1={clip['macro_f1']*100:.2f}%, ECE={clip['ece']*100:.2f}%")
    print(f"Ensemble (70/30):   Acc={ens['accuracy']*100:.2f}%, Macro F1={ens['macro_f1']*100:.2f}%, Weighted F1={ens['weighted_f1']*100:.2f}%")
    print(f"Ensemble ECE:       {ens['ece']*100:.2f}%, Brier Score={ens['brier_score']:.4f}")
    print("-" * 65)
    print("Ensemble Per-Class F1 Scores:")
    for cls in V2_CLASSES:
        m = ens["per_class"][cls]
        print(f"  {cls:<10}: F1={m['f1']*100:.2f}% | Prec={m['precision']*100:.2f}% | Rec={m['recall']*100:.2f}% (Support={m['support']})")
    print("-" * 65)
    print(f"Key Confusions:     Cardboard->Paper: {ens['cardboard_pred_paper']} | Paper->Cardboard: {ens['paper_pred_cardboard']}")
    print(f"                    Glass->Plastic:   {ens['glass_pred_plastic']} | Plastic->Glass:   {ens['plastic_pred_glass']}")
    print(f"Probability Norm:   Max Deviation from 1.0: {ens['normalization_audit']['max_deviation_from_one']:.2e}")
    print(f"Saved Results to:   {out_file.resolve()}")
    print("=" * 65 + "\n")

    return results


if __name__ == "__main__":
    run_test_evaluation()
