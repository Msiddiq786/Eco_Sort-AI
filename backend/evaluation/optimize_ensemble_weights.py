"""
EcoSort AI V2 — Validation-Only Ensemble Weight Grid Search & Analysis.

Strictly evaluates on backend/data/val (DOES NOT touch test set).
Sweeps ViT weights: 1.00 down to 0.00 (step 0.10) where CLIP weight = 1 - ViT weight.
Evaluates:
- Overall Accuracy
- Macro F1 (primary criterion)
- Weighted F1
- Per-class F1 across all 7 classes
Saves comprehensive results to:
backend/evaluation/results_v2/ensemble_weight_analysis.json
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

WEIGHT_GRID = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]


def run_weight_optimization():
    root = _BACKEND_DIR
    val_dir = root / "data" / "val"
    v2_best_dir = root / "models" / "vit_v2" / "best"
    out_file = root / "evaluation" / "results_v2" / "ensemble_weight_analysis.json"

    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found at {val_dir.resolve()}")
    if not (v2_best_dir / "config.json").exists():
        raise FileNotFoundError(f"V2 best checkpoint not found at {v2_best_dir.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "CUDA" if device.type == "cuda" else "CPU"

    print("=========================================================================")
    print("EcoSort AI V2 — Validation Ensemble Weight Grid Search")
    print("=========================================================================")
    print(f"Validation Dataset: {val_dir.resolve()} (Strictly held-out validation set)")
    print(f"ViT V2 Checkpoint:  {v2_best_dir.resolve()}")
    print(f"Compute Device:     {device_name}")
    print("=========================================================================\n")

    # 1. Load ViT V2 model
    print("Loading ViT V2 model...")
    vit_model = ViTForImageClassification.from_pretrained(str(v2_best_dir)).to(device)
    vit_processor = ViTImageProcessor.from_pretrained(str(v2_best_dir))
    vit_model.eval()

    # 2. Load CLIP Foundation model
    print("Loading CLIP V2 model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # 3. Collect validation samples
    samples = []
    for cls in V2_CLASSES:
        folder = val_dir / cls
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in folder.glob(ext):
                samples.append((p, CLASS_TO_IDX[cls], cls))
    samples = sorted(samples, key=lambda x: str(x[0]))
    y_true = np.array([s[1] for s in samples])
    num_samples = len(samples)
    print(f"Loaded {num_samples} validation samples across {len(V2_CLASSES)} classes.\n")

    # 4. Extract single-pass predictions for ViT and CLIP
    print("Extracting ViT and CLIP validation probabilities...")
    start_time = time.time()
    vit_probs_list: List[np.ndarray] = []
    clip_probs_list: List[np.ndarray] = []

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
            print(f"  Processed {idx}/{num_samples} validation images ({elapsed:.1f}s, {idx/elapsed:.2f} img/s)")

    vit_probs = np.array(vit_probs_list)    # [N, 7]
    clip_probs = np.array(clip_probs_list)  # [N, 7]

    print("\nExtraction complete. Running weight sweep across grid...")

    # 5. Sweep weights and calculate metrics
    weight_results: List[Dict[str, Any]] = []

    for w_vit in WEIGHT_GRID:
        w_clip = round(1.0 - w_vit, 2)
        fused_probs = w_vit * vit_probs + w_clip * clip_probs
        # Normalize sum to 1
        fused_probs = fused_probs / np.sum(fused_probs, axis=1, keepdims=True)

        preds = np.argmax(fused_probs, axis=1)

        acc = float(accuracy_score(y_true, preds))
        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))

        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, preds, labels=list(range(7)), zero_division=0
        )
        cm = confusion_matrix(y_true, preds, labels=list(range(7)))

        per_class_f1 = {V2_CLASSES[i]: round(float(f1[i]), 4) for i in range(7)}
        per_class_rec = {V2_CLASSES[i]: round(float(rec[i]), 4) for i in range(7)}
        per_class_prec = {V2_CLASSES[i]: round(float(prec[i]), 4) for i in range(7)}

        record = {
            "vit_weight": round(w_vit, 2),
            "clip_weight": round(w_clip, 2),
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "per_class_f1": per_class_f1,
            "per_class_recall": per_class_rec,
            "per_class_precision": per_class_prec,
            "cardboard_pred_paper": int(cm[CLASS_TO_IDX["cardboard"], CLASS_TO_IDX["paper"]]),
            "paper_pred_cardboard": int(cm[CLASS_TO_IDX["paper"], CLASS_TO_IDX["cardboard"]]),
            "glass_pred_plastic": int(cm[CLASS_TO_IDX["glass"], CLASS_TO_IDX["plastic"]]),
            "plastic_pred_glass": int(cm[CLASS_TO_IDX["plastic"], CLASS_TO_IDX["glass"]]),
            "confusion_matrix": cm.tolist(),
        }
        weight_results.append(record)

    # 6. Determine best weight (Primary: Macro F1, Secondary: Accuracy)
    best_record = max(weight_results, key=lambda x: (x["macro_f1"], x["accuracy"]))
    vit_only_record = next(r for r in weight_results if r["vit_weight"] == 1.00)
    clip_only_record = next(r for r in weight_results if r["vit_weight"] == 0.00)

    # 7. Print the required formatted table
    print("\n" + "=" * 80)
    print("ENSEMBLE WEIGHT GRID SEARCH RESULTS (VALIDATION SET: N=439)")
    print("=" * 80)
    header = f"{'ViT Weight':<12} | {'CLIP Weight':<12} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12} | {'Status'}"
    print(header)
    print("-" * 80)

    for r in weight_results:
        w_v_str = f"{r['vit_weight']:.2f}"
        w_c_str = f"{r['clip_weight']:.2f}"
        acc_str = f"{r['accuracy']*100:.2f}%"
        mf1_str = f"{r['macro_f1']*100:.2f}%"
        wf1_str = f"{r['weighted_f1']*100:.2f}%"
        status = ""
        if r["vit_weight"] == best_record["vit_weight"]:
            status = "★ BEST"
        elif r["vit_weight"] == 1.00:
            status = "(ViT-only)"
        elif r["vit_weight"] == 0.00:
            status = "(CLIP-only)"
        elif r["vit_weight"] == 0.70:
            status = "(Current 70/30)"
        print(f"{w_v_str:<12} | {w_c_str:<12} | {acc_str:<10} | {mf1_str:<10} | {wf1_str:<12} | {status}")
    print("-" * 80)

    # 8. Detailed Comparison: Best Ensemble vs ViT-Only
    best_w = best_record["vit_weight"]
    macro_f1_delta = best_record["macro_f1"] - vit_only_record["macro_f1"]
    acc_delta = best_record["accuracy"] - vit_only_record["accuracy"]
    weighted_f1_delta = best_record["weighted_f1"] - vit_only_record["weighted_f1"]

    improves_over_vit = (macro_f1_delta > 0) or (macro_f1_delta == 0 and acc_delta > 0)

    print("\n" + "=" * 80)
    print("DETAILED PER-CLASS F1 COMPARISON: ViT-Only (1.00) vs Best Setting")
    print("=" * 80)
    print(f"{'Class':<12} | {'ViT-Only (1.00)':<15} | {'Best (' + str(best_w) + ')':<15} | {'Delta':<10} | {'Change'}")
    print("-" * 80)

    class_deltas = {}
    for c in V2_CLASSES:
        vit_f1 = vit_only_record["per_class_f1"][c]
        best_f1 = best_record["per_class_f1"][c]
        d = round(best_f1 - vit_f1, 4)
        class_deltas[c] = d
        d_str = f"{d*100:+.2f}%"
        change_desc = "IMPROVED" if d > 0 else ("DECLINED" if d < 0 else "NO CHANGE")
        print(f"{c:<12} | {vit_f1*100:>6.2f}%         | {best_f1*100:>6.2f}%         | {d_str:>8}   | {change_desc}")
    print("-" * 80)

    # 9. Statistical & Practical Meaningfulness Assessment
    # Each sample in validation dataset (N=439) represents 1/439 = 0.228% accuracy.
    sample_accuracy_delta_count = round(acc_delta * num_samples)

    if improves_over_vit:
        if abs(macro_f1_delta) < 0.005 and abs(acc_delta) < 0.005:
            practical_meaning = "Marginal / Negligible: The improvement is within 1-2 samples of variance."
        else:
            practical_meaning = f"Substantial: +{macro_f1_delta*100:.2f}% Macro F1 (+{sample_accuracy_delta_count} net correct samples)."
    else:
        practical_meaning = "No Improvement: ViT-only performs equal to or superior to the ensemble on the validation dataset."

    analysis_data = {
        "dataset": "backend/data/val",
        "sample_count": num_samples,
        "selection_criteria": {
            "primary": "validation_macro_f1",
            "secondary": "validation_accuracy",
        },
        "best_setting": {
            "vit_weight": best_record["vit_weight"],
            "clip_weight": best_record["clip_weight"],
            "accuracy": best_record["accuracy"],
            "macro_f1": best_record["macro_f1"],
            "weighted_f1": best_record["weighted_f1"],
        },
        "vit_only_setting": {
            "vit_weight": vit_only_record["vit_weight"],
            "clip_weight": vit_only_record["clip_weight"],
            "accuracy": vit_only_record["accuracy"],
            "macro_f1": vit_only_record["macro_f1"],
            "weighted_f1": vit_only_record["weighted_f1"],
        },
        "clip_only_setting": {
            "vit_weight": clip_only_record["vit_weight"],
            "clip_weight": clip_only_record["clip_weight"],
            "accuracy": clip_only_record["accuracy"],
            "macro_f1": clip_only_record["macro_f1"],
            "weighted_f1": clip_only_record["weighted_f1"],
        },
        "comparison_to_vit_only": {
            "macro_f1_delta": round(macro_f1_delta, 4),
            "accuracy_delta": round(acc_delta, 4),
            "weighted_f1_delta": round(weighted_f1_delta, 4),
            "per_class_f1_deltas": class_deltas,
            "improves_over_vit": bool(improves_over_vit),
            "practical_meaning": practical_meaning,
        },
        "grid_results": weight_results,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2)

    print(f"\nAnalysis successfully written to: {out_file.resolve()}\n")
    return analysis_data


if __name__ == "__main__":
    run_weight_optimization()
