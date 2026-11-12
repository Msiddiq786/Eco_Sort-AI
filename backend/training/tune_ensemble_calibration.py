"""
Validation-only grid search for optimal ensemble weights & Temperature Scaling.

Strictly operates on backend/data/val/ (379 images).
Does NOT touch the test set.

Saves results to backend/evaluation/results/ensemble_calibration.json.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
from transformers import CLIPModel, CLIPProcessor, ViTForImageClassification, ViTImageProcessor

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.ml.calibration import TemperatureCalibrator, compute_ece
from app.ml.clip_model import CLIP_PROMPT_TEMPLATES, _extract_tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("tune_ensemble")

VAL_DATA_DIR = _BACKEND_DIR / "data" / "val"
VIT_CHECKPOINT_DIR = _BACKEND_DIR / "models" / "vit" / "best"
RESULTS_DIR = _BACKEND_DIR / "evaluation" / "results"

WASTE_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {c: i for i, c in enumerate(WASTE_CLASSES)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_val_dataset() -> List[Tuple[Path, int]]:
    samples = []
    for cls_name in WASTE_CLASSES:
        cls_dir = VAL_DATA_DIR / cls_name
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                samples.append((img_path, CLASS_TO_IDX[cls_name]))
    logger.info("Loaded %d validation samples across %d classes.", len(samples), len(WASTE_CLASSES))
    return samples


def run_tuning() -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_val_dataset()

    # 1. Load models
    vit_model = ViTForImageClassification.from_pretrained(str(VIT_CHECKPOINT_DIR)).to(DEVICE)
    vit_processor = ViTImageProcessor.from_pretrained(str(VIT_CHECKPOINT_DIR))
    vit_model.eval()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # 2. Precompute CLIP text embeddings
    with torch.no_grad():
        class_feats = []
        for cls_name in WASTE_CLASSES:
            prompts = CLIP_PROMPT_TEMPLATES[cls_name]
            txt_in = clip_processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
            txt_out = clip_model.get_text_features(**txt_in)
            t = _extract_tensor(txt_out)
            t = t / t.norm(dim=-1, keepdim=True)
            m = t.mean(dim=0, keepdim=True)
            class_feats.append(m / m.norm(dim=-1, keepdim=True))
        clip_text_emb = torch.cat(class_feats, dim=0)
        logit_scale = clip_model.logit_scale.exp()

    # 3. Predict on validation set
    y_true: List[int] = []
    vit_val_probs: List[np.ndarray] = []
    clip_val_probs: List[np.ndarray] = []

    batch_size = 32
    num_batches = (len(samples) + batch_size - 1) // batch_size

    logger.info("Computing validation probabilities (%d batches)...", num_batches)
    for b in range(num_batches):
        batch = samples[b * batch_size : (b + 1) * batch_size]
        pil_images = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p, _ in batch]
        y_true.extend([lbl for _, lbl in batch])

        with torch.no_grad():
            vit_in = vit_processor(images=pil_images, return_tensors="pt").to(DEVICE)
            vit_out = vit_model(**vit_in)
            vp = torch.softmax(vit_out.logits, dim=-1).cpu().numpy()
            vit_val_probs.append(vp)

            clip_in = clip_processor(images=pil_images, return_tensors="pt").to(DEVICE)
            img_out = clip_model.get_image_features(**clip_in)
            it = _extract_tensor(img_out)
            it = it / it.norm(dim=-1, keepdim=True)
            cp = torch.softmax(logit_scale * (it @ clip_text_emb.T), dim=-1).cpu().numpy()
            clip_val_probs.append(cp)

    vit_probs_all = np.concatenate(vit_val_probs, axis=0)  # [379, 6]
    clip_probs_all = np.concatenate(clip_val_probs, axis=0)  # [379, 6]
    y_true_arr = np.array(y_true, dtype=np.int64)

    # 4. Sweep ensemble weights on Validation set
    weight_candidates = [
        (0.90, 0.10),
        (0.80, 0.20),
        (0.70, 0.30),
        (0.60, 0.40),
        (0.50, 0.50),
        (0.40, 0.60),
        (0.30, 0.70),
        (0.20, 0.80),
        (0.10, 0.90),
    ]

    sweep_results = []
    best_weight_config = (0.70, 0.30)
    best_val_f1 = -1.0

    print("\n" + "=" * 75)
    print("ENSEMBLE WEIGHT SWEEP ON VALIDATION SET (379 SAMPLES)")
    print("=" * 75)
    print(f"{'ViT Weight':<12} | {'CLIP Weight':<12} | {'Val Accuracy':<14} | {'Balanced Acc':<14} | {'Macro F1':<10}")
    print("-" * 75)

    for w_vit, w_clip in weight_candidates:
        fused = w_vit * vit_probs_all + w_clip * clip_probs_all
        fused = fused / fused.sum(axis=-1, keepdims=True)
        preds = np.argmax(fused, axis=-1)

        acc = float(np.mean(preds == y_true_arr))
        b_acc = float(balanced_accuracy_score(y_true_arr, preds))
        macro_f1 = float(f1_score(y_true_arr, preds, average="macro", zero_division=0))

        res_entry = {
            "vit_weight": w_vit,
            "clip_weight": w_clip,
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(b_acc, 4),
            "macro_f1": round(macro_f1, 4),
        }
        sweep_results.append(res_entry)
        print(f"{w_vit:>10.2f}   | {w_clip:>10.2f}   | {acc*100:>12.2f}% | {b_acc*100:>12.2f}% | {macro_f1*100:>8.2f}%")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_weight_config = (w_vit, w_clip)

    print("=" * 75)
    logger.info("Best validation configuration: ViT=%.2f, CLIP=%.2f (Macro F1=%.4f)", best_weight_config[0], best_weight_config[1], best_val_f1)

    # 5. Temperature Scaling Calibration on Best Validation Predictions
    best_w_vit, best_w_clip = best_weight_config
    best_fused_val = best_w_vit * vit_probs_all + best_w_clip * clip_probs_all
    best_fused_val = best_fused_val / best_fused_val.sum(axis=-1, keepdims=True)

    uncal_ece = compute_ece(best_fused_val.tolist(), y_true)
    calibrator = TemperatureCalibrator()
    optimal_t = calibrator.fit(best_fused_val.tolist(), y_true)

    # Compute calibrated validation probabilities
    cal_probs_list = [calibrator.calibrate({WASTE_CLASSES[i]: float(p[i]) for i in range(6)}) for p in best_fused_val]
    cal_probs_arr = np.array([[p[c] for c in WASTE_CLASSES] for p in cal_probs_list])
    cal_ece = compute_ece(cal_probs_arr.tolist(), y_true)

    logger.info("Calibration Results: Uncalibrated ECE = %.4f -> Calibrated ECE = %.4f (T = %.4f)", uncal_ece, cal_ece, optimal_t)

    out_data = {
        "dataset": "TrashNet Validation Split",
        "num_val_samples": len(y_true),
        "classes": WASTE_CLASSES,
        "weight_sweep": sweep_results,
        "optimal_weights": {
            "vit_weight": best_w_vit,
            "clip_weight": best_w_clip,
            "val_accuracy": round(float(np.mean(np.argmax(best_fused_val, axis=-1) == y_true_arr)), 4),
            "val_macro_f1": round(best_val_f1, 4),
        },
        "calibration": {
            "temperature": optimal_t,
            "uncalibrated_ece": uncal_ece,
            "calibrated_ece": cal_ece,
            "ece_reduction_percent": round(((uncal_ece - cal_ece) / max(uncal_ece, 1e-6)) * 100, 2),
        },
    }

    out_file = RESULTS_DIR / "ensemble_calibration.json"
    out_file.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    logger.info("Saved ensemble tuning and calibration parameters to %s", out_file)
    return out_data


if __name__ == "__main__":
    run_tuning()
