"""
Comprehensive evaluation script for EcoSort AI (Phase 6).

Evaluates:
  1. Fine-tuned Vision Transformer (ViT)
  2. CLIP Foundation Model (zero-shot with prompt ensembling)
  3. Weighted Ensemble (70% ViT + 30% CLIP)

Dataset: Held-out TrashNet test dataset (backend/data/test/).
Outputs saved to backend/evaluation/results/:
  - metrics.json (used by /api/metrics and PerformancePage)
  - model_comparison.json (full side-by-side comparison)
  - confusion_matrix.json (6x6 matrix for each model)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import CLIPModel, CLIPProcessor, ViTForImageClassification, ViTImageProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("evaluate_all")

_BACKEND_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = _BACKEND_DIR / "data" / "test"
VIT_CHECKPOINT_DIR = _BACKEND_DIR / "models" / "vit" / "best"
RESULTS_DIR = _BACKEND_DIR / "evaluation" / "results"

WASTE_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {c: i for i, c in enumerate(WASTE_CLASSES)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Multi-prompt templates for robust zero-shot CLIP classification
CLIP_PROMPT_TEMPLATES: Dict[str, List[str]] = {
    "cardboard": [
        "a photo of a cardboard box or cardboard waste",
        "a photo of corrugated cardboard waste",
        "a photo of a brown cardboard box or carton",
        "a photo of discarded cardboard packaging",
    ],
    "glass": [
        "a photo of a glass bottle or glass container",
        "a photo of a beverage glass bottle",
        "a photo of a clear glass jar or container",
        "a photo of broken or recyclable glass",
    ],
    "metal": [
        "a photo of a metal can or metallic object",
        "a photo of an aluminum soda can",
        "a photo of a discarded tin food can",
        "a photo of recyclable scrap metal or aluminum can",
    ],
    "paper": [
        "a photo of paper waste or newspaper",
        "a photo of a sheet of paper or document",
        "a photo of crumpled office paper waste",
        "a photo of printed newspaper or magazine paper",
    ],
    "plastic": [
        "a photo of a plastic bottle or plastic bag",
        "a photo of a discarded plastic water bottle",
        "a photo of a clear disposable plastic bottle",
        "a photo of plastic packaging waste or plastic container",
        "a photo of a plastic cup or plastic detergent bottle",
    ],
    "trash": [
        "a photo of general household trash or garbage",
        "a photo of non-recyclable domestic garbage",
        "a photo of miscellaneous municipal waste or rubbish",
        "a photo of dirty non-recyclable trash",
    ],
}


def _extract_tensor(out: Any) -> torch.Tensor:
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if isinstance(out, torch.Tensor):
        return out
    return out[0]


def load_test_dataset() -> List[Tuple[Path, int]]:
    """Discover all test images across the 6 class directories."""
    samples: List[Tuple[Path, int]] = []
    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(f"Test directory not found at: {TEST_DATA_DIR}")

    for cls_name in WASTE_CLASSES:
        cls_dir = TEST_DATA_DIR / cls_name
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                samples.append((img_path, CLASS_TO_IDX[cls_name]))

    logger.info("Found %d test images across %d classes.", len(samples), len(WASTE_CLASSES))
    return samples


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_per, r_per, f_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(WASTE_CLASSES))), average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(WASTE_CLASSES))))

    per_class = {}
    for idx, name in enumerate(WASTE_CLASSES):
        per_class[name] = {
            "precision": round(float(p_per[idx]), 4),
            "recall": round(float(r_per[idx]), 4),
            "f1": round(float(f_per[idx]), 4),
            "support": int(sup_per[idx]),
        }

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(p_weighted), 4),
        "weighted_recall": round(float(r_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": WASTE_CLASSES,
    }


def run_evaluation() -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_test_dataset()
    if not samples:
        raise ValueError("No test images found in dataset.")

    # ── 1. Load ViT Model ────────────────────────────────────────────────────
    logger.info("Loading ViT model from %s ...", VIT_CHECKPOINT_DIR)
    vit_model = ViTForImageClassification.from_pretrained(str(VIT_CHECKPOINT_DIR)).to(DEVICE)
    vit_processor = ViTImageProcessor.from_pretrained(str(VIT_CHECKPOINT_DIR))
    vit_model.eval()

    # ── 2. Load CLIP Model ───────────────────────────────────────────────────
    logger.info("Loading CLIP model 'openai/clip-vit-base-patch32' ...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # ── 3. Precompute CLIP Class Text Embeddings ──────────────────────────────
    logger.info("Computing CLIP prompt embeddings for 6 classes...")
    class_text_embeddings_list = []
    with torch.no_grad():
        for cls_name in WASTE_CLASSES:
            prompts = CLIP_PROMPT_TEMPLATES[cls_name]
            txt_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
            txt_out = clip_model.get_text_features(**txt_inputs)
            txt_feats = _extract_tensor(txt_out)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            cls_feat = txt_feats.mean(dim=0, keepdim=True)
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
            class_text_embeddings_list.append(cls_feat)
        clip_text_embeddings = torch.cat(class_text_embeddings_list, dim=0)  # [6, 512]
        clip_logit_scale = clip_model.logit_scale.exp()

    # ── 4. Evaluate across all test samples ──────────────────────────────────
    y_true: List[int] = []
    vit_preds: List[int] = []
    clip_preds: List[int] = []
    ensemble_preds: List[int] = []

    logger.info("Running inference on %d test samples...", len(samples))
    batch_size = 32
    num_batches = (len(samples) + batch_size - 1) // batch_size

    for b in range(num_batches):
        batch_samples = samples[b * batch_size : (b + 1) * batch_size]
        pil_images = []
        batch_labels = []

        for p, lbl in batch_samples:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img).convert("RGB")
            pil_images.append(img)
            batch_labels.append(lbl)

        y_true.extend(batch_labels)

        # ViT forward
        with torch.no_grad():
            vit_inputs = vit_processor(images=pil_images, return_tensors="pt").to(DEVICE)
            vit_outputs = vit_model(**vit_inputs)
            vit_probs_batch = torch.softmax(vit_outputs.logits, dim=-1).cpu().numpy()  # [B, 6]

        # CLIP forward
        with torch.no_grad():
            clip_inputs = clip_processor(images=pil_images, return_tensors="pt").to(DEVICE)
            img_out = clip_model.get_image_features(**clip_inputs)
            img_feats = _extract_tensor(img_out)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            clip_logits = clip_logit_scale * (img_feats @ clip_text_embeddings.T)
            clip_probs_batch = torch.softmax(clip_logits, dim=-1).cpu().numpy()  # [B, 6]

        # Ensemble fusion: 70% ViT + 30% CLIP
        ensemble_probs_batch = 0.70 * vit_probs_batch + 0.30 * clip_probs_batch
        # Re-normalize
        ensemble_probs_batch = ensemble_probs_batch / ensemble_probs_batch.sum(axis=-1, keepdims=True)

        vit_preds.extend(vit_probs_batch.argmax(axis=-1).tolist())
        clip_preds.extend(clip_probs_batch.argmax(axis=-1).tolist())
        ensemble_preds.extend(ensemble_probs_batch.argmax(axis=-1).tolist())

        if (b + 1) % 4 == 0 or (b + 1) == num_batches:
            logger.info("  Processed %d/%d images (%.1f%%)", len(y_true), len(samples), (len(y_true) / len(samples)) * 100)

    # ── 5. Compute Metrics ───────────────────────────────────────────────────
    vit_metrics = compute_metrics(y_true, vit_preds)
    clip_metrics = compute_metrics(y_true, clip_preds)
    ensemble_metrics = compute_metrics(y_true, ensemble_preds)

    comparison = {
        "dataset": "TrashNet (Held-out Test Split)",
        "total_test_samples": len(y_true),
        "classes": WASTE_CLASSES,
        "vit": {
            "model_name": "google/vit-base-patch16-224-in21k",
            "checkpoint": str(VIT_CHECKPOINT_DIR),
            **vit_metrics,
        },
        "clip": {
            "model_name": "openai/clip-vit-base-patch32",
            "prompt_strategy": "multi_prompt_ensembling",
            **clip_metrics,
        },
        "ensemble": {
            "formula": "0.70 * vit + 0.30 * clip",
            "vit_weight": 0.70,
            "clip_weight": 0.30,
            **ensemble_metrics,
        },
    }

    # Save outputs
    (RESULTS_DIR / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(ensemble_metrics, indent=2), encoding="utf-8")
    (RESULTS_DIR / "confusion_matrix.json").write_text(
        json.dumps(
            {
                "vit": vit_metrics["confusion_matrix"],
                "clip": clip_metrics["confusion_matrix"],
                "ensemble": ensemble_metrics["confusion_matrix"],
                "class_names": WASTE_CLASSES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Evaluation complete. Metrics saved to %s", RESULTS_DIR)
    return comparison


if __name__ == "__main__":
    results = run_evaluation()
    print("\n" + "=" * 70)
    print("ECOSORT AI — HELD-OUT TEST DATASET EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Model':<18} | {'Accuracy':<10} | {'Macro Prec':<12} | {'Macro Rec':<12} | {'Macro F1':<10}")
    print("-" * 70)
    for name, key in [("ViT (Fine-tuned)", "vit"), ("CLIP (Zero-Shot)", "clip"), ("70:30 Ensemble", "ensemble")]:
        m = results[key]
        print(f"{name:<18} | {m['accuracy']*100:>8.2f}% | {m['macro_precision']*100:>10.2f}% | {m['macro_recall']*100:>10.2f}% | {m['macro_f1']*100:>8.2f}%")
    print("=" * 70)
