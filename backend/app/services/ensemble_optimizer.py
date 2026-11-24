"""
Adaptive Ensemble Optimizer Service — EcoSort AI V3.

Performs a grid search over ViT/CLIP weight combinations using the
validation dataset (backend/data/val/), selects the combination that
maximises macro-averaged F1, and persists the result to
backend/models/ensemble_config.json.

The validation set is NEVER the test set. No test-set leakage.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Backend root directory (backend/)
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

# Config file path (backend/models/ensemble_config.json)
_CONFIG_PATH = _BACKEND_DIR / "models" / "ensemble_config.json"

# Existing offline validation grid-search results path (if previously generated)
_SAVED_ANALYSIS_PATH = _BACKEND_DIR / "evaluation" / "results_v2" / "ensemble_weight_analysis.json"

# Default baseline weights (matches settings.py)
DEFAULT_VIT_WEIGHT = 0.70
DEFAULT_CLIP_WEIGHT = 0.30

# Weight grid for search (11 steps, pairs summing to 1.0)
WEIGHT_GRID = [round(i * 0.10, 2) for i in range(11)]


def _coerce_path(p: Path) -> Path:
    """Return p directly if parent exists, else check CWD fallback."""
    if p.exists() or p.parent.exists():
        return p
    cwd_candidate = Path("models") / p.name
    if cwd_candidate.exists() or cwd_candidate.parent.exists():
        return cwd_candidate
    return p


def load_ensemble_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load persisted ensemble config.

    Returns a dict with keys:
        vit_weight, clip_weight, macro_f1, accuracy, source, optimized_at
    Falls back to the default 70/30 split if no config file exists.
    """
    resolved_path = config_path if config_path is not None else _coerce_path(_CONFIG_PATH)
    if resolved_path.exists():
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded ensemble config from %s", resolved_path)
            return data
        except Exception as exc:
            logger.warning("Failed to read ensemble config (%s); using defaults.", exc)

    return {
        "vit_weight": DEFAULT_VIT_WEIGHT,
        "clip_weight": DEFAULT_CLIP_WEIGHT,
        "macro_f1": None,
        "accuracy": None,
        "source": "default",
        "optimized_at": None,
    }


def save_ensemble_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> Path:
    """Persist ensemble config to JSON. Returns the path written."""
    resolved_path = config_path if config_path is not None else _coerce_path(_CONFIG_PATH)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved ensemble config to %s", resolved_path)
    return resolved_path


def load_saved_grid_evaluation(analysis_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Load previously computed validation grid-search results from
    backend/evaluation/results_v2/ensemble_weight_analysis.json if available.
    """
    path = analysis_path if analysis_path is not None else _SAVED_ANALYSIS_PATH
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        best = raw.get("best_setting", {})
        grid = raw.get("grid_results", [])
        if not grid:
            return None

        baseline = next(
            (g for g in grid if abs(float(g.get("vit_weight", -1)) - DEFAULT_VIT_WEIGHT) < 0.001),
            {},
        )
        all_results = [
            {
                "vit_weight": round(float(g["vit_weight"]), 2),
                "clip_weight": round(float(g["clip_weight"]), 2),
                "macro_f1": round(float(g["macro_f1"]), 4),
                "accuracy": round(float(g["accuracy"]), 4),
            }
            for g in grid
        ]
        return {
            "best_vit_weight": round(float(best.get("vit_weight", DEFAULT_VIT_WEIGHT)), 2),
            "best_clip_weight": round(float(best.get("clip_weight", DEFAULT_CLIP_WEIGHT)), 2),
            "best_macro_f1": round(float(best.get("macro_f1", 0.0)), 4),
            "best_accuracy": round(float(best.get("accuracy", 0.0)), 4),
            "baseline_vit_weight": DEFAULT_VIT_WEIGHT,
            "baseline_clip_weight": DEFAULT_CLIP_WEIGHT,
            "baseline_macro_f1": round(float(baseline.get("macro_f1", 0.0)), 4) if baseline else None,
            "baseline_accuracy": round(float(baseline.get("accuracy", 0.0)), 4) if baseline else None,
            "num_samples": raw.get("sample_count"),
            "all_results": all_results,
        }
    except Exception as exc:
        logger.warning("Failed to parse saved grid evaluation at %s: %s", path, exc)
        return None


def _fuse_probs(
    vit_probs: Dict[str, float],
    clip_probs: Dict[str, float],
    vit_weight: float,
    clip_weight: float,
) -> Dict[str, float]:
    all_keys = set(list(vit_probs.keys()) + list(clip_probs.keys()))
    return {
        k: round(vit_weight * vit_probs.get(k, 0.0) + clip_weight * clip_probs.get(k, 0.0), 6)
        for k in all_keys
    }


def _macro_f1(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute macro-averaged F1 without scikit-learn."""
    classes = sorted(set(ground_truths))
    f1_per_class = []
    for cls in classes:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != cls and g == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)
    return float(sum(f1_per_class) / len(f1_per_class)) if f1_per_class else 0.0


def _accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    if not ground_truths:
        return 0.0
    return sum(1 for p, g in zip(predictions, ground_truths) if p == g) / len(ground_truths)


def _load_validation_data(val_dir: Path) -> List[Tuple[Any, str]]:
    """
    Load PIL images and ground-truth labels from a directory structure:
        val_dir/
            cardboard/img1.jpg ...
            glass/img2.jpg ...
            ...

    Returns list of (pil_image, label_str) tuples.
    Raises FileNotFoundError if val_dir does not exist.
    """
    from PIL import Image  # local import to keep module importable without PIL

    if not val_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}. "
            "Place labelled images under backend/data/val/<class_name>/ to run optimization."
        )

    samples: List[Tuple[Any, str]] = []
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for class_dir in sorted(val_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.lower()
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in image_exts:
                try:
                    img = Image.open(img_path).convert("RGB")
                    samples.append((img, label))
                except Exception as exc:
                    logger.warning("Could not load %s: %s", img_path, exc)

    if not samples:
        raise ValueError(f"No valid images found in {val_dir}.")

    return samples


async def run_grid_search(
    registry,
    val_dir: Optional[Path] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run ViT/CLIP weight grid search over the validation set.

    Args:
        registry: model registry with vit_model, vit_processor, clip_model, clip_processor.
        val_dir: path to validation data root (class-subdirectory layout).
                 Defaults to backend/data/val/.
        progress_callback: optional async callable(step, total) for progress updates.

    Returns:
        dict with keys:
            best_vit_weight, best_clip_weight, best_macro_f1, best_accuracy,
            baseline_macro_f1, baseline_accuracy,
            all_results (list of weight/metric dicts)
    """
    import asyncio
    from app.ml.vit_model import predict_vit
    from app.ml.clip_model_v2 import predict_clip_v2

    if val_dir is None:
        # Try candidate locations (strictly validation set, never test set)
        candidates = [
            _BACKEND_DIR / "data" / "val",
            Path("backend") / "data" / "val",
            Path("data") / "val",
        ]
        val_dir = next((p for p in candidates if p.exists()), candidates[0])

    logger.info("Loading validation data from %s", val_dir)
    samples = _load_validation_data(val_dir)
    logger.info("Loaded %d validation samples.", len(samples))

    # Pre-compute ViT + CLIP probabilities for each sample (avoid re-inference per grid step)
    logger.info("Running ViT + CLIP inference on %d samples...", len(samples))
    vit_prob_list: List[Dict[str, float]] = []
    clip_prob_list: List[Dict[str, float]] = []
    ground_truths: List[str] = []

    total_steps = len(samples)
    for i, (pil_img, label) in enumerate(samples):
        # Yield control periodically so we don't block the event loop
        if i % 10 == 0:
            await asyncio.sleep(0)
            if progress_callback:
                await progress_callback(i, total_steps)

        try:
            vp = predict_vit(registry.vit_model, registry.vit_processor, pil_img)
        except Exception:
            vp = {}

        cp: Dict[str, float] = {}
        if registry.clip_loaded:
            try:
                cr = predict_clip_v2(registry.clip_model, registry.clip_processor, pil_img)
                cp = cr.probabilities
            except Exception:
                cp = vp.copy()
        else:
            cp = vp.copy()

        vit_prob_list.append(vp)
        clip_prob_list.append(cp)
        ground_truths.append(label)

    logger.info("Inference complete. Running weight grid search...")

    all_results: List[Dict[str, Any]] = []
    best_vit_w = DEFAULT_VIT_WEIGHT
    best_clip_w = DEFAULT_CLIP_WEIGHT
    best_f1 = -1.0
    best_acc = 0.0
    baseline_f1 = 0.0
    baseline_acc = 0.0

    grid_pairs = [(v, round(1.0 - v, 2)) for v in WEIGHT_GRID]

    for step_i, (vw, cw) in enumerate(grid_pairs):
        await asyncio.sleep(0)

        preds = []
        for vp, cp in zip(vit_prob_list, clip_prob_list):
            fused = _fuse_probs(vp, cp, vw, cw)
            pred = max(fused, key=fused.get) if fused else "trash"
            preds.append(pred)

        f1 = _macro_f1(preds, ground_truths)
        acc = _accuracy(preds, ground_truths)

        result_entry = {
            "vit_weight": vw,
            "clip_weight": cw,
            "macro_f1": round(f1, 4),
            "accuracy": round(acc, 4),
        }
        all_results.append(result_entry)

        # Capture baseline (70/30)
        if abs(vw - DEFAULT_VIT_WEIGHT) < 0.001:
            baseline_f1 = f1
            baseline_acc = acc

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_vit_w = vw
            best_clip_w = cw

    logger.info(
        "Grid search complete. Best: ViT=%.2f CLIP=%.2f F1=%.4f (baseline 70/30: F1=%.4f)",
        best_vit_w, best_clip_w, best_f1, baseline_f1,
    )

    return {
        "best_vit_weight": best_vit_w,
        "best_clip_weight": best_clip_w,
        "best_macro_f1": round(best_f1, 4),
        "best_accuracy": round(best_acc, 4),
        "baseline_vit_weight": DEFAULT_VIT_WEIGHT,
        "baseline_clip_weight": DEFAULT_CLIP_WEIGHT,
        "baseline_macro_f1": round(baseline_f1, 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "num_samples": len(samples),
        "all_results": all_results,
    }
