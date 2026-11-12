"""
Phase I: Real Image Testing Procedure for EcoSort AI.

Tests 6 standard waste categories (from held-out test split) + Out-of-Distribution (OOD) images.
Reports:
  - Ground-truth category
  - ViT prediction
  - CLIP prediction
  - Ensemble prediction
  - Raw confidence
  - Calibrated confidence (Temperature Scaling)
  - OOD Warning Status
  - Correctness
"""
from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.ml.calibration import TemperatureCalibrator, detect_out_of_distribution
from app.ml.clip_model import load_clip, predict_clip
from app.ml.ensemble import fuse
from app.ml.preprocessor import load_image_pil
from app.ml.vit_model import load_vit, predict_vit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("test_real_images")


def create_synthetic_vegetable_image() -> Image.Image:
    """Generate a realistic green vegetable / organic food image (OOD sample)."""
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))  # Forest green
    draw = ImageDraw.Draw(img)
    # Draw organic leafy patterns and carrots/tomatoes
    draw.ellipse([30, 30, 190, 190], fill=(46, 117, 34), outline=(20, 80, 20))
    draw.ellipse([60, 50, 140, 150], fill=(220, 80, 20))    # orange carrot/produce
    draw.ellipse([90, 80, 170, 160], fill=(180, 30, 30))    # red tomato/produce
    return img


def create_synthetic_battery_image() -> Image.Image:
    """Generate a battery / electronic waste image (OOD sample)."""
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([70, 40, 154, 184], fill=(50, 50, 50), outline=(20, 20, 20))
    draw.rectangle([95, 25, 129, 40], fill=(180, 180, 180)) # battery terminal
    draw.text((85, 95), "LI-ION 9V", fill=(255, 215, 0))
    return img


def run_real_image_test_suite() -> List[Dict[str, Any]]:
    # 1. Load models
    vit_model, vit_processor = load_vit()
    clip_model, clip_processor = load_clip()

    calibrator = TemperatureCalibrator(temperature=0.35)

    def get_test_image(cls_name: str) -> Path:
        cls_dir = _BACKEND_DIR / "data" / "test" / cls_name
        files = sorted(list(cls_dir.glob("*.jpg")))
        if not files:
            raise FileNotFoundError(f"No .jpg files found in {cls_dir}")
        return files[0]

    test_cases = [
        {"name": "Plastic Water Bottle", "ground_truth": "plastic", "source": get_test_image("plastic"), "is_ood": False},
        {"name": "Paper Sheet / Document", "ground_truth": "paper", "source": get_test_image("paper"), "is_ood": False},
        {"name": "Metal Beverage Can", "ground_truth": "metal", "source": get_test_image("metal"), "is_ood": False},
        {"name": "Cardboard Box", "ground_truth": "cardboard", "source": get_test_image("cardboard"), "is_ood": False},
        {"name": "Glass Beverage Bottle", "ground_truth": "glass", "source": get_test_image("glass"), "is_ood": False},
        {"name": "Household Trash / Waste", "ground_truth": "trash", "source": get_test_image("trash"), "is_ood": False},
        {"name": "Fresh Organic Vegetable", "ground_truth": "out_of_distribution", "image_generator": create_synthetic_vegetable_image, "is_ood": True},
        {"name": "Lithium Battery / E-Waste", "ground_truth": "out_of_distribution", "image_generator": create_synthetic_battery_image, "is_ood": True},
    ]

    results = []

    print("\n" + "=" * 95)
    print(f"{'Image Description':<25} | {'Ground Truth':<12} | {'ViT':<10} | {'CLIP':<10} | {'Ensemble':<10} | {'Raw Conf':<9} | {'Cal Conf':<9} | {'Status':<12}")
    print("-" * 95)

    for tc in test_cases:
        if "source" in tc:
            pil_img = load_image_pil(tc["source"])
        else:
            pil_img = tc["image_generator"]()

        vit_probs = predict_vit(vit_model, vit_processor, pil_img)
        clip_res = predict_clip(clip_model, clip_processor, pil_img)
        clip_probs = clip_res.probabilities

        fused = fuse(vit_probs, clip_probs, vit_weight=0.70, clip_weight=0.30, temperature=0.35)

        is_correct = (fused.predicted_category == tc["ground_truth"]) if not tc["is_ood"] else fused.is_out_of_distribution

        record = {
            "image_name": tc["name"],
            "ground_truth": tc["ground_truth"],
            "vit_prediction": fused.vit_prediction,
            "clip_prediction": fused.clip_prediction,
            "ensemble_prediction": fused.predicted_category,
            "raw_confidence": fused.confidence,
            "calibrated_confidence": fused.calibrated_confidence,
            "confidence_status": fused.confidence_status,
            "is_out_of_distribution": fused.is_out_of_distribution,
            "ood_warning": fused.ood_warning,
            "is_correct": is_correct,
        }
        results.append(record)

        status_str = "OOD FLAGGED" if fused.is_out_of_distribution else ("CORRECT" if is_correct else "INCORRECT")
        print(f"{tc['name']:<25} | {tc['ground_truth']:<12} | {fused.vit_prediction:<10} | {fused.clip_prediction:<10} | {fused.predicted_category:<10} | {fused.confidence*100:>7.2f}% | {fused.calibrated_confidence*100:>7.2f}% | {status_str:<12}")

    print("=" * 95)

    out_file = _BACKEND_DIR / "evaluation" / "results" / "real_image_test_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved real-world image test results to %s", out_file)
    return results


if __name__ == "__main__":
    run_real_image_test_suite()
