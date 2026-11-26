"""
EcoSort AI V2/V3 — Dedicated Waste Object Detector Training Pipeline.

Architecture: YOLOv8 Object Detection on TACO / Multi-Object Waste Scenes.
Supports:
- 7 categories (cardboard, glass, metal, paper, plastic, organic, trash)
- Training ONLY against a verified YOLO data.yaml (never overwrites ViT V1/V2 checkpoints)
- Automatic saving of best weights to backend/models/detection_v2/weights/best.pt
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_detector_v2")

_BACKEND_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = _BACKEND_DIR / "models" / "detection_v2" / "config.yaml"
DEFAULT_DATA_YAML = _BACKEND_DIR / "data" / "detection" / "data.yaml"
DEST_BEST_WEIGHTS = _BACKEND_DIR / "models" / "detection_v2" / "weights" / "best.pt"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Detector config not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train V2/V3 7-Class Waste Object Detector")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Path to YOLO data.yaml")
    parser.add_argument("--base-weights", type=str, default="yolov8s.pt", help="Pretrained base checkpoint")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--min-map50", type=float, default=0.30, help="Minimum validation mAP@50 required before overwriting best.pt")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without training")
    args = parser.parse_args()

    cfg = load_config()
    data_yaml = Path(args.data)

    import torch
    has_cuda = torch.cuda.is_available()

    print("=================================================================")
    print("EcoSort AI V3 — Waste Object Detector Training Pipeline")
    print(f"Architecture:      {cfg['model']['architecture']}")
    print(f"Target Checkpoint: {DEST_BEST_WEIGHTS}")
    print(f"Dataset YAML:      {data_yaml} (exists={data_yaml.is_file()})")
    print(f"Hardware:          {'CUDA GPU' if has_cuda else 'CPU-only (long run warning)'}")
    print(f"Config Status:     {cfg['status']['detector_status']}")
    print("=================================================================")

    if args.dry_run:
        print("[Dry Run] Detector training pipeline verified cleanly.")
        return 0

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"YOLO detection dataset config not found at {data_yaml}. "
            "Run prepare_detection_dataset.py --hf-taco --export-yolo first."
        )

    if not has_cuda and args.epochs > 10:
        logger.warning(
            "Running %d epochs on CPU will take several hours. Consider GPU training or lower epochs.",
            args.epochs,
        )

    from ultralytics import YOLO

    model = YOLO(args.base_weights)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        project=str(_BACKEND_DIR / "models" / "detection_v2" / "runs"),
        name="ecosort_det_v3",
        exist_ok=True,
    )

    val_map50 = float(getattr(results, "results_dict", {}).get("metrics/mAP50(B)", 0.0) or 0.0)
    save_dir = Path(getattr(results, "save_dir", _BACKEND_DIR / "models" / "detection_v2" / "runs" / "ecosort_det_v3"))
    trained_best = save_dir / "weights" / "best.pt"
    if trained_best.is_file() and val_map50 >= args.min_map50:
        DEST_BEST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trained_best, DEST_BEST_WEIGHTS)
        logger.info("Copied trained best weights (mAP50=%.4f) to %s", val_map50, DEST_BEST_WEIGHTS)
    elif trained_best.is_file():
        logger.warning(
            "Skipping overwrite of %s because val mAP50 (%.4f) < min-map50 (%.4f).",
            DEST_BEST_WEIGHTS,
            val_map50,
            args.min_map50,
        )
    return 0


if __name__ == "__main__":
    main()
