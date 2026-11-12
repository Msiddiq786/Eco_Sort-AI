"""
Dataset preparation, validation, and stratified splitting for EcoSort AI.

Supported classes (strictly 6):
  1. cardboard
  2. glass
  3. metal
  4. paper
  5. plastic
  6. trash

Pipeline:
  1. Validate source directory exists.
  2. Validate that all 6 required waste category directories exist.
  3. Scan each class directory, detecting and filtering unreadable/corrupt files.
  4. Perform deterministic, stratified train / val / test split using configured seed.
  5. Verify zero data leakage across splits.
  6. Copy images into target split directories (train/<class>, val/<class>, test/<class>).
  7. Generate and save a reproducible split summary report in JSON format.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("prepare_dataset")

# ── The official 6 EcoSort AI waste classes ──────────────────────────────────
REQUIRED_CLASSES: tuple[str, ...] = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
)


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file or return defaults."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def is_image_valid(image_path: Path) -> bool:
    """
    Verify that an image file exists, is non-empty, and can be opened
    and decoded by PIL without error.
    """
    if not image_path.is_file() or image_path.stat().st_size == 0:
        return False
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Verify only tests headers; reopen to ensure data stream decodes
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception as exc:
        logger.warning("Corrupt or unreadable image skipped '%s': %s", image_path, exc)
        return False


def validate_raw_dataset(
    raw_dir: Path,
    required_classes: tuple[str, ...] = REQUIRED_CLASSES,
) -> dict[str, list[Path]]:
    """
    Inspect raw directory and ensure:
      1. raw_dir exists.
      2. Every required class folder exists.
      3. Valid images are scanned and corrupt ones discarded.

    Raises:
      FileNotFoundError: if raw_dir does not exist.
      ValueError: if any required class directory is missing or empty.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Dataset raw directory does not exist: '{raw_dir.resolve()}'. "
            "Please ensure the dataset is placed at the specified path."
        )

    # Some datasets extract with a nested container like 'dataset-resized/'
    nested_resized = raw_dir / "dataset-resized"
    effective_dir = nested_resized if nested_resized.exists() else raw_dir

    # Check for missing class directories
    existing_dirs = {
        p.name.lower(): p
        for p in effective_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    }

    missing_classes = [cls for cls in required_classes if cls not in existing_dirs]
    if missing_classes:
        raise ValueError(
            f"Missing required class directories in '{effective_dir.resolve()}': {missing_classes}. "
            f"Expected all 6 classes: {list(required_classes)}. "
            f"Found subdirectories: {list(existing_dirs.keys())}."
        )

    class_images: dict[str, list[Path]] = {}
    total_found = 0
    total_corrupt = 0

    for cls in required_classes:
        cls_folder = existing_dirs[cls]
        valid_paths: list[Path] = []

        for item in sorted(cls_folder.iterdir()):
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                if is_image_valid(item):
                    valid_paths.append(item)
                else:
                    total_corrupt += 1

        if not valid_paths:
            raise ValueError(
                f"Class directory '{cls_folder.resolve()}' contains 0 readable images."
            )

        class_images[cls] = valid_paths
        total_found += len(valid_paths)

    logger.info(
        "Validated raw dataset at '%s': %d valid images across %d classes (%d corrupt rejected).",
        effective_dir,
        total_found,
        len(required_classes),
        total_corrupt,
    )
    for cls in required_classes:
        logger.info("  - %-10s: %d images", cls, len(class_images[cls]))

    return class_images


def split_dataset(
    class_images: Mapping[str, list[Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict[str, list[tuple[Path, str]]]:
    """
    Perform deterministic, stratified splitting into train, val, and test splits.

    Ensures zero data leakage between splits.

    Returns:
      dict with keys 'train', 'val', 'test', each containing a list of (Path, label).
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-4:
        raise ValueError(
            f"Split ratios must sum to 1.0; got {train_ratio} + {val_ratio} + {test_ratio} = {ratio_sum}"
        )

    all_paths: list[Path] = []
    all_labels: list[str] = []

    for cls, paths in class_images.items():
        for p in paths:
            all_paths.append(p)
            all_labels.append(cls)

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=val_test_ratio,
        stratify=all_labels,
        random_state=random_seed,
    )

    # Second split: val vs test
    val_relative_ratio = val_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        train_size=val_relative_ratio,
        stratify=temp_labels,
        random_state=random_seed,
    )

    # ── Strict Data Leakage Verification ─────────────────────────────────────
    train_set = {str(p.resolve()) for p in train_paths}
    val_set = {str(p.resolve()) for p in val_paths}
    test_set = {str(p.resolve()) for p in test_paths}

    leak_train_val = train_set & val_set
    leak_train_test = train_set & test_set
    leak_val_test = val_set & test_set

    if leak_train_val:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED: {len(leak_train_val)} images overlap between train and val!"
        )
    if leak_train_test:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED: {len(leak_train_test)} images overlap between train and test!"
        )
    if leak_val_test:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED: {len(leak_val_test)} images overlap between val and test!"
        )

    total_split_count = len(train_paths) + len(val_paths) + len(test_paths)
    if total_split_count != len(all_paths):
        raise RuntimeError(
            f"Split count mismatch: total input images {len(all_paths)} != sum of splits {total_split_count}"
        )

    logger.info("Data leakage check PASSED: 0 overlapping images across all splits.")
    logger.info("Total images: %d", len(all_paths))
    logger.info("  Train: %d (%.1f%%)", len(train_paths), len(train_paths) / len(all_paths) * 100)
    logger.info("  Val:   %d (%.1f%%)", len(val_paths), len(val_paths) / len(all_paths) * 100)
    logger.info("  Test:  %d (%.1f%%)", len(test_paths), len(test_paths) / len(all_paths) * 100)

    return {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
    }


def write_split_directories(
    splits: Mapping[str, list[tuple[Path, str]]],
    output_base_dir: Path,
    copy_files: bool = True,
) -> dict[str, dict[str, int]]:
    """
    Populate output_base_dir with:
      train/<class>/
      val/<class>/
      test/<class>/

    Returns distribution dictionary: {split: {class: count}}.
    """
    distribution: dict[str, dict[str, int]] = {}

    for split_name, items in splits.items():
        split_dir = output_base_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_counts: dict[str, int] = {cls: 0 for cls in REQUIRED_CLASSES}

        for src_path, label in items:
            dest_class_dir = split_dir / label
            dest_class_dir.mkdir(parents=True, exist_ok=True)

            if copy_files:
                dest_file = dest_class_dir / src_path.name
                # Avoid unnecessary recopy if already identical
                if not dest_file.exists():
                    shutil.copy2(src_path, dest_file)

            split_counts[label] += 1

        distribution[split_name] = split_counts

    return distribution


def prepare_dataset(
    raw_dir: Path | str,
    output_dir: Path | str,
    config: Mapping[str, Any] | None = None,
    copy_files: bool = True,
) -> dict[str, Any]:
    """
    High-level entrypoint for dataset validation, splitting, directory creation,
    and summary generation.

    Returns the complete split summary dictionary.
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    cfg = config or load_config()
    data_cfg = cfg.get("data", {})

    train_ratio = float(data_cfg.get("train_ratio", 0.70))
    val_ratio = float(data_cfg.get("val_ratio", 0.15))
    test_ratio = float(data_cfg.get("test_ratio", 0.15))
    random_seed = int(data_cfg.get("random_seed", 42))

    # 1. Validate raw dataset
    class_images = validate_raw_dataset(raw_path, required_classes=REQUIRED_CLASSES)

    # 2. Perform stratified split
    splits = split_dataset(
        class_images,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    # 3. Write structured split directories
    distribution = write_split_directories(
        splits,
        output_base_dir=output_path,
        copy_files=copy_files,
    )

    # 4. Generate summary report
    total_images = sum(len(items) for items in splits.values())
    summary: dict[str, Any] = {
        "status": "SUCCESS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "classes": list(REQUIRED_CLASSES),
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "counts": {
            "total": total_images,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "distribution": distribution,
        "leakage_check": "PASSED",
        "raw_directory": str(raw_path.resolve()),
        "output_directory": str(output_path.resolve()),
    }

    # Save summary manifest
    summary_path = output_path / "splits_summary.json"
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved dataset split summary to '%s'", summary_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and split TrashNet waste dataset.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("backend/data/raw"),
        help="Path to raw dataset folder containing 6 class subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/data"),
        help="Output base folder for train/, val/, test/ directories",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Only validate and create summary without copying images",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    prepare_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        config=cfg,
        copy_files=not args.no_copy,
    )


if __name__ == "__main__":
    main()
