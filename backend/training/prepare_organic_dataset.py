"""
EcoSort AI V2 — Organic Dataset Ingestion, Validation, and Stratified Splitting.

This script fetches authentic biological / organic waste images from the verified
open waste dataset (omasteam/waste-garbage-management-dataset, 'biological' category),
validates each image against corruption and duplicates, and places them into the
standard EcoSort dataset directory structure:
  - backend/data/raw/organic/   (400 images)
  - backend/data/train/organic/ (280 images, 70%)
  - backend/data/val/organic/   (60 images, 15%)
  - backend/data/test/organic/  (60 images, 15%)

Safety:
  - Does NOT alter, delete, or overwrite any existing V1 images or classes.
  - Prevents data leakage using MD5 hash deduplication.
  - Produces backend/data/splits_summary_v2.json without modifying V1 summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_organic_dataset")

REPO_ID = "omasteam/waste-garbage-management-dataset"
SRC_PREFIX = "biological/"
TARGET_COUNT = 400
RANDOM_SEED = 42

V2_CLASSES = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "organic",
    "trash",
)


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of image bytes for deduplication."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_image_valid_and_convert_rgb(src_path: Path, dst_path: Path) -> bool:
    """
    Validate that the image is uncorrupted, can be decoded, and save as
    clean 3-channel RGB JPEG.
    """
    try:
        with Image.open(src_path) as img:
            img.verify()
        with Image.open(src_path) as img:
            rgb_img = img.convert("RGB")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_img.save(dst_path, format="JPEG", quality=95)
        return True
    except Exception as exc:
        logger.warning("Corrupt or invalid image '%s': %s", src_path, exc)
        return False


def download_single_image(repo_file: str) -> Path:
    """Download single image file from Hugging Face hub."""
    return Path(hf_hub_download(REPO_ID, repo_file, repo_type="dataset"))


def prepare_organic_dataset(
    base_data_dir: Path,
    target_count: int = TARGET_COUNT,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """
    Download, validate, deduplicate, and split organic waste dataset.
    """
    raw_organic_dir = base_data_dir / "raw" / "organic"
    train_organic_dir = base_data_dir / "train" / "organic"
    val_organic_dir = base_data_dir / "val" / "organic"
    test_organic_dir = base_data_dir / "test" / "organic"

    for d in (raw_organic_dir, train_organic_dir, val_organic_dir, test_organic_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Discover biological files from repo
    logger.info("Discovering candidate biological images from '%s'...", REPO_ID)
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    candidate_files = [f for f in all_files if f.startswith(SRC_PREFIX)]
    logger.info("Found %d candidate biological images in repository.", len(candidate_files))

    # Take first candidates up to target with buffer for duplicates/corrupt
    buffer_count = min(len(candidate_files), target_count + 100)
    selected_candidates = candidate_files[:buffer_count]

    # 2. Download concurrently
    logger.info("Downloading candidate images using thread pool...")
    downloaded_paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_file = {
            executor.submit(download_single_image, f): f
            for f in selected_candidates
        }
        for future in as_completed(future_to_file):
            try:
                p = future.result()
                downloaded_paths.append(p)
            except Exception as exc:
                logger.warning("Download failed: %s", exc)

    logger.info("Downloaded %d candidate files. Validating & deduplicating...", len(downloaded_paths))

    # 3. Validate and deduplicate into raw/organic
    valid_raw_files: list[Path] = []
    seen_hashes: set[str] = set()

    for idx, temp_path in enumerate(downloaded_paths, start=1):
        if len(valid_raw_files) >= target_count:
            break

        dst_file = raw_organic_dir / f"organic_{len(valid_raw_files) + 1}.jpg"
        if not is_image_valid_and_convert_rgb(temp_path, dst_file):
            continue

        file_hash = compute_file_hash(dst_file)
        if file_hash in seen_hashes:
            logger.info("Duplicate hash %s skipped: %s", file_hash[:8], dst_file.name)
            dst_file.unlink(missing_ok=True)
            continue

        seen_hashes.add(file_hash)
        valid_raw_files.append(dst_file)

    logger.info(
        "Successfully populated raw/organic with %d unique, validated images.",
        len(valid_raw_files),
    )

    if len(valid_raw_files) < 100:
        raise RuntimeError(
            f"Insufficient valid organic images obtained: {len(valid_raw_files)} < 100."
        )

    # 4. Stratified Split: 70% Train, 15% Val, 15% Test
    indices = list(range(len(valid_raw_files)))
    train_idx, test_val_idx = train_test_split(
        indices, test_size=0.30, random_state=random_seed, shuffle=True
    )
    val_idx, test_idx = train_test_split(
        test_val_idx, test_size=0.50, random_state=random_seed, shuffle=True
    )

    train_files = [valid_raw_files[i] for i in train_idx]
    val_files = [valid_raw_files[i] for i in val_idx]
    test_files = [valid_raw_files[i] for i in test_idx]

    logger.info(
        "Splitting: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        len(train_files),
        (len(train_files) / len(valid_raw_files)) * 100,
        len(val_files),
        (len(val_files) / len(valid_raw_files)) * 100,
        len(test_files),
        (len(test_files) / len(valid_raw_files)) * 100,
    )

    # 5. Copy images into split directories
    def copy_split(files: list[Path], target_dir: Path) -> list[str]:
        target_hashes: list[str] = []
        for src in files:
            dst = target_dir / src.name
            shutil.copy2(src, dst)
            target_hashes.append(compute_file_hash(dst))
        return target_hashes

    train_hashes = copy_split(train_files, train_organic_dir)
    val_hashes = copy_split(val_files, val_organic_dir)
    test_hashes = copy_split(test_files, test_organic_dir)

    # 6. Strict Leakage Verification
    train_set = set(train_hashes)
    val_set = set(val_hashes)
    test_set = set(test_hashes)

    leakage_train_val = train_set.intersection(val_set)
    leakage_train_test = train_set.intersection(test_set)
    leakage_val_test = val_set.intersection(test_set)

    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise RuntimeError(
            f"LEAKAGE DETECTED in organic split! "
            f"Train-Val: {len(leakage_train_val)}, "
            f"Train-Test: {len(leakage_train_test)}, "
            f"Val-Test: {len(leakage_val_test)}"
        )
    logger.info("Data leakage check PASSED: Zero overlap between Train, Val, and Test splits.")

    # 7. Generate Full V2 Splits Summary Report
    summary = {
        "status": "SUCCESS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "classes": list(V2_CLASSES),
        "split_ratios": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
        },
        "counts": {
            "total_organic_raw": len(valid_raw_files),
            "train_organic": len(train_files),
            "val_organic": len(val_files),
            "test_organic": len(test_files),
        },
        "distribution": {
            "train": {
                cls: len(list((base_data_dir / "train" / cls).glob("*.*")))
                for cls in V2_CLASSES
                if (base_data_dir / "train" / cls).exists()
            },
            "val": {
                cls: len(list((base_data_dir / "val" / cls).glob("*.*")))
                for cls in V2_CLASSES
                if (base_data_dir / "val" / cls).exists()
            },
            "test": {
                cls: len(list((base_data_dir / "test" / cls).glob("*.*")))
                for cls in V2_CLASSES
                if (base_data_dir / "test" / cls).exists()
            },
        },
        "leakage_check": "PASSED",
        "raw_directory": str(base_data_dir / "raw"),
        "output_directory": str(base_data_dir),
    }

    summary_file = base_data_dir / "splits_summary_v2.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote V2 dataset splits summary to '%s'", summary_file)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare Organic dataset for EcoSort AI V2.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="backend/data",
        help="Path to backend/data directory",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=TARGET_COUNT,
        help="Target number of raw organic images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for splitting",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    print("=================================================================")
    print("EcoSort AI V2 — Organic Dataset Ingestion & Validation Pipeline")
    print(f"Target Directory: {data_dir}")
    print(f"Target Count:     {args.count} images")
    print(f"Random Seed:      {args.seed}")
    print("=================================================================")

    summary = prepare_organic_dataset(
        base_data_dir=data_dir,
        target_count=args.count,
        random_seed=args.seed,
    )

    print("\nDataset Preparation Complete:")
    print(json.dumps(summary["distribution"], indent=2))
    print(f"Leakage Check: {summary['leakage_check']}")
    print("=================================================================")


if __name__ == "__main__":
    main()
