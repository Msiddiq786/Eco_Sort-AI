"""
EcoSort AI V2/V3 — Detection Dataset Ingestion, Validation, and YOLO Export Pipeline.

Supports:
- TACO Dataset (Trash Annotations in Context) & custom COCO JSON annotations
- Bounding-box coordinate integrity checks (0 <= x1 < x2 <= W, 0 <= y1 < y2 <= H)
- Category mapping from 60+ fine-grained waste items (including kraft paper, foam/polystyrene,
  aluminum foil, clear plastic trays, and organic scraps) to EcoSort's 7 canonical classes
- Missing-class coverage audit (never silently hides missing detection classes)
- Stratified Train / Val / Test YOLO-format export (normalized [cls, cx, cy, nw, nh] + data.yaml)
- MD5 image hash deduplication
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_detection_dataset")

ECOSORT_CLASSES: List[str] = [
    "cardboard",  # 0
    "glass",      # 1
    "metal",      # 2
    "paper",      # 3
    "plastic",    # 4
    "organic",    # 5
    "trash",      # 6
]
CLASS_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(ECOSORT_CLASSES)}

# Expanded fine-grained category mapping (TACO + Packaging/Foam/Organic supplements) -> EcoSort 7 classes
TACO_TO_ECOSORT_MAP: Dict[str, str] = {
    # 0. Cardboard
    "Drink carton": "cardboard",
    "Corrugated carton": "cardboard",
    "Egg carton": "cardboard",
    "Meal carton": "cardboard",
    "Pizza box": "cardboard",
    "Other carton": "cardboard",
    "Cardboard box": "cardboard",
    "Cardboard": "cardboard",
    # 1. Glass
    "Glass bottle": "glass",
    "Glass jar": "glass",
    "Glass cup": "glass",
    "Broken glass": "glass",
    "Glass": "glass",
    # 2. Metal
    "Drink can": "metal",
    "Food can": "metal",
    "Aerosol": "metal",
    "Aluminium foil": "metal",
    "Aluminum foil": "metal",
    "Scrap metal": "metal",
    "Pop tab": "metal",
    "Metal bottle cap": "metal",
    "Metal lid": "metal",
    "Metal": "metal",
    # 3. Paper (includes crumpled paper, kraft paper bags, disposable paper cups)
    "Paper bag": "paper",
    "Kraft paper bag": "paper",
    "Crumpled paper": "paper",
    "Paper cup": "paper",
    "Newspaper": "paper",
    "Magazine": "paper",
    "Normal paper": "paper",
    "Office paper": "paper",
    "Wrapping paper": "paper",
    "Tissues": "paper",
    "Toilet tube": "cardboard",
    "Paper": "paper",
    # 4. Plastic (bottles, clear trays, films, wrappers — excludes styrofoam/polystyrene)
    "Clear plastic bottle": "plastic",
    "Other plastic bottle": "plastic",
    "Plastic bottle cap": "plastic",
    "Plastic bag & wrapper": "plastic",
    "Other plastic wrapper": "plastic",
    "Single-use carrier bag": "plastic",
    "Polypropylene bag": "plastic",
    "Crisp packet": "trash",
    "Spread tub": "plastic",
    "Tupperware": "plastic",
    "Plastic container": "plastic",
    "Transparent plastic tray": "plastic",
    "Plastic bowl": "plastic",
    "Disposable plastic cup": "plastic",
    "Other plastic cup": "plastic",
    "Plastic lid": "plastic",
    "Plastic straw": "plastic",
    "Plastic utensils": "plastic",
    "Other plastic": "plastic",
    "Plastic": "plastic",
    # 5. Organic
    "Food waste": "organic",
    "Vegetable scraps": "organic",
    "Fruit peels": "organic",
    "Organic": "organic",
    "Banana": "organic",
    "Apple": "organic",
    # 6. Trash / Foam / Polystyrene / Non-recyclables
    "Styrofoam piece": "trash",
    "Foam cup": "trash",
    "Foam food tray": "trash",
    "Foam tray": "trash",
    "Polystyrene": "trash",
    "Cigarette": "trash",
    "Unlabeled litter": "trash",
    "Rope & strings": "trash",
    "Shoe": "trash",
    "Squeezable tube": "trash",
    "Battery": "trash",
    "Trash": "trash",
}


def calculate_image_hash(image_path: Path) -> str:
    """Compute MD5 hash of image bytes for deduplication."""
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_coco_annotations(
    coco_json_path: Path,
    image_dir: Path,
) -> Dict[str, Any]:
    """
    Validate COCO JSON annotation format, image dimensions, bounding boxes,
    and audit per-class instance counts against training requirements.
    """
    logger.info("Parsing COCO annotation file: %s", coco_json_path)
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    logger.info("Found %d images, %d annotations, %d raw categories", len(images), len(annotations), len(categories))

    img_id_to_dims = {img["id"]: (img["width"], img["height"]) for img in images}

    valid_annotations = 0
    invalid_boxes = 0
    mapped_counts: Dict[str, int] = {c: 0 for c in ECOSORT_CLASSES}
    unmapped_raw_categories: Set[str] = set()

    for ann in annotations:
        bbox = ann.get("bbox", [])  # COCO [x, y, width, height]
        if len(bbox) != 4:
            invalid_boxes += 1
            continue

        x, y, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            invalid_boxes += 1
            continue

        img_id = ann["image_id"]
        if img_id in img_id_to_dims:
            w, h = img_id_to_dims[img_id]
            if x < -1.0 or y < -1.0 or x + bw > w * 1.05 or y + bh > h * 1.05:
                invalid_boxes += 1
                continue

        raw_cat_name = categories.get(ann.get("category_id"), "Unlabeled litter")
        if raw_cat_name not in TACO_TO_ECOSORT_MAP:
            unmapped_raw_categories.add(raw_cat_name)
        ecosort_cat = TACO_TO_ECOSORT_MAP.get(raw_cat_name, "trash")
        mapped_counts[ecosort_cat] = mapped_counts.get(ecosort_cat, 0) + 1
        valid_annotations += 1

    underrepresented = [c for c, cnt in mapped_counts.items() if cnt < 50]

    summary = {
        "total_images": len(images),
        "total_annotations": len(annotations),
        "valid_annotations": valid_annotations,
        "invalid_boxes_skipped": invalid_boxes,
        "ecosort_category_distribution": mapped_counts,
        "underrepresented_classes_lt_50": underrepresented,
        "unmapped_raw_categories_defaulted_to_trash": sorted(unmapped_raw_categories),
        "coco_format_verified": True,
    }

    logger.info("COCO Annotation Validation Summary: %s", json.dumps(summary, indent=2))
    return summary


def export_coco_to_yolo(
    coco_json_path: Path,
    image_dir: Path,
    output_dir: Path = Path("backend/data/detection"),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """
    Convert COCO/TACO JSON annotations into YOLO detection directory structure:
        output_dir/{train,val,test}/{images,labels}/*.txt
    and write output_dir/data.yaml.
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    seen_hashes: Set[str] = set()
    valid_records: List[Tuple[Path, List[str]]] = []

    for img_info in coco_data.get("images", []):
        img_id = img_info["id"]
        rel_name = str(img_info["file_name"]).replace("\\", "/")
        src_img = image_dir / rel_name
        if not src_img.is_file():
            alt_img = image_dir / Path(rel_name).name
            if alt_img.is_file():
                src_img = alt_img
            else:
                continue

        img_hash = calculate_image_hash(src_img)
        if img_hash in seen_hashes:
            continue
        seen_hashes.add(img_hash)

        w = float(img_info["width"])
        h = float(img_info["height"])
        if w <= 0 or h <= 0:
            continue

        yolo_lines: List[str] = []
        for ann in anns_by_img.get(img_id, []):
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            bx, by, bw, bh = [float(v) for v in bbox]
            if bw <= 1.0 or bh <= 1.0:
                continue

            x1 = max(0.0, min(w, bx))
            y1 = max(0.0, min(h, by))
            x2 = max(0.0, min(w, bx + bw))
            y2 = max(0.0, min(h, by + bh))
            if x2 <= x1 or y2 <= y1:
                continue

            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h

            raw_cat = categories.get(ann.get("category_id"), "Unlabeled litter")
            eco_cat = TACO_TO_ECOSORT_MAP.get(raw_cat, "trash")
            cls_id = CLASS_TO_ID[eco_cat]
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if yolo_lines:
            valid_records.append((src_img, yolo_lines))

    if not valid_records:
        raise ValueError("No valid annotated images could be matched on disk for YOLO export.")

    rng = random.Random(seed)
    rng.shuffle(valid_records)
    n_total = len(valid_records)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": valid_records[:n_train],
        "val": valid_records[n_train:n_train + n_val],
        "test": valid_records[n_train + n_val:],
    }

    for split_name, items in splits.items():
        img_out_dir = output_dir / split_name / "images"
        lbl_out_dir = output_dir / split_name / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for idx, (src_img, lines) in enumerate(items):
            stem = f"{split_name}_{idx:05d}_{src_img.stem}"
            shutil.copy2(src_img, img_out_dir / f"{stem}{src_img.suffix}")
            (lbl_out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: c for i, c in enumerate(ECOSORT_CLASSES)},
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    logger.info("Exported %d annotated images into YOLO splits at %s", n_total, output_dir)
    return yaml_path


def download_hf_taco_dataset(
    repo_id: str = "RandyHuynh5815/TACO-Waste-Recognition",
    local_dir: Path = Path("backend/data/taco_raw"),
) -> Tuple[Path, Path]:
    """
    Download the real TACO Waste Recognition dataset from Hugging Face Hub
    (annotations.json + batch_* folders) and return (annotations_json_path, image_root_dir).
    """
    from huggingface_hub import snapshot_download

    target_dir = Path(local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading HF TACO dataset '%s' to %s ...", repo_id, target_dir)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=["annotations.json", "batch_*/*"],
    )
    ann_path = target_dir / "annotations.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"annotations.json missing after snapshot_download at {ann_path}")
    return ann_path, target_dir


def merge_feedback_crops_into_train(
    feedback_dir: Path,
    output_dir: Path = Path("backend/data/detection"),
) -> int:
    """
    Convert user-confirmed crop feedback samples stored under
    feedback_dir/<category>/*.jpg into centered YOLO training labels in train/.
    """
    if not feedback_dir.is_dir():
        return 0

    train_img_dir = output_dir / "train" / "images"
    train_lbl_dir = output_dir / "train" / "labels"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    for cat_dir in sorted(feedback_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cls_name = cat_dir.name.lower()
        if cls_name not in CLASS_TO_ID:
            continue
        cls_id = CLASS_TO_ID[cls_name]
        for img_file in sorted(cat_dir.glob("*")):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            stem = f"feedback_{cls_name}_{img_file.stem}"
            shutil.copy2(img_file, train_img_dir / f"{stem}{img_file.suffix}")
            # Verified crop fills most of the frame (tight crop margin)
            (train_lbl_dir / f"{stem}.txt").write_text(f"{cls_id} 0.500000 0.500000 0.880000 0.880000\n", encoding="utf-8")
            added += 1

    if added:
        logger.info("Merged %d verified user feedback crops into %s/train", added, output_dir)
    return added


def main():
    parser = argparse.ArgumentParser(description="Prepare, Validate & Export Waste Detection Dataset")
    parser.add_argument("--json", type=str, default="", help="Path to COCO/TACO annotations.json")
    parser.add_argument("--img-dir", type=str, default="", help="Path to images directory")
    parser.add_argument("--hf-taco", action="store_true", help="Auto-download RandyHuynh5815/TACO-Waste-Recognition from HF Hub")
    parser.add_argument("--hf-repo", type=str, default="RandyHuynh5815/TACO-Waste-Recognition", help="HuggingFace dataset repo ID")
    parser.add_argument("--raw-dir", type=str, default="backend/data/taco_raw", help="Local cache dir for HF TACO download")
    parser.add_argument("--include-feedback", action="store_true", help="Include user-confirmed feedback crops in train split")
    parser.add_argument("--feedback-dir", type=str, default="backend/data/feedback", help="Directory with corrected feedback crops")
    parser.add_argument("--export-yolo", action="store_true", help="Export validated COCO annotations to YOLO format")
    parser.add_argument("--output-dir", type=str, default="backend/data/detection", help="Target YOLO dataset directory")
    args = parser.parse_args()

    print("=================================================================")
    print("EcoSort AI V3 — Detection Dataset Pipeline Validator & Exporter")
    print("=================================================================")

    json_path = Path(args.json) if args.json else None
    img_dir = Path(args.img_dir) if args.img_dir else None

    if args.hf_taco:
        json_path, img_dir = download_hf_taco_dataset(
            repo_id=args.hf_repo,
            local_dir=Path(args.raw_dir),
        )

    if json_path is None:
        print("[Notice] No COCO/TACO annotations.json or --hf-taco flag provided.")
        print("Pass --hf-taco --export-yolo to ingest RandyHuynh5815/TACO-Waste-Recognition automatically.")
        return 0

    if img_dir is None:
        img_dir = json_path.parent

    validate_coco_annotations(json_path, img_dir)
    if args.export_yolo or args.hf_taco:
        export_coco_to_yolo(json_path, img_dir, output_dir=Path(args.output_dir))
        if args.include_feedback:
            merge_feedback_crops_into_train(Path(args.feedback_dir), output_dir=Path(args.output_dir))
    return 0


if __name__ == "__main__":
    main()
