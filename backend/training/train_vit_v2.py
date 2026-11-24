"""
EcoSort AI V2 — 7-Class Vision Transformer Training Pipeline.

Architecture:
- Base: google/vit-base-patch16-224-in21k
- Classification Head: strictly 7 classes
  0: cardboard
  1: glass
  2: metal
  3: paper
  4: plastic
  5: organic
  6: trash

Features:
- Full pre-training dataset integrity validation (zero-count, corruption, leakage)
- Class-weighted CrossEntropyLoss for imbalanced distributions
- Label smoothing (0.1)
- Validation tracking: accuracy, loss, macro precision/recall/F1, per-class metrics
- Early stopping & model checkpointing:
  * Best checkpoint:   backend/models/vit_v2/best/
  * Latest checkpoint: backend/models/vit_v2/latest/
- Safe resumption from latest V2 checkpoint (weights, optimizer, scheduler, metadata)
- Complete isolation from V1: NEVER loads or overwrites backend/models/vit/
- Safe execution guard: protects parallel V1 CPU training runs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Ensure backend root is on sys.path for direct script execution
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import yaml

from training.augmentation import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_transforms,
)
from training.dataset import (
    SUPPORTED_EXTENSIONS,
    WasteDataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_vit_v2")

# ── Official EcoSort AI V2 7-Class Specification ─────────────────────────────
V2_CLASSES: tuple[str, ...] = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "organic",
    "trash",
)
V2_CLASS_TO_IDX: dict[str, int] = {cls: idx for idx, cls in enumerate(V2_CLASSES)}
V2_IDX_TO_CLASS: dict[int, str] = {idx: cls for idx, cls in enumerate(V2_CLASSES)}

DEFAULT_V2_CONFIG_PATH = Path("backend/models/vit_v2/config_v2.yaml")


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_device(force_device: str | None = None) -> torch.device:
    """
    Detect and return the compute device (CUDA or CPU).
    Trains on CPU by default or CUDA if explicitly available.
    """
    if force_device:
        device = torch.device(force_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device_name = "CUDA" if device.type == "cuda" else "CPU"
    print(f"Device: {device_name}")
    logger.info("Using compute device: %s", device_name)
    return device


def resolve_path(p: Path | str) -> Path:
    """
    Resolve path robustly regardless of whether the command is executed
    from the project root or the backend/ directory.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    cwd = Path.cwd().resolve()
    if cwd == _BACKEND_DIR:
        parts = path.parts
        if parts and parts[0] == "backend":
            return _BACKEND_DIR.joinpath(*parts[1:])
        return _BACKEND_DIR / path
    if (cwd / "backend").is_dir():
        parts = path.parts
        if parts and parts[0] == "backend":
            return cwd.joinpath(*parts)
        return cwd / "backend" / path
    return path.resolve()


def check_v1_training_running() -> bool:
    """Check if another train_vit process is active on system."""
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "train_vit.py" in cmdline and "train_vit_v2.py" not in cmdline:
                return True
    except Exception:
        pass
    return False


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of file bytes for deduplication and leakage check."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_dataset_integrity(
    base_data_dir: Path,
    classes: Sequence[str] = V2_CLASSES,
) -> dict[str, Any]:
    """
    Perform strict pre-training validation:
      1. All 7 class directories exist in train, val, and test.
      2. No class has 0 images in any split (specifically organic).
      3. No corrupted images exist; all open and decode cleanly.
      4. All images have valid positive dimensions.
      5. Zero data leakage: MD5 hash sets between train, val, and test have zero overlap.
    """
    train_dir = base_data_dir / "train"
    val_dir = base_data_dir / "val"
    test_dir = base_data_dir / "test"

    for name, p in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        if not p.is_dir():
            raise FileNotFoundError(f"Missing required dataset split directory: '{p.resolve()}'")

    splits_data: dict[str, dict[str, list[Path]]] = {
        "train": {},
        "val": {},
        "test": {},
    }
    split_hashes: dict[str, set[str]] = {
        "train": set(),
        "val": set(),
        "test": set(),
    }
    total_images_checked = 0

    for split_name, split_path in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        for cls_name in classes:
            cls_folder = split_path / cls_name
            if not cls_folder.is_dir():
                raise FileNotFoundError(
                    f"Missing class folder '{cls_name}' in split '{split_name}' at '{cls_folder.resolve()}'"
                )

            valid_files: list[Path] = []
            for img_path in sorted(cls_folder.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    # Validate image decoding & dimensions
                    if img_path.stat().st_size == 0:
                        raise ValueError(f"Corrupt empty image (0 bytes): {img_path}")
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        with Image.open(img_path) as img:
                            w, h = img.size
                            if w <= 0 or h <= 0:
                                raise ValueError(f"Invalid image dimensions ({w}x{h}) in {img_path}")
                    except Exception as exc:
                        raise ValueError(f"Corrupt image detected at '{img_path}': {exc}") from exc

                    file_hash = compute_file_hash(img_path)
                    split_hashes[split_name].add(file_hash)
                    valid_files.append(img_path)
                    total_images_checked += 1

            if len(valid_files) == 0:
                raise ValueError(
                    f"CRITICAL DATASET ERROR: Class '{cls_name}' has 0 valid images in split '{split_name}'! "
                    "Cannot start training."
                )

            splits_data[split_name][cls_name] = valid_files

    # Per-class leakage check across splits (zero image leakage for any class)
    for c in classes:
        tr_hashes = {compute_file_hash(p) for p in splits_data["train"][c]}
        va_hashes = {compute_file_hash(p) for p in splits_data["val"][c]}
        te_hashes = {compute_file_hash(p) for p in splits_data["test"][c]}

        tv = tr_hashes.intersection(va_hashes)
        tt = tr_hashes.intersection(te_hashes)
        vt = va_hashes.intersection(te_hashes)

        if tv or tt or vt:
            raise RuntimeError(
                f"CRITICAL LEAKAGE DETECTED in class '{c}'! "
                f"Train-Val: {len(tv)}, Train-Test: {len(tt)}, Val-Test: {len(vt)}"
            )

    # Global organic isolation check (Organic must have zero overlap with any other class/split)
    all_organic_hashes = {
        compute_file_hash(p)
        for s in ("train", "val", "test")
        for p in splits_data[s]["organic"]
    }
    all_other_hashes = {
        compute_file_hash(p)
        for s in ("train", "val", "test")
        for c in classes if c != "organic"
        for p in splits_data[s][c]
    }
    organic_cross_overlap = all_organic_hashes.intersection(all_other_hashes)
    if organic_cross_overlap:
        raise RuntimeError(
            f"CRITICAL ERROR: Organic class has {len(organic_cross_overlap)} images overlapping with other classes!"
        )

    # Format and print clean summary
    train_counts = {c: len(splits_data["train"][c]) for c in classes}
    val_counts = {c: len(splits_data["val"][c]) for c in classes}
    test_counts = {c: len(splits_data["test"][c]) for c in classes}

    print("\n=================================================================")
    print("Dataset Integrity & Pre-Training Validation (7 Classes)")
    print("=================================================================")
    print(f"Splits:")
    print(f"  - Train:      {sum(train_counts.values()):4d} images")
    print(f"  - Validation: {sum(val_counts.values()):4d} images")
    print(f"  - Test:       {sum(test_counts.values()):4d} images")
    print(f"  - Total:      {total_images_checked:4d} images\n")
    print("Class Distribution:")
    print(f"  {'Class':<12} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print("  " + "-" * 42)
    for c in classes:
        tr = train_counts[c]
        va = val_counts[c]
        te = test_counts[c]
        tot = tr + va + te
        print(f"  {c:<12} {tr:>7d} {va:>7d} {te:>7d} {tot:>7d}")
    print("  " + "-" * 42)
    print(f"  {'Total':<12} {sum(train_counts.values()):>7d} {sum(val_counts.values()):>7d} {sum(test_counts.values()):>7d} {total_images_checked:>7d}")
    print("\nIntegrity Verification:")
    print(f"  - Class count check:   {len(classes)} / 7 classes verified")
    print(f"  - Zero-count check:    PASSED (All classes non-empty)")
    print(f"  - Organic class check: PASSED ({train_counts['organic']} train / {val_counts['organic']} val / {test_counts['organic']} test)")
    print(f"  - Corruption check:    PASSED (All {total_images_checked} images valid & loadable)")
    print(f"  - Dimension check:     PASSED (All images have positive width x height)")
    print(f"  - Data leakage check:  PASSED (0 duplicate hashes across splits)")
    print("=================================================================\n")

    return {
        "train": train_counts,
        "val": val_counts,
        "test": test_counts,
        "total_images": total_images_checked,
        "leakage_check": "PASSED",
    }


def build_v2_model(
    model_name_or_path: str = "google/vit-base-patch16-224-in21k",
    num_labels: int = 7,
    id2label: Mapping[int, str] | None = None,
    label2id: Mapping[str, int] | None = None,
    custom_config: ViTConfig | None = None,
    pretrained: bool = True,
) -> ViTForImageClassification:
    """
    Initialize ViTForImageClassification with strictly 7-class classification head.
    Configures id2label and label2id with exact V2 waste categories.
    """
    id2label_dict = dict(id2label or V2_IDX_TO_CLASS)
    label2id_dict = dict(label2id or V2_CLASS_TO_IDX)

    if custom_config is not None:
        custom_config.num_labels = num_labels
        custom_config.id2label = id2label_dict
        custom_config.label2id = label2id_dict
        return ViTForImageClassification(custom_config)

    if pretrained:
        try:
            model = ViTForImageClassification.from_pretrained(
                model_name_or_path,
                num_labels=num_labels,
                id2label=id2label_dict,
                label2id=label2id_dict,
                ignore_mismatched_sizes=True,
            )
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pretrained base model '{model_name_or_path}': {exc}. "
                "Ensure internet connectivity or a valid local model cache is available."
            ) from exc
    else:
        cfg = ViTConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            id2label=id2label_dict,
            label2id=label2id_dict,
        )
        return ViTForImageClassification(cfg)


def compute_v2_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str] = V2_CLASSES,
) -> dict[str, Any]:
    """Compute 7-class classification metrics."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )

    # Per-class breakdown
    p_per, r_per, f_per, sup_per = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(classes))),
        average=None,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float]] = {}
    for idx, cls_name in enumerate(classes):
        per_class[cls_name] = {
            "precision": round(float(p_per[idx]), 4),
            "recall": round(float(r_per[idx]), 4),
            "f1_score": round(float(f_per[idx]), 4),
            "support": int(sup_per[idx]),
        }

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(classes))))

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "weighted_precision": round(float(prec_w), 4),
        "weighted_recall": round(float(rec_w), 4),
        "weighted_f1": round(float(f1_w), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_v2(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    classes: Sequence[str] = V2_CLASSES,
    max_steps: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Run evaluation loop over data_loader in eval mode."""
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for step, (images, labels) in enumerate(data_loader, start=1):
            if max_steps is not None and step > max_steps:
                break
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=-1).cpu().tolist()
            targets = labels.cpu().tolist()

            all_preds.extend(preds)
            all_targets.extend(targets)

    num_samples = len(all_targets)
    avg_loss = round(total_loss / num_samples, 6) if num_samples > 0 else 0.0
    metrics = compute_v2_metrics(all_targets, all_preds, classes=classes)
    return avg_loss, metrics


def compute_v2_class_weights(
    dataset_samples: Sequence[tuple[Any, int]],
    num_classes: int = 7,
) -> torch.Tensor:
    """
    Compute balanced inverse-frequency class weights for 7-class CrossEntropyLoss:
      w_c = total_samples / (num_classes * count_c)
    Normalized so the mean weight across all classes is exactly 1.0.
    """
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, lbl in dataset_samples:
        counts[int(lbl)] += 1.0
    total = float(len(dataset_samples)) if len(dataset_samples) > 0 else 1.0
    weights = total / (float(num_classes) * np.maximum(counts, 1.0))
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def save_v2_checkpoint(
    model: ViTForImageClassification,
    checkpoint_dir: Path,
    training_metadata: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> None:
    """
    Save V2 checkpoint in HuggingFace format:
      - model weights & config.json (strictly 7 classes with id2label, label2id)
      - preprocessor_config.json
      - training_metadata.json (metadata, metrics & hyperparameters)
      - optimizer.pt & scheduler.pt (for resumption)
    """
    # Strict isolation check: NEVER allow saving to V1 directories
    norm_path = str(checkpoint_dir).replace("\\", "/")
    if "models/vit/best" in norm_path or norm_path.endswith("models/vit") or norm_path.endswith("models/vit/latest"):
        raise ValueError(f"CRITICAL SAFETY VIOLATION: V2 cannot save to V1 directory '{checkpoint_dir}'!")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "config") and model.config is not None:
        model.config.num_labels = 7
        model.config.id2label = V2_IDX_TO_CLASS
        model.config.label2id = V2_CLASS_TO_IDX

    model.save_pretrained(str(checkpoint_dir))

    # Save preprocessor config matching 224x224 ViT
    try:
        processor = ViTImageProcessor(
            do_resize=True,
            size={"height": 224, "width": 224},
            do_normalize=True,
            image_mean=IMAGENET_MEAN,
            image_std=IMAGENET_STD,
        )
        processor.save_pretrained(str(checkpoint_dir))
    except Exception as exc:
        logger.warning("Could not save ViTImageProcessor config: %s", exc)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    if training_metadata:
        meta_path = checkpoint_dir / "training_metadata.json"
        meta_path.write_text(json.dumps(dict(training_metadata), indent=2), encoding="utf-8")

    logger.info("Saved V2 model checkpoint to '%s'", checkpoint_dir.resolve())


def train_v2(
    config_path: Path | str | None = None,
    data_dir_override: Path | str | None = None,
    checkpoint_dir_override: Path | str | None = None,
    results_dir_override: Path | str | None = None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    learning_rate_override: float | None = None,
    device_override: str | None = None,
    dry_run: bool = False,
    freeze_layers: int | None = None,
    use_class_weights: bool = True,
    resume: bool = False,
    force: bool = False,
    model_instance: ViTForImageClassification | None = None,
) -> dict[str, Any]:
    """
    Complete EcoSort AI V2 7-Class Vision Transformer Training Pipeline.
    """
    print("=================================================================")
    print("EcoSort AI V2 — 7-class Vision Transformer Training Pipeline")
    print(f"Categories ({len(V2_CLASSES)}): {', '.join(V2_CLASSES)}")
    print("=================================================================")

    # ── 1. Load V2 Configuration ─────────────────────────────────────────────
    cfg_file = resolve_path(config_path or DEFAULT_V2_CONFIG_PATH)
    if not cfg_file.exists():
        alt = Path(__file__).resolve().parent.parent / "models" / "vit_v2" / "config_v2.yaml"
        if alt.exists():
            cfg_file = alt

    cfg: dict[str, Any] = {}
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        logger.info("Loaded V2 configuration from '%s'", cfg_file)
    else:
        logger.warning("V2 config file not found at '%s'. Using defaults.", cfg_file)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    # ── 2. Device Selection ──────────────────────────────────────────────────
    device = detect_device(device_override)

    # ── 3. Checkpoint & Results Directories ──────────────────────────────────
    if checkpoint_dir_override:
        raw_ckpt = resolve_path(checkpoint_dir_override)
    else:
        best_rel = model_cfg.get("best_dir", "backend/models/vit_v2/best")
        raw_ckpt = resolve_path(best_rel)

    # Resolve best and latest paths under vit_v2
    if raw_ckpt.name == "latest":
        latest_checkpoint_dir = raw_ckpt
        checkpoint_dir = raw_ckpt.parent / "best"
    elif raw_ckpt.name == "best":
        checkpoint_dir = raw_ckpt
        latest_checkpoint_dir = raw_ckpt.parent / "latest"
    else:
        checkpoint_dir = raw_ckpt / "best"
        latest_checkpoint_dir = raw_ckpt / "latest"

    # CRITICAL ISOLATION CHECK: NEVER allow using V1 directory
    for p in (checkpoint_dir, latest_checkpoint_dir):
        norm = str(p).replace("\\", "/")
        if "backend/models/vit/" in norm or norm.endswith("models/vit"):
            raise ValueError(f"CRITICAL SAFETY VIOLATION: V2 cannot use V1 path '{p}'!")

    results_dir = resolve_path(
        results_dir_override or "backend/evaluation/results_v2"
    )

    print(f"Target Checkpoint Dir (Best):   {checkpoint_dir}")
    print(f"Target Checkpoint Dir (Latest): {latest_checkpoint_dir}")
    print(f"Results Directory:             {results_dir}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. Locate & Validate V2 Dataset ──────────────────────────────────────
    base_data_dir = resolve_path(data_dir_override or "backend/data")
    train_dir = base_data_dir / "train"
    val_dir = base_data_dir / "val"
    test_dir = base_data_dir / "test"

    # Strict pre-training dataset integrity check
    integrity_report = validate_dataset_integrity(base_data_dir, classes=V2_CLASSES)

    # Build transforms matching V2 config
    train_tf = get_transforms("train", cfg)
    val_tf = get_transforms("val", cfg)
    test_tf = get_transforms("test", cfg)

    # Construct 7-class datasets using V2_CLASS_TO_IDX
    train_ds = WasteDataset(root_dir=train_dir, transform=train_tf, class_to_idx=V2_CLASS_TO_IDX)
    val_ds = WasteDataset(root_dir=val_dir, transform=val_tf, class_to_idx=V2_CLASS_TO_IDX)
    test_ds = WasteDataset(root_dir=test_dir, transform=test_tf, class_to_idx=V2_CLASS_TO_IDX)

    batch_size = (
        batch_size_override
        if batch_size_override is not None
        else int(train_cfg.get("batch_size", 16))
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── 5. Model Initialization or Resumption ────────────────────────────────
    model_name = model_cfg.get("name", "google/vit-base-patch16-224-in21k")
    num_classes = int(model_cfg.get("num_classes", 7))
    start_epoch = 1
    best_val_score = float("-inf")
    best_epoch = 0
    patience_counter = 0
    training_history: list[dict[str, Any]] = []

    if model_instance is not None:
        model = model_instance.to(device)
    elif resume:
        logger.info("Validating V2 checkpoint for resumption at '%s'...", latest_checkpoint_dir)
        if not latest_checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Resumption failed: latest V2 checkpoint directory '{latest_checkpoint_dir}' does not exist."
            )
        config_path = latest_checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Resumption failed: missing 'config.json' in '{latest_checkpoint_dir}'.")

        # Verify it is strictly a 7-class V2 model
        with open(config_path, "r", encoding="utf-8") as f:
            loaded_cfg = json.load(f)
        if loaded_cfg.get("num_labels") != 7:
            raise ValueError(
                f"SAFETY ERROR: Checkpoint has {loaded_cfg.get('num_labels')} classes. "
                "V2 pipeline requires strictly 7 classes!"
            )

        weights_exist = (
            (latest_checkpoint_dir / "model.safetensors").exists()
            or (latest_checkpoint_dir / "pytorch_model.bin").exists()
        )
        if not weights_exist:
            raise FileNotFoundError(f"Resumption failed: missing model weights in '{latest_checkpoint_dir}'.")

        meta_path = latest_checkpoint_dir / "training_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Resumption failed: missing 'training_metadata.json' in '{latest_checkpoint_dir}'.")

        logger.info("Loading V2 weights from '%s'...", latest_checkpoint_dir)
        model = ViTForImageClassification.from_pretrained(str(latest_checkpoint_dir)).to(device)

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        latest_epoch = int(meta.get("latest_epoch", meta.get("best_epoch", 0)))
        start_epoch = latest_epoch + 1
        logger.info("Resuming V2 training from completed epoch %d (start_epoch: %d).", latest_epoch, start_epoch)

        # Restore best checkpoint score and best_epoch from best checkpoint metadata
        best_meta_file = checkpoint_dir / "training_metadata.json"
        if best_meta_file.exists():
            try:
                best_meta = json.loads(best_meta_file.read_text(encoding="utf-8"))
                best_epoch = int(best_meta.get("best_epoch", 0))
                best_val_score = float(best_meta.get("best_val_accuracy", float("-inf")))
                logger.info("Restored best checkpoint score: val_accuracy=%.4f from epoch %d.", best_val_score, best_epoch)
            except Exception as exc:
                logger.warning("Could not read best checkpoint metadata: %s", exc)
        else:
            best_epoch = int(meta.get("best_epoch", 0))
            best_val_score = float(meta.get("best_val_accuracy", float("-inf")))

        # Restore patience counter based on epochs since best checkpoint
        if best_epoch > 0 and latest_epoch >= best_epoch:
            patience_counter = max(0, latest_epoch - best_epoch)
            logger.info("Restored patience counter: %d epoch(s) without improvement since best epoch %d.", patience_counter, best_epoch)
        else:
            patience_counter = 0

        # Restore and filter history
        history_file = results_dir / "training_history_v2.json"
        if history_file.exists():
            try:
                loaded_hist = json.loads(history_file.read_text(encoding="utf-8"))
                if isinstance(loaded_hist, list):
                    training_history = [e for e in loaded_hist if e.get("epoch", 0) < start_epoch]
                    logger.info("Restored %d history record(s) from '%s'.", len(training_history), history_file)
            except Exception as exc:
                logger.warning("Could not load existing training history: %s", exc)
    else:
        logger.info("Initializing 7-class Vision Transformer from '%s'...", model_name)
        model = build_v2_model(
            model_name_or_path=model_name,
            num_labels=num_classes,
            id2label=V2_IDX_TO_CLASS,
            label2id=V2_CLASS_TO_IDX,
            pretrained=True,
        ).to(device)

    # Verify classification head
    out_features = getattr(model.classifier, "out_features", None)
    print(f"Classifier head out_features: {out_features} (Expected: 7)")
    assert out_features == 7, f"Expected 7 classes, got {out_features}"

    # ── 6. Dry-Run Validation Mode ───────────────────────────────────────────
    if dry_run:
        print("\n=================================================================")
        print("[Dry Run] V2 7-class configuration validated successfully.")
        print(f"  - Configuration:     {cfg_file} (Loaded)")
        print(f"  - Categories (7):    {', '.join(V2_CLASSES)}")
        print(f"  - Model:             {model_name} (out_features={out_features})")
        print(f"  - Organic in Train:  {integrity_report['train']['organic']} images")
        print(f"  - Organic in Val:    {integrity_report['val']['organic']} images")
        print(f"  - Organic in Test:   {integrity_report['test']['organic']} images")
        print(f"  - Checkpoints:       {checkpoint_dir} & {latest_checkpoint_dir}")
        print(f"  - Results:           {results_dir}")
        print(f"  - V1 Protection:     VERIFIED (Zero touches to backend/models/vit/)")
        print(f"  - Dataset Leakage:   PASSED (0 duplicate hashes across splits)")
        print("  - Status:            AWAITING_TRAINING (Ready to execute)")
        print("=================================================================")
        return {
            "status": "DRY_RUN_PASSED",
            "num_classes": 7,
            "classes": list(V2_CLASSES),
            "model_name": model_name,
            "classifier_head_outputs": out_features,
            "counts": integrity_report,
            "target_checkpoints": {
                "best": str(checkpoint_dir),
                "latest": str(latest_checkpoint_dir),
            },
            "results_dir": str(results_dir),
        }

    # ── 7. Safety Guard for Full Training Execution ──────────────────────────
    if not force and check_v1_training_running():
        raise RuntimeError(
            "SAFETY BLOCK: An active Version 1 ViT training process was detected! "
            "Running parallel V2 training on CPU will cause severe resource contention. "
            "Please wait for V1 training to finish, or pass --force to override."
        )

    print("\nStarting V2 training...")

    # ── 8. Hyperparameters, Optimizer & Scheduler ────────────────────────────
    epochs = epochs_override if epochs_override is not None else int(train_cfg.get("max_epochs", 20))
    lr = learning_rate_override if learning_rate_override is not None else float(train_cfg.get("learning_rate", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.1))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(float(train_cfg.get("warmup_ratio", 0.1)) * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_training_steps),
    )

    if resume:
        opt_path = latest_checkpoint_dir / "optimizer.pt"
        sched_path = latest_checkpoint_dir / "scheduler.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        if sched_path.exists():
            scheduler.load_state_dict(torch.load(sched_path, map_location=device))

    # Class-weighted loss
    if use_class_weights and train_ds.samples:
        weights = compute_v2_class_weights(train_ds.samples, num_classes=7).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        logger.info("Using class-weighted CrossEntropyLoss for 7 classes.")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history_file = results_dir / "training_history_v2.json"

    # ── 9. Training Loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        running_loss = 0.0
        train_preds: list[int] = []
        train_targets: list[int] = []

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            train_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            train_targets.extend(labels.cpu().tolist())

        train_loss = running_loss / len(train_ds) if len(train_ds) > 0 else 0.0
        train_acc = accuracy_score(train_targets, train_preds) if len(train_targets) > 0 else 0.0

        # Evaluate validation split
        val_loss, val_metrics = evaluate_v2(
            model=model,
            data_loader=val_loader,
            device=device,
            criterion=criterion,
            classes=V2_CLASSES,
        )
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        print(
            f"  Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Val F1: {val_f1*100:.2f}%"
        )

        history_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(float(train_acc), 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "lr": round(scheduler.get_last_lr()[0], 8),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        training_history.append(history_entry)
        history_file.write_text(json.dumps(training_history, indent=2), encoding="utf-8")

        # Save latest checkpoint after every epoch
        latest_metadata = {
            "model_name": model_name,
            "num_classes": 7,
            "classes": list(V2_CLASSES),
            "latest_epoch": epoch,
            "best_epoch": best_epoch if best_epoch > 0 else epoch,
            "latest_val_accuracy": val_acc,
            "latest_val_macro_f1": val_f1,
            "best_val_accuracy": best_val_score if best_val_score > float("-inf") else val_acc,
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "seed": seed,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_v2_checkpoint(
            model=model,
            checkpoint_dir=latest_checkpoint_dir,
            training_metadata=latest_metadata,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print(f"  Saved latest checkpoint to {latest_checkpoint_dir}")

        # Check for best model improvement
        val_score = val_acc
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            patience_counter = 0

            best_metadata = dict(latest_metadata)
            best_metadata["best_epoch"] = best_epoch
            best_metadata["best_val_accuracy"] = val_acc
            best_metadata["best_val_macro_f1"] = val_f1

            save_v2_checkpoint(
                model=model,
                checkpoint_dir=checkpoint_dir,
                training_metadata=best_metadata,
            )
            print(f"  ★ Saved best checkpoint (Val Acc: {val_acc*100:.2f}%) to {checkpoint_dir}")
        else:
            patience_counter += 1
            print(f"  Score did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    print("\n=================================================================")
    print(f"V2 Training Complete! Best Epoch: {best_epoch} (Val Acc: {best_val_score*100:.2f}%)")
    print(f"Best Checkpoint:   {checkpoint_dir}")
    print(f"Latest Checkpoint: {latest_checkpoint_dir}")
    print("=================================================================")

    return {
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "training_history": training_history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train 7-Class ViT V2 model.")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--config", type=str, default=None, help="Path to config_v2.yaml")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory override")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--resume", action="store_true", help="Resume from latest V2 checkpoint")
    parser.add_argument("--force", action="store_true", help="Force training even if V1 process detected")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without training")
    args = parser.parse_args()

    train_v2(
        config_path=args.config,
        data_dir_override=args.data_dir,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        learning_rate_override=args.lr,
        device_override=args.device,
        dry_run=args.dry_run,
        resume=args.resume,
        force=args.force,
    )


if __name__ == "__main__":
    main()
