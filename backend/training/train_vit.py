"""
Vision Transformer (ViT) fine-tuning pipeline for EcoSort AI.

Base model: google/vit-base-patch16-224-in21k
Target classes (strictly 6):
  0: cardboard
  1: glass
  2: metal
  3: paper
  4: plastic
  5: trash

Pipeline:
  1. Detect hardware (CUDA vs CPU) and print clearly.
  2. Load configuration from backend/training/config.yaml.
  3. Validate that dataset split directories exist and contain images.
  4. Build model and adapt classification head to exactly 6 classes with id2label/label2id.
  5. Train with AdamW optimizer, learning-rate scheduler, and early stopping.
  6. Track train_loss, val_loss, accuracy, macro precision, recall, and F1 score.
  7. Checkpoint the best model to backend/models/vit/best/ using HuggingFace save_pretrained().
  8. Run evaluation on test set and write results to backend/evaluation/results/.
"""
from __future__ import annotations

import argparse
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
from torch import nn
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

from training.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    WASTE_CLASSES,
    WasteDataset,
    create_dataloaders,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_vit")

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


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
    Prints clearly: Device: CUDA or Device: CPU.
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


def build_model(
    model_name_or_path: str = "google/vit-base-patch16-224-in21k",
    num_labels: int = 6,
    id2label: Mapping[int, str] | None = None,
    label2id: Mapping[str, int] | None = None,
    custom_config: ViTConfig | None = None,
    pretrained: bool = True,
) -> ViTForImageClassification:
    """
    Initialize ViTForImageClassification with 6-class classification head.
    Configures id2label and label2id with exact waste categories.
    """
    id2label = dict(id2label or IDX_TO_CLASS)
    label2id = dict(label2id or CLASS_TO_IDX)

    if custom_config is not None:
        custom_config.num_labels = num_labels
        custom_config.id2label = id2label
        custom_config.label2id = label2id
        return ViTForImageClassification(custom_config)

    if pretrained:
        try:
            model = ViTForImageClassification.from_pretrained(
                model_name_or_path,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
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
            id2label=id2label,
            label2id=label2id,
        )
        return ViTForImageClassification(cfg)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str] = WASTE_CLASSES,
) -> dict[str, Any]:
    """
    Compute classification metrics:
      - accuracy
      - precision (macro)
      - recall (macro)
      - f1_score (macro)
      - per-class breakdown (precision, recall, f1, support)
      - confusion matrix
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
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


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    classes: Sequence[str] = WASTE_CLASSES,
    max_steps: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Run evaluation loop over data_loader in eval mode.
    Returns (average_loss, metrics_dict).
    """
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
    metrics = compute_metrics(all_targets, all_preds, classes=classes)
    return avg_loss, metrics


def compute_class_weights(
    dataset_samples: Sequence[tuple[Any, int]],
    num_classes: int = 6,
) -> torch.Tensor:
    """
    Compute balanced inverse-frequency class weights for CrossEntropyLoss:
      w_c = total_samples / (num_classes * count_c)
    Normalized so the mean weight across all classes is exactly 1.0.
    """
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, lbl in dataset_samples:
        counts[int(lbl)] += 1.0
    total = float(len(dataset_samples))
    weights = total / (float(num_classes) * np.maximum(counts, 1.0))
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(
    model: ViTForImageClassification,
    checkpoint_dir: Path,
    training_metadata: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> None:
    """
    Save checkpoint in HuggingFace format:
      - model weights & config.json (with id2label, label2id, num_labels)
      - preprocessor_config.json
      - training_metadata.json (metadata, metrics & hyperparameters)
      - optimizer.pt & scheduler.pt (for resumption if provided)
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "config") and model.config is not None:
        model.config.num_labels = len(model.config.id2label) if model.config.id2label else 6
    model.save_pretrained(str(checkpoint_dir))

    # Also save preprocessor config matching ViT
    try:
        processor = ViTImageProcessor(
            do_resize=True,
            size={"height": 224, "width": 224},
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
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

    logger.info("Saved model checkpoint to '%s'", checkpoint_dir.resolve())


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


def train(
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
    model_instance: ViTForImageClassification | None = None,
) -> dict[str, Any]:
    """
    Complete ViT fine-tuning pipeline.

    Supports:
      - Class-weighted CrossEntropyLoss for imbalanced datasets (e.g. Trash class)
      - Selective layer freezing for fast CPU training
      - Checkpoint resumption from latest state (model, optimizer, scheduler)
      - Metric logging and artifact generation
    """
    # ── 1. Load Configuration ────────────────────────────────────────────────
    cfg_file = Path(config_path or DEFAULT_CONFIG_PATH)
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    eval_cfg = cfg.get("evaluation", {})

    seed = int(train_cfg.get("random_seed", 42))
    set_seed(seed)

    # ── 2. Device Selection ──────────────────────────────────────────────────
    device = detect_device(device_override)

    # ── 3. Paths & Directories ───────────────────────────────────────────────
    if data_dir_override:
        base_data_dir = resolve_path(data_dir_override)
        train_dir = base_data_dir / "train"
        val_dir = base_data_dir / "val"
        test_dir = base_data_dir / "test"
    else:
        base_data_dir = resolve_path(data_cfg.get("data_dir", "backend/data"))
        train_dir = resolve_path(data_cfg.get("train_dir", base_data_dir / "train"))
        val_dir = resolve_path(data_cfg.get("val_dir", base_data_dir / "val"))
        test_dir = resolve_path(data_cfg.get("test_dir", base_data_dir / "test"))

    raw_ckpt = resolve_path(
        checkpoint_dir_override or ckpt_cfg.get("save_dir", "backend/models/vit/best")
    )
    # Robustly resolve best and latest directories regardless of whether
    # backend/models/vit, backend/models/vit/best, or backend/models/vit/latest is passed
    if raw_ckpt.name == "latest":
        latest_checkpoint_dir = raw_ckpt
        checkpoint_dir = raw_ckpt.parent / "best"
    elif raw_ckpt.name == "best":
        checkpoint_dir = raw_ckpt
        latest_checkpoint_dir = raw_ckpt.parent / "latest"
    elif (raw_ckpt / "best").exists() or (raw_ckpt / "latest").exists() or raw_ckpt.name == "vit":
        checkpoint_dir = raw_ckpt / "best"
        latest_checkpoint_dir = raw_ckpt / "latest"
    else:
        checkpoint_dir = raw_ckpt
        latest_checkpoint_dir = raw_ckpt.parent / "latest"

    results_dir = resolve_path(
        results_dir_override or eval_cfg.get("results_dir", "backend/evaluation/results")
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. Verify Dataset Existence ──────────────────────────────────────────
    for split_name, split_path in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        if not split_path.exists():
            raise FileNotFoundError(
                f"Dataset split directory '{split_path.resolve()}' does not exist. "
                "Please run 'python backend/training/prepare_dataset.py' to prepare the dataset first."
            )

    # ── 5. DataLoaders ───────────────────────────────────────────────────────
    batch_size = (
        batch_size_override
        if batch_size_override is not None
        else int(train_cfg.get("batch_size", 32))
    )
    num_workers = int(data_cfg.get("num_workers", 0))
    image_size = int(model_cfg.get("image_size", 224))

    loaders = create_dataloaders(
        data_dir=base_data_dir,
        config=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    if len(train_loader) == 0:
        raise ValueError(
            f"Train split at '{train_dir.resolve()}' contains 0 images. "
            "Please run prepare_dataset.py with valid TrashNet images."
        )

    # ── 6. Model Initialization ──────────────────────────────────────────────
    model_name = model_cfg.get("checkpoint", "google/vit-base-patch16-224-in21k")
    num_classes = int(model_cfg.get("num_labels", 6))

    start_epoch = 1
    if model_instance is not None:
        model = model_instance.to(device)
    elif resume:
        # Explicit validation of resumption checkpoint prerequisites
        logger.info("Validating latest checkpoint for safe resumption at '%s'...", latest_checkpoint_dir.resolve())
        if not latest_checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Resumption failed: latest checkpoint directory '{latest_checkpoint_dir.resolve()}' does not exist. "
                "Cannot resume training without a prior checkpoint."
            )
        config_path = latest_checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Resumption failed: missing 'config.json' in '{latest_checkpoint_dir.resolve()}'."
            )
        weights_exist = (
            (latest_checkpoint_dir / "model.safetensors").exists()
            or (latest_checkpoint_dir / "pytorch_model.bin").exists()
        )
        if not weights_exist:
            raise FileNotFoundError(
                f"Resumption failed: missing model weights ('model.safetensors' or 'pytorch_model.bin') in '{latest_checkpoint_dir.resolve()}'."
            )
        meta_path = latest_checkpoint_dir / "training_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Resumption failed: missing 'training_metadata.json' in '{latest_checkpoint_dir.resolve()}'."
            )
        opt_path = latest_checkpoint_dir / "optimizer.pt"
        if not opt_path.exists():
            raise FileNotFoundError(
                f"Resumption failed: missing optimizer state '{opt_path.resolve()}'. "
                "Resumption requires optimizer.pt to preserve AdamW momentum and learning rate state."
            )
        sched_path = latest_checkpoint_dir / "scheduler.pt"
        if not sched_path.exists():
            raise FileNotFoundError(
                f"Resumption failed: missing scheduler state '{sched_path.resolve()}'. "
                "Resumption requires scheduler.pt to preserve warmup and learning rate decay schedule."
            )

        logger.info("Restoring model weights from latest checkpoint '%s'...", latest_checkpoint_dir)
        try:
            model = ViTForImageClassification.from_pretrained(str(latest_checkpoint_dir)).to(device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model weights from '{latest_checkpoint_dir}': {exc}") from exc

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            latest_epoch = int(meta.get("latest_epoch", meta.get("best_epoch", 0)))
            if latest_epoch <= 0:
                raise ValueError(f"Invalid latest_epoch ({latest_epoch}) in {meta_path}")
            start_epoch = latest_epoch + 1
            logger.info(
                "Verified latest checkpoint: completed epoch %d. Training will resume at start_epoch: %d.",
                latest_epoch, start_epoch
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to parse training metadata from '{meta_path}': {exc}") from exc
    else:
        logger.info("Initializing ViT base model from '%s'...", model_name)
        model = build_model(
            model_name_or_path=model_name,
            num_labels=num_classes,
            id2label=IDX_TO_CLASS,
            label2id=CLASS_TO_IDX,
            pretrained=True,
        ).to(device)

    # ── 7. Selective Layer Freezing (Optional for Fast CPU Training) ──────────
    if freeze_layers is not None and freeze_layers > 0:
        if hasattr(model, "vit") and hasattr(model.vit, "embeddings"):
            for p in model.vit.embeddings.parameters():
                p.requires_grad = False
        if hasattr(model, "vit") and hasattr(model.vit, "layers"):
            for i in range(min(freeze_layers, len(model.vit.layers))):
                for p in model.vit.layers[i].parameters():
                    p.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "Backbone frozen up to layer %d: %d / %d trainable parameters (%.1f%%).",
            freeze_layers,
            trainable_params,
            total_params,
            (trainable_params / total_params) * 100,
        )

    # ── 8. Training Hyperparameters & Optimizer ──────────────────────────────
    epochs = epochs_override if epochs_override is not None else int(train_cfg.get("epochs", 20))
    lr = learning_rate_override if learning_rate_override is not None else float(train_cfg.get("learning_rate", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_clip = float(train_cfg.get("gradient_clip", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    use_amp = bool(train_cfg.get("mixed_precision", True)) and (device.type == "cuda")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    total_training_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = int(train_cfg.get("warmup_ratio", 0.1) * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_training_steps),
    )

    # Load optimizer/scheduler state if resuming
    if resume:
        try:
            optimizer.load_state_dict(torch.load(latest_checkpoint_dir / "optimizer.pt", map_location=device))
            scheduler.load_state_dict(torch.load(latest_checkpoint_dir / "scheduler.pt", map_location=device))
            logger.info("Successfully restored optimizer & scheduler states from previous run.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to restore optimizer/scheduler state from '{latest_checkpoint_dir}': {exc}"
            ) from exc

    # Class-weighted loss
    if use_class_weights and hasattr(train_loader.dataset, "samples") and train_loader.dataset.samples:
        weights = compute_class_weights(train_loader.dataset.samples, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        logger.info(
            "Using class-weighted CrossEntropyLoss: %s",
            {cls: round(float(w), 3) for cls, w in zip(WASTE_CLASSES, weights)},
        )
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    logger.info(
        "Starting fine-tuning: start_epoch=%d, total_epochs=%d, batch_size=%d, lr=%.2e, patience=%d, mixed_precision=%s",
        start_epoch, epochs, batch_size, lr, patience, use_amp
    )

    target_metric = train_cfg.get("metric_for_best_model", "accuracy")
    best_val_score = float("-inf")
    best_epoch = 0
    patience_counter = 0
    training_history: list[dict[str, Any]] = []

    # Restore existing training history if available
    history_file = results_dir / "training_history.json"
    if history_file.exists():
        try:
            loaded_hist = json.loads(history_file.read_text(encoding="utf-8"))
            if isinstance(loaded_hist, list):
                if resume:
                    training_history = [entry for entry in loaded_hist if entry.get("epoch", 0) < start_epoch]
                else:
                    training_history = loaded_hist
                logger.info("Restored %d history record(s) from '%s'.", len(training_history), history_file)
        except Exception as exc:
            logger.warning("Could not load existing training history: %s", exc)

    if resume:
        # Restore best checkpoint score and best_epoch from best checkpoint metadata
        best_meta_file = checkpoint_dir / "training_metadata.json"
        latest_meta_file = latest_checkpoint_dir / "training_metadata.json"

        if best_meta_file.exists():
            try:
                best_meta = json.loads(best_meta_file.read_text(encoding="utf-8"))
                best_epoch = int(best_meta.get("best_epoch", 0))
                if target_metric == "f1_score":
                    best_val_score = float(best_meta.get("best_val_macro_f1", float("-inf")))
                else:
                    best_val_score = float(best_meta.get("best_val_accuracy", float("-inf")))
                logger.info(
                    "Restored best checkpoint score: %s=%.4f from epoch %d.",
                    target_metric, best_val_score, best_epoch
                )
            except Exception as exc:
                logger.warning("Could not read best checkpoint metadata: %s", exc)

        # Fallback to latest_meta if best_meta did not supply best_epoch
        if best_epoch == 0 and latest_meta_file.exists():
            try:
                latest_meta = json.loads(latest_meta_file.read_text(encoding="utf-8"))
                best_epoch = int(latest_meta.get("best_epoch", 0))
                if best_val_score == float("-inf"):
                    best_val_score = float(latest_meta.get("best_val_accuracy", float("-inf")))
            except Exception:
                pass

        # Restore patience counter based on epochs since best checkpoint
        latest_completed_epoch = start_epoch - 1
        if best_epoch > 0 and latest_completed_epoch >= best_epoch:
            patience_counter = max(0, latest_completed_epoch - best_epoch)
            logger.info(
                "Restored patience counter: %d epoch(s) without improvement since best epoch %d (limit: %d).",
                patience_counter, best_epoch, patience
            )
        else:
            patience_counter = 0

    max_steps_per_epoch: int | None = None
    if dry_run:
        epochs = 1
        max_steps_per_epoch = 2
        logger.info("Dry-run mode enabled: executing 1 epoch with 2 batches per split.")

    # ── 9. Training Loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds: list[int] = []
        train_targets: list[int] = []

        optimizer.zero_grad()

        for step, (images, labels) in enumerate(train_loader, start=1):
            if max_steps_per_epoch is not None and step > max_steps_per_epoch:
                break

            images = images.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(pixel_values=images)
                    loss = criterion(outputs.logits, labels)
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(pixel_values=images)
                loss = criterion(outputs.logits, labels)
                loss = loss / grad_accum_steps
                loss.backward()

            running_loss += loss.item() * grad_accum_steps * images.size(0)
            train_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            train_targets.extend(labels.cpu().tolist())

            if (
                step % grad_accum_steps == 0
                or step == len(train_loader)
                or (max_steps_per_epoch is not None and step == max_steps_per_epoch)
            ):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            if step % 10 == 0 or step == len(train_loader) or (max_steps_per_epoch is not None and step == max_steps_per_epoch):
                logger.info(
                    "  [Epoch %d/%d] Step %d/%d (%.1f%%) - batch_loss: %.4f",
                    epoch,
                    epochs,
                    step,
                    min(len(train_loader), max_steps_per_epoch or len(train_loader)),
                    (step / min(len(train_loader), max_steps_per_epoch or len(train_loader))) * 100,
                    loss.item() * grad_accum_steps,
                )

        epoch_train_loss = running_loss / len(train_targets) if train_targets else 0.0
        train_metrics = compute_metrics(train_targets, train_preds, classes=WASTE_CLASSES)

        # ── Validation ───────────────────────────────────────────────────────
        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            classes=WASTE_CLASSES,
            max_steps=max_steps_per_epoch,
        )

        history_entry = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 6),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": round(optimizer.param_groups[0]["lr"], 8),
        }
        training_history.append(history_entry)

        logger.info(
            "Epoch %02d/%02d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f f1=%.4f",
            epoch, epochs, epoch_train_loss, train_metrics["accuracy"],
            val_loss, val_metrics["accuracy"], val_metrics["macro_f1"],
        )

        # Always save latest checkpoint for safe resumption
        save_checkpoint(
            model,
            latest_checkpoint_dir,
            training_metadata={
                "model_name": model_name,
                "num_classes": num_classes,
                "classes": list(WASTE_CLASSES),
                "latest_epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_score,
                "latest_val_accuracy": val_metrics["accuracy"],
                "latest_val_macro_f1": val_metrics["macro_f1"],
                "hyperparameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "random_seed": seed,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # Persist updated training history after each epoch for resumption recovery
        try:
            (results_dir / "training_history.json").write_text(
                json.dumps(training_history, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Could not persist intermediate training history: %s", exc)

        # Checkpoint selection (metric: val_macro_f1 or val_accuracy)
        current_score = val_metrics["macro_f1"] if train_cfg.get("metric_for_best_model") == "f1_score" else val_metrics["accuracy"]
        if current_score > best_val_score:
            best_val_score = current_score
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                model,
                checkpoint_dir,
                training_metadata={
                    "model_name": model_name,
                    "num_classes": num_classes,
                    "classes": list(WASTE_CLASSES),
                    "best_epoch": best_epoch,
                    "best_val_accuracy": val_metrics["accuracy"],
                    "best_val_macro_f1": val_metrics["macro_f1"],
                    "hyperparameters": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "weight_decay": weight_decay,
                        "random_seed": seed,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                optimizer=optimizer,
                scheduler=scheduler,
            )
            logger.info("  ✓ Best checkpoint saved (val_score=%.4f at epoch %d).", best_val_score, epoch)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered: no improvement for %d epochs.", patience)
                break

    # ── 9. Final Evaluation on Test Set ──────────────────────────────────────
    logger.info("Running final evaluation on Test Set using best checkpoint...")
    # Load best checkpoint if saved
    if (checkpoint_dir / "config.json").exists():
        best_model = ViTForImageClassification.from_pretrained(str(checkpoint_dir)).to(device)
    else:
        best_model = model

    test_loss, test_metrics = evaluate(
        best_model,
        test_loader,
        device=device,
        criterion=criterion,
        classes=WASTE_CLASSES,
        max_steps=max_steps_per_epoch,
    )

    # ── 10. Generate Results Artifacts ───────────────────────────────────────
    final_results = {
        "model": model_name,
        "num_classes": num_classes,
        "classes": list(WASTE_CLASSES),
        "device": device.type.upper(),
        "best_epoch": best_epoch,
        "val_accuracy": best_val_score,
        "test_loss": test_loss,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["macro_precision"],
        "recall": test_metrics["macro_recall"],
        "f1_score": test_metrics["macro_f1"],
        "averaging": "macro",
        "weighted_metrics": {
            "precision": test_metrics["weighted_precision"],
            "recall": test_metrics["weighted_recall"],
            "f1_score": test_metrics["weighted_f1"],
        },
        "per_class": test_metrics["per_class"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    metrics_file = results_dir / "metrics.json"
    metrics_file.write_text(json.dumps(final_results, indent=2), encoding="utf-8")

    history_file = results_dir / "training_history.json"
    history_file.write_text(json.dumps(training_history, indent=2), encoding="utf-8")

    cm_file = results_dir / "confusion_matrix.json"
    cm_file.write_text(
        json.dumps(
            {
                "classes": list(WASTE_CLASSES),
                "matrix": test_metrics["confusion_matrix"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Evaluation complete. Metrics saved to '%s'", metrics_file)
    logger.info("Test Accuracy: %.4f | Macro F1: %.4f", test_metrics["accuracy"], test_metrics["macro_f1"])

    return {
        "training_history": training_history,
        "metrics": final_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ViT on waste classification dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--data-dir", type=Path, default=None, help="Root directory containing train/val/test")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory to save best checkpoint")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory to save metrics and logs")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Compute device override")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch dry-run for validation")
    parser.add_argument("--freeze-layers", type=int, default=None, help="Freeze lower N encoder layers (e.g. 10 or 12)")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class-weighted CrossEntropyLoss")
    parser.add_argument("--resume", action="store_true", help="Resume from latest saved checkpoint")

    args = parser.parse_args()

    train(
        config_path=args.config,
        data_dir_override=args.data_dir,
        checkpoint_dir_override=args.checkpoint_dir,
        results_dir_override=args.results_dir,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        learning_rate_override=args.lr,
        device_override=args.device,
        dry_run=args.dry_run,
        freeze_layers=args.freeze_layers,
        use_class_weights=not args.no_class_weights,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
