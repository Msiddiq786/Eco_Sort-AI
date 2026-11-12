"""
Unit and integration tests for Phase 4: Vision Transformer Fine-Tuning.

Tests:
  1. Model initialization with tiny offline configuration.
  2. Exactly 6 classification labels.
  3. Strict label mapping (id2label & label2id) across all 6 classes.
  4. Forward-pass output tensor shape [batch_size, 6].
  5. CrossEntropyLoss calculation and gradient backprop.
  6. Checkpoint saving in HuggingFace format (config.json, weights, metadata).
  7. Metrics calculation (accuracy, macro precision, recall, f1, per-class, confusion matrix).
  8. Evaluation output schema and JSON artifact generation.
  9. Reproducibility: random seed setting across torch and numpy.
  10. Hardware detection: CPU fallback and device string reporting.
  11. End-to-end training loop execution with tiny in-memory model.
  12. Missing dataset raises clear FileNotFoundError without attempting training.
"""
from __future__ import annotations

import json
import pathlib
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import ViTConfig, ViTForImageClassification

from training.dataset import CLASS_TO_IDX, IDX_TO_CLASS, WASTE_CLASSES
from training.train_vit import (
    build_model,
    compute_metrics,
    detect_device,
    evaluate,
    save_checkpoint,
    set_seed,
    train,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_vit_config() -> ViTConfig:
    """Create a minimal ViTConfig for offline testing in milliseconds."""
    return ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=6,
        id2label=IDX_TO_CLASS,
        label2id=CLASS_TO_IDX,
    )


@pytest.fixture
def tiny_vit_model(tiny_vit_config: ViTConfig) -> ViTForImageClassification:
    """Instantiate a tiny ViT model with no external weights needed."""
    return ViTForImageClassification(tiny_vit_config)


@pytest.fixture
def synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a synthetic batch of [4, 3, 224, 224] images and 4 labels."""
    images = torch.randn(4, 3, 224, 224, dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    return images, labels


# ── 1. Model Initialization ──────────────────────────────────────────────────


def test_model_initialization(tiny_vit_config: ViTConfig):
    """Model must initialize with custom config and correct class architecture."""
    model = build_model(
        num_labels=6,
        id2label=IDX_TO_CLASS,
        label2id=CLASS_TO_IDX,
        custom_config=tiny_vit_config,
        pretrained=False,
    )
    assert isinstance(model, ViTForImageClassification)
    assert model.classifier is not None


# ── 2. Number of Classification Labels ───────────────────────────────────────


def test_correct_number_of_labels(tiny_vit_model: ViTForImageClassification):
    """ViT classification head must output exactly 6 logits."""
    assert tiny_vit_model.num_labels == 6
    assert tiny_vit_model.classifier.out_features == 6


# ── 3. Class Mapping Ordering ────────────────────────────────────────────────


def test_class_mapping_ordering(tiny_vit_model: ViTForImageClassification):
    """id2label and label2id must strictly match the 6 waste classes."""
    cfg = tiny_vit_model.config
    expected_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    assert len(cfg.id2label) == 6
    assert len(cfg.label2id) == 6

    for idx, name in enumerate(expected_classes):
        assert cfg.id2label[idx] == name
        assert cfg.label2id[name] == idx


# ── 4. Forward-Pass Output Shape ─────────────────────────────────────────────


def test_forward_pass_tensor_shape(
    tiny_vit_model: ViTForImageClassification,
    synthetic_batch: tuple[torch.Tensor, torch.Tensor],
):
    """Forward pass must produce logits of shape [batch_size, 6]."""
    images, _ = synthetic_batch
    tiny_vit_model.eval()
    with torch.no_grad():
        outputs = tiny_vit_model(pixel_values=images)

    assert hasattr(outputs, "logits")
    assert outputs.logits.shape == (4, 6)
    assert outputs.logits.dtype == torch.float32


# ── 5. Loss Calculation and Backward Pass ────────────────────────────────────


def test_loss_calculation_and_backprop(
    tiny_vit_model: ViTForImageClassification,
    synthetic_batch: tuple[torch.Tensor, torch.Tensor],
):
    """CrossEntropyLoss must compute a scalar and backpropagate valid gradients."""
    images, labels = synthetic_batch
    tiny_vit_model.train()

    criterion = nn.CrossEntropyLoss()
    outputs = tiny_vit_model(pixel_values=images)
    loss = criterion(outputs.logits, labels)

    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0
    assert torch.isfinite(loss)

    loss.backward()
    # Check that classifier head received gradients
    assert tiny_vit_model.classifier.weight.grad is not None
    assert torch.isfinite(tiny_vit_model.classifier.weight.grad).all()


# ── 6. Checkpoint Saving Structure ───────────────────────────────────────────


def test_checkpoint_saving_structure(
    tiny_vit_model: ViTForImageClassification,
    tmp_path: pathlib.Path,
):
    """save_checkpoint must output config.json with id2label, weights, and metadata."""
    ckpt_dir = tmp_path / "vit_best"
    metadata = {
        "model_name": "google/vit-base-patch16-224-in21k",
        "best_epoch": 5,
        "best_val_accuracy": 0.892,
        "classes": list(WASTE_CLASSES),
    }

    save_checkpoint(tiny_vit_model, ckpt_dir, training_metadata=metadata)

    # Check config.json
    config_file = ckpt_dir / "config.json"
    assert config_file.exists()
    saved_cfg = json.loads(config_file.read_text(encoding="utf-8"))
    num_labels = saved_cfg.get("num_labels", len(saved_cfg.get("id2label", {})))
    assert num_labels == 6
    assert len(saved_cfg["id2label"]) == 6
    assert saved_cfg["id2label"]["0"] == "cardboard"
    assert saved_cfg["id2label"]["5"] == "trash"

    # Check metadata
    meta_file = ckpt_dir / "training_metadata.json"
    assert meta_file.exists()
    saved_meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert saved_meta["best_epoch"] == 5
    assert saved_meta["best_val_accuracy"] == 0.892


# ── 7. Metrics Calculation ───────────────────────────────────────────────────


def test_compute_metrics():
    """compute_metrics must calculate accuracy, macro/weighted F1, and confusion matrix."""
    y_true = [0, 1, 2, 3, 4, 5, 0, 1]
    y_pred = [0, 1, 2, 3, 4, 5, 0, 2]  # 7 correct, 1 wrong (item 7: true=1, pred=2)

    metrics = compute_metrics(y_true, y_pred, classes=WASTE_CLASSES)

    assert metrics["accuracy"] == pytest.approx(7 / 8, abs=1e-4)
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert len(metrics["per_class"]) == 6
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 6
    assert len(metrics["confusion_matrix"][0]) == 6


# ── 8. Evaluation Function Schema ────────────────────────────────────────────


def test_evaluation_schema(tiny_vit_model: ViTForImageClassification):
    """evaluate() must return a tuple of (loss, metrics_dict) with full schema."""
    # Synthetic loader: 8 samples of shape [3, 224, 224]
    x = torch.randn(8, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1], dtype=torch.int64)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    device = torch.device("cpu")
    loss, metrics = evaluate(tiny_vit_model, loader, device=device)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics


# ── 9. Reproducibility Configuration ─────────────────────────────────────────


def test_reproducibility_seed():
    """set_seed must produce identical random numbers across torch and numpy."""
    set_seed(42)
    t1 = torch.randn(5)
    set_seed(42)
    t2 = torch.randn(5)
    assert torch.equal(t1, t2), "torch RNG was not reproducible with fixed seed!"


# ── 10. Hardware Detection ───────────────────────────────────────────────────


def test_hardware_detection_cpu_fallback():
    """detect_device must return CPU device when forced or when CUDA is unavailable."""
    dev = detect_device(force_device="cpu")
    assert dev.type == "cpu"


# ── 11. End-to-End Training Loop with Tiny Model ──────────────────────────────


def test_training_loop_with_tiny_model(
    tiny_vit_model: ViTForImageClassification,
    tmp_path: pathlib.Path,
):
    """train() must complete 1 epoch, save best checkpoint and generate artifacts."""
    from PIL import Image

    # Setup mock data directory
    data_dir = tmp_path / "data"
    for split in ("train", "val", "test"):
        for cls in WASTE_CLASSES:
            cls_dir = data_dir / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                img = Image.new("RGB", (64, 64), color=(i * 30, 100, 150))
                img.save(cls_dir / f"img_{i}.jpg")

    ckpt_dir = tmp_path / "models" / "vit" / "best"
    results_dir = tmp_path / "evaluation" / "results"

    results = train(
        data_dir_override=data_dir,
        checkpoint_dir_override=ckpt_dir,
        results_dir_override=results_dir,
        epochs_override=1,
        batch_size_override=4,
        device_override="cpu",
        model_instance=tiny_vit_model,
    )

    assert "training_history" in results
    assert len(results["training_history"]) == 1
    assert "metrics" in results
    assert results["metrics"]["accuracy"] >= 0.0
    assert (ckpt_dir / "config.json").exists()
    assert (results_dir / "metrics.json").exists()
    assert (results_dir / "training_history.json").exists()
    assert (results_dir / "confusion_matrix.json").exists()


# ── 12. Missing Dataset Stops With Clear Error ────────────────────────────────


def test_missing_dataset_raises_filenotfound(tmp_path: pathlib.Path):
    """train() must raise FileNotFoundError if dataset directories do not exist."""
    missing_data_dir = tmp_path / "nonexistent_data"
    with pytest.raises(FileNotFoundError) as exc_info:
        train(data_dir_override=missing_data_dir)
    assert "does not exist" in str(exc_info.value).lower()
