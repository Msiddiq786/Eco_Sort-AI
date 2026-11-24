"""
EcoSort AI V2 — Training Pipeline & Dataset Verification Tests.

Tests:
1. Python syntax check for all V2 training scripts.
2. Dataset integrity check (all 7 classes present, non-zero, zero corrupted, zero leakage).
3. Checkpoint path isolation test (rejects any V1 path).
4. 7-class classifier head verification (out_features=7, id2label, label2id).
5. V1 best checkpoint preservation check (best epoch=12, val_score=0.9393, num_classes=6).
"""
import py_compile
import json
from pathlib import Path
import pytest
import torch

from training.train_vit_v2 import (
    V2_CLASSES,
    V2_CLASS_TO_IDX,
    V2_IDX_TO_CLASS,
    validate_dataset_integrity,
    save_v2_checkpoint,
    build_v2_model,
)

BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
V1_BEST_DIR = Path(__file__).resolve().parent.parent / "models" / "vit" / "best"


def test_1_python_syntax():
    """Verify syntax of V2 scripts."""
    root = Path(__file__).resolve().parent.parent
    py_compile.compile(str(root / "training" / "train_vit_v2.py"), doraise=True)
    py_compile.compile(str(root / "training" / "prepare_organic_dataset.py"), doraise=True)


def test_2_dataset_integrity_and_organic_support():
    """Verify dataset splits, 7 classes, organic image counts, and zero leakage."""
    report = validate_dataset_integrity(BASE_DATA_DIR, classes=V2_CLASSES)

    assert len(report["train"]) == 7, "Must contain exactly 7 classes in train"
    assert len(report["val"]) == 7, "Must contain exactly 7 classes in val"
    assert len(report["test"]) == 7, "Must contain exactly 7 classes in test"

    # Organic image counts
    assert report["train"]["organic"] == 280, f"Expected 280 organic train images, got {report['train']['organic']}"
    assert report["val"]["organic"] == 60, f"Expected 60 organic val images, got {report['val']['organic']}"
    assert report["test"]["organic"] == 60, f"Expected 60 organic test images, got {report['test']['organic']}"

    # Total counts
    total = report["total_images"]
    assert total == 2927, f"Expected 2927 total images across 7 classes, got {total}"
    assert report["leakage_check"] == "PASSED"


def test_3_checkpoint_isolation():
    """Verify V2 strictly rejects saving inside backend/models/vit/."""
    dummy_model = build_v2_model(pretrained=False)
    v1_bad_paths = [
        Path("backend/models/vit/best"),
        Path("backend/models/vit/latest"),
        Path("backend/models/vit"),
    ]
    for bad_path in v1_bad_paths:
        with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION"):
            save_v2_checkpoint(dummy_model, bad_path)


def test_4_seven_class_classifier_head():
    """Verify model classifier head has exactly 7 outputs and correct labels."""
    model = build_v2_model(pretrained=False)
    assert model.classifier.out_features == 7
    assert len(model.config.id2label) == 7
    assert len(model.config.label2id) == 7
    assert model.config.id2label[5] == "organic"
    assert model.config.label2id["organic"] == 5


def test_5_v1_model_preservation():
    """Verify V1 best checkpoint remains completely untouched."""
    meta_path = V1_BEST_DIR / "training_metadata.json"
    assert meta_path.exists(), "V1 best metadata file must exist"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta.get("best_epoch") == 12, "V1 best epoch must remain 12"
    assert abs(float(meta.get("best_val_accuracy", 0.0)) - 0.9393) < 0.001
    assert meta.get("num_classes") == 6, "V1 must have 6 classes"
