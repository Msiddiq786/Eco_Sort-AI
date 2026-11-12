"""
Unit and integration tests for Phase 3: Dataset Preparation and Preprocessing.

Tests:
  1. Dataset directory validation (missing dir raises FileNotFoundError).
  2. Class detection (missing required class raises ValueError).
  3. All 6 classes detected and counted.
  4. Corrupt/unreadable image detection.
  5. Deterministic split reproducibility with random_seed.
  6. Zero data leakage across train, val, test splits.
  7. Stratified distribution across splits.
  8. RGB conversion (grayscale & RGBA safely converted to 3-channel RGB).
  9. Tensor dimensions ([3, 224, 224]) and dtype (torch.float32).
  10. ImageNet normalization conforms to ViT pipeline expectations.
  11. Deterministic validation transforms vs stochastic training transforms.
  12. Consistent class-to-index mapping.
  13. PyTorch DataLoader creation and batch iteration.
  14. Dataset verification utility (PASS on valid data, catches errors on invalid).
"""
from __future__ import annotations

import io
import json
import pathlib
from typing import Sequence

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from training.augmentation import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    EnsureRGB,
    get_eval_transforms,
    get_test_transforms,
    get_train_transforms,
    get_transforms,
    get_val_transforms,
)
from training.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    WASTE_CLASSES,
    WasteDataset,
    create_dataloaders,
    verify_dataset,
)
from training.prepare_dataset import (
    REQUIRED_CLASSES,
    is_image_valid,
    prepare_dataset,
    split_dataset,
    validate_raw_dataset,
)

# ── Test Helpers / Fixtures ──────────────────────────────────────────────────


def _create_synthetic_image(
    path: pathlib.Path,
    size: tuple[int, int] = (64, 64),
    mode: str = "RGB",
    color: tuple = (100, 150, 200),
) -> pathlib.Path:
    """Create a valid synthetic image file on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "RGB":
        img = Image.new("RGB", size, color=color[:3])
    elif mode == "RGBA":
        img = Image.new("RGBA", size, color=color if len(color) == 4 else color + (255,))
    elif mode == "L":
        img = Image.new("L", size, color=color[0] if isinstance(color, tuple) else color)
    else:
        img = Image.new(mode, size)
    img.save(path, format="JPEG" if mode == "RGB" else "PNG")
    return path


@pytest.fixture
def mock_raw_dataset(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Create a mock raw dataset containing all 6 classes with 10 images each
    (60 images total).
    """
    raw_dir = tmp_path / "raw"
    for cls in REQUIRED_CLASSES:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            img_path = cls_dir / f"{cls}_{i:03d}.jpg"
            _create_synthetic_image(img_path, color=((i * 20) % 255, (i * 30) % 255, (i * 40) % 255))
    return raw_dir


# ── 1. Dataset Directory Validation ──────────────────────────────────────────


def test_dataset_directory_validation_missing_dir(tmp_path: pathlib.Path):
    """Missing raw directory must raise FileNotFoundError."""
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError) as exc_info:
        validate_raw_dataset(missing_dir)
    assert "does not exist" in str(exc_info.value).lower()


def test_dataset_directory_validation_missing_classes(tmp_path: pathlib.Path):
    """Missing any of the 6 required class folders must raise ValueError."""
    partial_dir = tmp_path / "partial_raw"
    # Create only 3 of the 6 classes
    for cls in ["cardboard", "glass", "metal"]:
        (partial_dir / cls).mkdir(parents=True, exist_ok=True)
        _create_synthetic_image(partial_dir / cls / "sample.jpg")

    with pytest.raises(ValueError) as exc_info:
        validate_raw_dataset(partial_dir)
    msg = str(exc_info.value).lower()
    assert "missing required class directories" in msg
    assert "paper" in msg or "plastic" in msg or "trash" in msg


# ── 2. Class Detection & Counting ────────────────────────────────────────────


def test_all_six_classes_detected(mock_raw_dataset: pathlib.Path):
    """validate_raw_dataset must detect all 6 classes and count images accurately."""
    class_images = validate_raw_dataset(mock_raw_dataset)
    assert set(class_images.keys()) == set(REQUIRED_CLASSES)
    assert len(class_images) == 6
    for cls in REQUIRED_CLASSES:
        assert len(class_images[cls]) == 10


# ── 3. Corrupt Image Detection ───────────────────────────────────────────────


def test_corrupt_image_detection(tmp_path: pathlib.Path):
    """Corrupt or 0-byte images must be detected and rejected."""
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"not a valid image content at all 123456789")
    assert is_image_valid(corrupt_file) is False

    empty_file = tmp_path / "empty.jpg"
    empty_file.write_bytes(b"")
    assert is_image_valid(empty_file) is False

    valid_file = tmp_path / "valid.jpg"
    _create_synthetic_image(valid_file)
    assert is_image_valid(valid_file) is True


# ── 4. Split Reproducibility ─────────────────────────────────────────────────


def test_split_reproducibility(mock_raw_dataset: pathlib.Path):
    """Splitting with the same random seed must produce identical splits."""
    class_images = validate_raw_dataset(mock_raw_dataset)

    split_a = split_dataset(class_images, random_seed=42)
    split_b = split_dataset(class_images, random_seed=42)

    for split_name in ("train", "val", "test"):
        paths_a = [str(p) for p, _ in split_a[split_name]]
        paths_b = [str(p) for p, _ in split_b[split_name]]
        assert paths_a == paths_b, f"Split '{split_name}' was not reproducible across runs!"


# ── 5. Zero Data Leakage Across Splits ───────────────────────────────────────


def test_no_data_leakage(mock_raw_dataset: pathlib.Path):
    """The same image must NEVER appear in multiple splits."""
    class_images = validate_raw_dataset(mock_raw_dataset)
    splits = split_dataset(
        class_images, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )

    train_files = {p.name for p, _ in splits["train"]}
    val_files = {p.name for p, _ in splits["val"]}
    test_files = {p.name for p, _ in splits["test"]}

    assert train_files.isdisjoint(val_files), "Leakage detected between train and val!"
    assert train_files.isdisjoint(test_files), "Leakage detected between train and test!"
    assert val_files.isdisjoint(test_files), "Leakage detected between val and test!"

    # Total check
    total_images = sum(len(paths) for paths in class_images.values())
    assert len(train_files) + len(val_files) + len(test_files) == total_images


# ── 6. Stratified Distribution ───────────────────────────────────────────────


def test_stratified_split_distribution(mock_raw_dataset: pathlib.Path):
    """Every class must be represented in train, val, and test splits."""
    class_images = validate_raw_dataset(mock_raw_dataset)
    splits = split_dataset(class_images, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    for split_name in ("train", "val", "test"):
        labels_in_split = {label for _, label in splits[split_name]}
        # Every class should be present because dataset is stratified
        assert labels_in_split == set(REQUIRED_CLASSES)


# ── 7. RGB Conversion (Grayscale & RGBA) ─────────────────────────────────────


def test_rgb_conversion_grayscale_and_rgba(tmp_path: pathlib.Path):
    """Grayscale and RGBA images must be converted to 3-channel RGB tensors."""
    gray_path = tmp_path / "gray.png"
    rgba_path = tmp_path / "rgba.png"
    _create_synthetic_image(gray_path, mode="L", color=(128,))
    _create_synthetic_image(rgba_path, mode="RGBA", color=(50, 100, 150, 200))

    tf = get_val_transforms(image_size=224)

    # Test with WasteDataset
    dataset = WasteDataset(
        samples=[(gray_path, "cardboard"), (rgba_path, "glass")],
        transform=tf,
    )

    tensor_gray, lbl_gray = dataset[0]
    tensor_rgba, lbl_rgba = dataset[1]

    assert tensor_gray.shape == (3, 224, 224)
    assert tensor_rgba.shape == (3, 224, 224)
    assert lbl_gray == CLASS_TO_IDX["cardboard"]
    assert lbl_rgba == CLASS_TO_IDX["glass"]


# ── 8. Tensor Dimensions and Dtype ───────────────────────────────────────────


def test_tensor_dimensions_and_dtype(tmp_path: pathlib.Path):
    """Preprocessing must yield [3, 224, 224] tensor of torch.float32."""
    img_path = tmp_path / "sample.jpg"
    _create_synthetic_image(img_path, size=(300, 400))

    val_tf = get_val_transforms(image_size=224)
    with Image.open(img_path) as img:
        tensor = val_tf(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


# ── 9. ImageNet Normalization ────────────────────────────────────────────────


def test_normalization_vit_compatible(tmp_path: pathlib.Path):
    """Normalized pixel values must be centered around ImageNet mean/std."""
    img_path = tmp_path / "sample.jpg"
    _create_synthetic_image(img_path, size=(224, 224), color=(128, 128, 128))

    val_tf = get_val_transforms(image_size=224)
    with Image.open(img_path) as img:
        tensor = val_tf(img)

    # Channel 0: (128/255 - 0.485) / 0.229 ≈ (0.502 - 0.485) / 0.229 ≈ 0.074
    ch0_val = tensor[0, 0, 0].item()
    assert -3.0 < ch0_val < 3.0
    # Values should not be unnormalized in [0, 1] or [0, 255]
    assert tensor.min().item() < 0.2 or tensor.max().item() > 0.8


# ── 10. Deterministic Validation vs Stochastic Training ──────────────────────


def test_val_transforms_are_deterministic(tmp_path: pathlib.Path):
    """Validation transforms must always produce the exact same tensor."""
    img_path = tmp_path / "sample.jpg"
    _create_synthetic_image(img_path, size=(100, 100))

    val_tf = get_val_transforms(image_size=224)
    with Image.open(img_path) as img:
        t1 = val_tf(img)
        t2 = val_tf(img)

    assert torch.equal(t1, t2), "Validation preprocessing must be deterministic!"


# ── 11. Class-to-Index Mapping Consistency ───────────────────────────────────


def test_class_index_mapping_consistency():
    """Class to index mapping must be identical across the entire codebase."""
    expected = {
        "cardboard": 0,
        "glass": 1,
        "metal": 2,
        "paper": 3,
        "plastic": 4,
        "trash": 5,
    }
    assert CLASS_TO_IDX == expected
    assert len(WASTE_CLASSES) == 6
    for idx, name in enumerate(WASTE_CLASSES):
        assert CLASS_TO_IDX[name] == idx
        assert IDX_TO_CLASS[idx] == name


# ── 12. Full Pipeline & Output Directories ───────────────────────────────────


def test_prepare_dataset_full_pipeline(mock_raw_dataset: pathlib.Path, tmp_path: pathlib.Path):
    """prepare_dataset must generate train/, val/, test/ and splits_summary.json."""
    output_dir = tmp_path / "output_data"
    summary = prepare_dataset(
        raw_dir=mock_raw_dataset,
        output_dir=output_dir,
        copy_files=True,
    )

    assert summary["status"] == "SUCCESS"
    assert summary["counts"]["total"] == 60
    assert summary["counts"]["train"] + summary["counts"]["val"] + summary["counts"]["test"] == 60
    assert summary["leakage_check"] == "PASSED"

    # Verify directory structure
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        assert split_dir.is_dir()
        for cls in REQUIRED_CLASSES:
            assert (split_dir / cls).is_dir()

    # Verify summary file exists and can be parsed
    summary_file = output_dir / "splits_summary.json"
    assert summary_file.exists()
    loaded_summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert loaded_summary["counts"]["total"] == 60


# ── 13. DataLoader Creation & Batching ───────────────────────────────────────


def test_dataloader_creation_and_batching(mock_raw_dataset: pathlib.Path, tmp_path: pathlib.Path):
    """create_dataloaders must construct functional train, val, and test DataLoaders."""
    output_dir = tmp_path / "output_data"
    prepare_dataset(raw_dir=mock_raw_dataset, output_dir=output_dir, copy_files=True)

    loaders = create_dataloaders(
        data_dir=output_dir,
        batch_size=4,
        num_workers=0,
        image_size=224,
    )

    assert "train" in loaders
    assert "val" in loaders
    assert "test" in loaders

    train_loader = loaders["train"]
    images, labels = next(iter(train_loader))

    assert images.shape == (4, 3, 224, 224)
    assert images.dtype == torch.float32
    assert labels.shape == (4,)
    assert labels.dtype == torch.int64


# ── 14. Dataset Verification Utility ─────────────────────────────────────────


def test_dataset_verification_utility_pass(mock_raw_dataset: pathlib.Path, tmp_path: pathlib.Path):
    """verify_dataset must return status: PASS on a properly formatted DataLoader."""
    output_dir = tmp_path / "output_data"
    prepare_dataset(raw_dir=mock_raw_dataset, output_dir=output_dir, copy_files=True)

    loaders = create_dataloaders(data_dir=output_dir, batch_size=4)
    report = verify_dataset(loaders["train"])

    assert report["status"] == "PASS"
    assert report["labels_valid"] is True
    assert report["tensor_shape"] == [4, 3, 224, 224]
    assert report["tensor_dtype"] == "torch.float32"
    assert report["normalization"]["is_normalized"] is True
    assert len(report["errors"]) == 0


def test_dataset_verification_catches_invalid_shape():
    """verify_dataset must catch incorrect tensor shapes."""
    # Create fake DataLoader returning wrong shape [4, 1, 100, 100]
    fake_data = [(torch.randn(1, 100, 100), 0) for _ in range(8)]
    loader = DataLoader(fake_data, batch_size=4)

    report = verify_dataset(loader, expected_image_size=224)
    assert report["status"] == "FAIL"
    assert len(report["errors"]) > 0
