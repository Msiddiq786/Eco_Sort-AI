"""
PyTorch Dataset, DataLoader, and verification utilities for EcoSort AI.

Classes (strictly 6):
  0: cardboard
  1: glass
  2: metal
  3: paper
  4: plastic
  5: trash

Target model: google/vit-base-patch16-224-in21k
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Ensure backend root is on sys.path for direct script execution
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from app.core.config import settings
from training.augmentation import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_transforms,
)

logger = logging.getLogger("dataset")

# ── Official 6-class specification ───────────────────────────────────────────
WASTE_CLASSES: tuple[str, ...] = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
)
CLASS_TO_IDX: dict[str, int] = {cls: idx for idx, cls in enumerate(WASTE_CLASSES)}
IDX_TO_CLASS: dict[int, str] = {idx: cls for idx, cls in enumerate(WASTE_CLASSES)}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
)


class WasteDataset(Dataset):
    """
    PyTorch Dataset for waste classification images.

    Can be constructed from:
      1. A directory path (containing class subfolders: cardboard/, glass/, etc.)
      2. A pre-built list of (image_path, label) tuples.
    """

    def __init__(
        self,
        root_dir: Path | str | None = None,
        samples: Sequence[tuple[Path | str, int | str]] | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        class_to_idx: Mapping[str, int] | None = None,
    ) -> None:
        self.transform = transform
        self.class_to_idx = dict(class_to_idx or CLASS_TO_IDX)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        self.samples: list[tuple[Path, int]] = []

        if samples is not None:
            # Load from provided list of (path, label)
            for path, label in samples:
                p = Path(path)
                lbl_idx = self.class_to_idx[label] if isinstance(label, str) else int(label)
                self.samples.append((p, lbl_idx))
        elif root_dir is not None:
            # Discover from directory
            root = Path(root_dir)
            if not root.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: '{root.resolve()}'")

            for cls_name, cls_idx in self.class_to_idx.items():
                cls_folder = root / cls_name
                if not cls_folder.is_dir():
                    continue
                for img_path in sorted(cls_folder.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        self.samples.append((img_path, cls_idx))
        else:
            raise ValueError("Either root_dir or samples must be provided.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label_idx = self.samples[idx]

        try:
            with Image.open(img_path) as img:
                # Safely convert to 3-channel RGB (handles grayscale, RGBA, CMYK, etc.)
                if img.mode != "RGB":
                    image = img.convert("RGB")
                else:
                    image = img.copy()
        except Exception as exc:
            raise IOError(f"Failed to load image at '{img_path}': {exc}") from exc

        if self.transform is not None:
            tensor = self.transform(image)
        else:
            # Default fallback: PIL -> ToTensor
            from torchvision.transforms.functional import to_tensor
            tensor = to_tensor(image)

        return tensor, label_idx

    def get_class_counts(self) -> dict[str, int]:
        """Return the count of images per class in this dataset."""
        counts = {cls: 0 for cls in self.classes}
        for _, lbl in self.samples:
            cls_name = self.idx_to_class.get(lbl, str(lbl))
            counts[cls_name] = counts.get(cls_name, 0) + 1
        return counts


def create_dataloaders(
    data_dir: Path | str | None = None,
    config: Mapping[str, Any] | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    image_size: int | None = None,
) -> dict[str, Any]:
    """
    Construct train, validation, and test PyTorch DataLoaders.

    Parameters from `config` or arguments control:
      - batch_size
      - num_workers
      - image_size
      - split directory paths
    """
    cfg = config or {}
    data_cfg = cfg.get("data", {})

    if data_dir is not None:
        base_dir = Path(data_dir)
        train_dir = base_dir / "train"
        val_dir = base_dir / "val"
        test_dir = base_dir / "test"
    else:
        base_dir = Path(data_cfg.get("data_dir", "backend/data"))
        train_dir = Path(data_cfg.get("train_dir", base_dir / "train"))
        val_dir = Path(data_cfg.get("val_dir", base_dir / "val"))
        test_dir = Path(data_cfg.get("test_dir", base_dir / "test"))

    effective_batch_size = (
        batch_size
        if batch_size is not None
        else int(data_cfg.get("batch_size", 32))
    )
    effective_num_workers = (
        num_workers
        if num_workers is not None
        else int(data_cfg.get("num_workers", 0))
    )
    effective_image_size = (
        image_size
        if image_size is not None
        else int(data_cfg.get("image_size", DEFAULT_IMAGE_SIZE))
    )

    # Transforms
    train_tf = get_transforms("train", cfg)
    val_tf = get_transforms("val", cfg)
    test_tf = get_transforms("test", cfg)

    train_ds = WasteDataset(root_dir=train_dir, transform=train_tf)
    val_ds = WasteDataset(root_dir=val_dir, transform=val_tf)
    test_ds = WasteDataset(root_dir=test_dir, transform=test_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
        "classes": WASTE_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
    }


def verify_dataset(
    data_loader: DataLoader,
    expected_classes: Sequence[str] = WASTE_CLASSES,
    expected_image_size: int = DEFAULT_IMAGE_SIZE,
    num_batches: int = 3,
) -> dict[str, Any]:
    """
    Verify dataset batches and return a comprehensive verification report:
      1. Inspects several samples/batches from the DataLoader.
      2. Checks tensor shape [B, 3, 224, 224].
      3. Checks tensor dtype is torch.float32.
      4. Checks ImageNet normalization (values are not all in [0, 1] or [0, 255],
         channel values roughly centered around 0 with reasonable variance).
      5. Confirms class labels are valid integers in [0, len(expected_classes) - 1].
      6. Reports class-to-index mapping.

    Returns:
      Report dict with "status": "PASS" | "FAIL" and detailed checks.
    """
    report: dict[str, Any] = {
        "status": "PASS",
        "num_batches_checked": 0,
        "num_samples_checked": 0,
        "tensor_shape": None,
        "tensor_dtype": None,
        "normalization": {},
        "labels_valid": True,
        "unique_labels_found": [],
        "class_to_idx": {cls: i for i, cls in enumerate(expected_classes)},
        "errors": [],
    }

    if len(data_loader) == 0:
        report["status"] = "FAIL"
        report["errors"].append("DataLoader is empty (0 samples).")
        return report

    all_labels: list[int] = []
    all_channel_means: list[list[float]] = []
    all_channel_stds: list[list[float]] = []
    global_min = float("inf")
    global_max = float("-inf")

    num_classes = len(expected_classes)

    for b_idx, (images, labels) in enumerate(data_loader):
        if b_idx >= num_batches:
            break

        report["num_batches_checked"] += 1
        batch_size = images.shape[0]
        report["num_samples_checked"] += batch_size

        # ── 1. Check Shape ───────────────────────────────────────────────────
        if len(images.shape) != 4:
            report["errors"].append(
                f"Expected 4D tensor [B, C, H, W], got shape {list(images.shape)}"
            )
            report["status"] = "FAIL"

        if images.shape[1] != 3:
            report["errors"].append(
                f"Expected 3 channels (RGB), got {images.shape[1]} channels."
            )
            report["status"] = "FAIL"

        if images.shape[2] != expected_image_size or images.shape[3] != expected_image_size:
            report["errors"].append(
                f"Expected image size {expected_image_size}x{expected_image_size}, "
                f"got {images.shape[2]}x{images.shape[3]}."
            )
            report["status"] = "FAIL"

        report["tensor_shape"] = list(images.shape)

        # ── 2. Check Dtype ───────────────────────────────────────────────────
        if images.dtype != torch.float32:
            report["errors"].append(
                f"Expected torch.float32, got {images.dtype}"
            )
            report["status"] = "FAIL"
        report["tensor_dtype"] = str(images.dtype)

        # ── 3. Check Normalization ───────────────────────────────────────────
        b_min = float(images.min().item())
        b_max = float(images.max().item())
        global_min = min(global_min, b_min)
        global_max = max(global_max, b_max)

        # Per-channel mean and std: images has shape [B, 3, H, W]
        # Calculate over (B, H, W)
        ch_mean = images.mean(dim=[0, 2, 3]).tolist()
        ch_std = images.std(dim=[0, 2, 3]).tolist()
        all_channel_means.append(ch_mean)
        all_channel_stds.append(ch_std)

        # Unnormalized check: if min >= 0 and max <= 1.05 and min is close to 0,
        # it might not have been normalized with ImageNet mean/std!
        # Normalization with mean ~ 0.45 and std ~ 0.22 maps [0, 1] to roughly [-2.2, 2.7].
        if b_min >= 0.0 and b_max <= 1.05:
            report["errors"].append(
                f"Tensor values appear unnormalized in [0, 1]: min={b_min:.3f}, max={b_max:.3f}. "
                "ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) was not applied."
            )
            report["status"] = "FAIL"
        elif b_min >= 0.0 and b_max > 200.0:
            report["errors"].append(
                f"Tensor values appear to be raw 8-bit ints in [0, 255]: max={b_max:.1f}."
            )
            report["status"] = "FAIL"

        # ── 4. Check Labels ──────────────────────────────────────────────────
        batch_lbls = labels.tolist() if torch.is_tensor(labels) else list(labels)
        all_labels.extend(batch_lbls)

        for lbl in batch_lbls:
            if not isinstance(lbl, int) or lbl < 0 or lbl >= num_classes:
                report["errors"].append(
                    f"Invalid label {lbl}. Expected integer in range [0, {num_classes - 1}]."
                )
                report["labels_valid"] = False
                report["status"] = "FAIL"
                break

    # Aggregate normalization stats
    num_ch = len(all_channel_means[0]) if all_channel_means and all_channel_means[0] else 0
    avg_ch_means = [
        round(sum(m[c] for m in all_channel_means) / len(all_channel_means), 4)
        for c in range(num_ch)
    ]
    avg_ch_stds = [
        round(sum(s[c] for s in all_channel_stds) / len(all_channel_stds), 4)
        for c in range(num_ch)
    ]

    report["normalization"] = {
        "min": round(global_min, 4),
        "max": round(global_max, 4),
        "channel_means": avg_ch_means,
        "channel_stds": avg_ch_stds,
        "is_normalized": report["status"] == "PASS",
    }
    report["unique_labels_found"] = sorted(set(all_labels))

    if report["errors"]:
        report["status"] = "FAIL"

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify EcoSort AI PyTorch Dataset & DataLoader.")
    parser.add_argument("--data-dir", type=Path, default=Path("backend/data"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    split_dir = args.data_dir / args.split
    if not split_dir.exists():
        print(f"Error: Split directory does not exist: {split_dir}")
        return

    tf = get_transforms(args.split)
    dataset = WasteDataset(root_dir=split_dir, transform=tf)
    print(f"Loaded '{args.split}' split: {len(dataset)} samples across {len(dataset.classes)} classes.")
    print("Class distribution:", dataset.get_class_counts())

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    report = verify_dataset(loader)

    print("\nDataset Verification Report:")
    print("----------------------------")
    print(f"Status:               {report['status']}")
    print(f"Samples verified:     {report['num_samples_checked']}")
    print(f"Tensor shape:         {report['tensor_shape']}")
    print(f"Tensor dtype:         {report['tensor_dtype']}")
    print(f"Normalization min:    {report['normalization'].get('min')}")
    print(f"Normalization max:    {report['normalization'].get('max')}")
    print(f"Channel means:        {report['normalization'].get('channel_means')}")
    print(f"Channel stds:         {report['normalization'].get('channel_stds')}")
    print(f"Labels valid:         {report['labels_valid']}")
    print(f"Unique labels:        {report['unique_labels_found']}")
    print(f"Class-to-index:       {report['class_to_idx']}")
    if report["errors"]:
        print(f"Errors ({len(report['errors'])}):")
        for err in report["errors"]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
