"""
Image preprocessing and augmentation transforms for EcoSort AI.

Target model: google/vit-base-patch16-224-in21k
Input requirements:
  - 224 x 224 RGB images
  - ImageNet normalization:
      mean = [0.485, 0.456, 0.406]
      std  = [0.229, 0.224, 0.225]

Training transforms include mild random augmentations (flip, rotation, jitter).
Validation and test transforms are strictly deterministic.
All transforms ensure safe conversion of Grayscale and RGBA images to 3-channel RGB.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torchvision import transforms
from PIL import Image

# ── Standard ViT ImageNet constants ──────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = 224


class EnsureRGB:
    """Safely converts any PIL image (grayscale, RGBA, CMYK, etc.) to 3-channel RGB."""

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def get_train_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    horizontal_flip: bool = True,
    flip_prob: float = 0.5,
    rotation_degrees: int = 15,
    color_jitter_params: Mapping[str, float] | None = None,
    random_erasing: bool = False,
    erasing_prob: float = 0.1,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> transforms.Compose:
    """
    Construct training-only augmentations.

    Includes resize/crop, horizontal flip, slight rotation, color jitter,
    RGB conversion, ToTensor, and ImageNet normalization.
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    transform_list: list[Any] = [
        EnsureRGB(),
        # Resize slightly larger then randomly crop to target size, or random resized crop
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    ]

    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))

    if rotation_degrees > 0:
        transform_list.append(
            transforms.RandomRotation(
                degrees=rotation_degrees,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )

    if color_jitter_params:
        transform_list.append(
            transforms.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.2),
                contrast=color_jitter_params.get("contrast", 0.2),
                saturation=color_jitter_params.get("saturation", 0.2),
                hue=color_jitter_params.get("hue", 0.1),
            )
        )
    else:
        # Default mild color jitter
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if random_erasing:
        transform_list.append(
            transforms.RandomErasing(
                p=erasing_prob, scale=(0.02, 0.2), value="random"
            )
        )

    return transforms.Compose(transform_list)


def get_val_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> transforms.Compose:
    """
    Construct deterministic validation/test preprocessing.

    NO random augmentations are applied.
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    return transforms.Compose([
        EnsureRGB(),
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_test_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> transforms.Compose:
    """Test preprocessing is identical to validation preprocessing (deterministic)."""
    return get_val_transforms(image_size=image_size, mean=mean, std=std)


# Evaluation transforms alias (eval == val)
get_eval_transforms = get_val_transforms


def get_transforms(
    split: str,
    config: Mapping[str, Any] | None = None,
) -> transforms.Compose:
    """
    Factory function to retrieve transforms for 'train', 'val', or 'test'.

    Parameters from config (if provided) take precedence.
    """
    split = split.lower().strip()
    config = config or {}

    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {})
    norm_cfg = config.get("normalization", {})

    image_size = data_cfg.get("image_size", DEFAULT_IMAGE_SIZE)
    mean = norm_cfg.get("mean", IMAGENET_MEAN)
    std = norm_cfg.get("std", IMAGENET_STD)

    if split == "train":
        return get_train_transforms(
            image_size=image_size,
            horizontal_flip=aug_cfg.get("random_horizontal_flip", True),
            flip_prob=aug_cfg.get("horizontal_flip_prob", 0.5),
            rotation_degrees=aug_cfg.get("random_rotation", 15),
            color_jitter_params=aug_cfg.get("color_jitter"),
            random_erasing=aug_cfg.get("random_erasing", False),
            erasing_prob=aug_cfg.get("random_erasing_prob", 0.1),
            mean=mean,
            std=std,
        )
    elif split in ("val", "validation", "test"):
        return get_val_transforms(image_size=image_size, mean=mean, std=std)
    else:
        raise ValueError(
            f"Unknown split '{split}'. Expected 'train', 'val', or 'test'."
        )


# Backward-compatible aliases for existing Phase 1/2 references
TRAIN_TRANSFORMS = get_train_transforms()
VAL_TRANSFORMS = get_val_transforms()
TEST_TRANSFORMS = get_test_transforms()
