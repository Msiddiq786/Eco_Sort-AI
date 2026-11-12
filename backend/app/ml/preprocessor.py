"""
Image preprocessing for both ViT and CLIP inference.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def load_image_pil(source: Union[str, Path, bytes]) -> Image.Image:
    """
    Load an image from a file path or raw bytes into a PIL Image.
    Handles EXIF orientation and converts to RGB to ensure 3-channel consistency.
    """
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    else:
        img = Image.open(io.BytesIO(source))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def validate_image(image_bytes: bytes, max_size_mb: int = 10) -> None:
    """
    Validate that the bytes represent a readable image and are within size limits.

    Raises:
        ValueError: if the image is invalid or too large.
    """
    mb = len(image_bytes) / (1024 * 1024)
    if mb > max_size_mb:
        raise ValueError(f"Image size {mb:.1f} MB exceeds maximum {max_size_mb} MB.")
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception as exc:
        raise ValueError(f"Cannot read image: {exc}") from exc


def preprocess_for_display(image_bytes: bytes, max_dim: int = 512) -> np.ndarray:
    """
    Resize an image for display purposes. Uses OpenCV if available, otherwise Pillow.
    Not used for model inference — purely for thumbnail generation.
    """
    if cv2 is not None:
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("OpenCV could not decode the image.")
        h, w = img_bgr.shape[:2]
        scale = min(max_dim / h, max_dim / w, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img_bgr
    else:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        return np.array(pil_img)
