"""
Secure image upload utilities.

Responsibilities:
- Validate uploaded files (extension, size, magic bytes).
- Generate UUID-safe filenames.
- Persist image bytes under the configured uploads directory.

Never trust the original filename as a filesystem path.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

# Allowed extensions (lowercase, without leading dot)
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"jpg", "jpeg", "png", "webp"})

# Magic byte signatures for image types we accept
_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"RIFF", "webp"),          # RIFF....WEBP — checked further below
    (b"WEBP", "webp"),
]

MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB


def _sanitize_filename(name: str) -> str:
    """
    Strip path components and replace unsafe characters.

    Keeps alphanumerics, hyphens, underscores, and dots.
    """
    # Take only the base name (guard against path traversal)
    name = Path(name).name
    # Replace anything outside the safe set with an underscore
    name = re.sub(r"[^\w.\-]", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name).strip("_.")
    return name or "upload"


def validate_upload(image_bytes: bytes, original_filename: str) -> None:
    """
    Raise HTTP 400 if the file is too large, has a bad extension, or fails
    magic-byte validation.

    Does NOT touch the filesystem.
    """
    # 1. Size check
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds the maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )

    # 2. Extension check
    ext = (original_filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File extension '.{ext}' is not allowed. "
                f"Accepted types: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    # 3. Magic byte check (at least 12 bytes needed)
    if len(image_bytes) < 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is too small to be a valid image.",
        )

    header = image_bytes[:12]
    detected: str | None = None
    for magic, kind in _MAGIC:
        if header.startswith(magic):
            detected = kind
            break

    # Special-case WebP: "RIFF" at 0 and "WEBP" at offset 8
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        detected = "webp"

    if detected is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content does not match a supported image format (JPEG, PNG, WebP).",
        )


def save_upload(
    image_bytes: bytes,
    original_filename: str,
    upload_dir: Path,
) -> tuple[str, str]:
    """
    Write *image_bytes* to *upload_dir* under a UUID-prefixed safe filename.

    Returns ``(safe_filename, relative_image_path)`` where
    *relative_image_path* is ``"uploads/<safe_filename>"``.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)

    sanitized = _sanitize_filename(original_filename)
    safe_name = f"{uuid.uuid4().hex}_{sanitized}"
    dest = upload_dir / safe_name
    dest.write_bytes(image_bytes)

    relative_path = f"uploads/{safe_name}"
    return safe_name, relative_path
