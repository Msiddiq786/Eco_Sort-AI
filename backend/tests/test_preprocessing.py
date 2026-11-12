"""
Preprocessing unit tests.
"""
import io
import pytest
from PIL import Image


def _make_jpeg_bytes(w=64, h=64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_load_image_pil_from_bytes():
    from app.ml.preprocessor import load_image_pil

    img = load_image_pil(_make_jpeg_bytes())
    assert img.mode == "RGB"
    assert img.size == (64, 64)


def test_validate_image_valid():
    from app.ml.preprocessor import validate_image

    validate_image(_make_jpeg_bytes())  # should not raise


def test_validate_image_too_large():
    from app.ml.preprocessor import validate_image

    large_bytes = b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024)
    with pytest.raises(ValueError, match="exceeds maximum"):
        validate_image(large_bytes, max_size_mb=10)


def test_validate_image_invalid():
    from app.ml.preprocessor import validate_image

    with pytest.raises(ValueError, match="Cannot read image"):
        validate_image(b"not an image at all")
