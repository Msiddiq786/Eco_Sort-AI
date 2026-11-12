"""
Tests for EcoSort AI V2 Multi-Object Waste Detection and Recycling Decision Engine.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.dependencies import get_model_registry
from app.main import app
from app.ml.loader import ModelRegistry
from app.services.recycling_engine import (
    DisposalStatus,
    get_recycling_decision,
)


def test_recycling_decision_engine_rules():
    """Verify rules for all waste categories."""
    # Plastic
    dec_plastic = get_recycling_decision("plastic", confidence=0.85)
    assert dec_plastic.disposal_status == DisposalStatus.CONDITIONALLY_RECYCLABLE
    assert "rinse" in dec_plastic.action.lower()
    assert len(dec_plastic.preparation_steps) > 0

    # Metal
    dec_metal = get_recycling_decision("metal", confidence=0.90)
    assert dec_metal.disposal_status == DisposalStatus.RECYCLABLE
    assert "metal" in dec_metal.disposal_bin.lower() or "blue" in dec_metal.disposal_bin.lower()

    # Organic
    dec_organic = get_recycling_decision("organic", confidence=0.92)
    assert dec_organic.disposal_status == DisposalStatus.COMPOSTABLE
    assert "green" in dec_organic.disposal_bin.lower()

    # Low confidence / OOD routes to UNKNOWN_REQUIRES_VERIFICATION
    dec_ood = get_recycling_decision("plastic", confidence=0.20, is_ood=True)
    assert dec_ood.disposal_status == DisposalStatus.UNKNOWN_REQUIRES_VERIFICATION
    assert "low" in dec_ood.reason.lower() or "out-of-distribution" in dec_ood.reason.lower()


def test_detect_endpoint_requires_models_loaded():
    """POST /api/detect returns 503 when models are not loaded."""
    unloaded_registry = ModelRegistry()
    unloaded_registry._vit_loaded = False
    unloaded_registry._clip_loaded = False

    app.dependency_overrides[get_model_registry] = lambda: unloaded_registry
    try:
        buf = io.BytesIO()
        Image.new("RGB", (100, 100), color=(120, 120, 120)).save(buf, format="JPEG")
        buf.seek(0)

        with TestClient(app) as client:
            resp = client.post(
                "/api/detect",
                files={"file": ("test.jpg", buf, "image/jpeg")},
            )
        assert resp.status_code == 503
        assert "not loaded" in resp.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_model_registry, None)


def test_detect_endpoint_successful_detection():
    """POST /api/detect returns valid DetectResponse with bounding boxes and recycling summary."""
    mock_vit_model = MagicMock()
    mock_vit_proc = MagicMock()

    # Mock ViT predict
    mock_vit_probs = {
        "cardboard": 0.05,
        "glass": 0.05,
        "metal": 0.05,
        "paper": 0.05,
        "plastic": 0.75,
        "trash": 0.05,
    }

    loaded_registry = ModelRegistry()
    loaded_registry._vit_loaded = True
    loaded_registry._clip_loaded = False
    loaded_registry.vit_model = mock_vit_model
    loaded_registry.vit_processor = mock_vit_proc

    app.dependency_overrides[get_model_registry] = lambda: loaded_registry
    try:
        # Create an image with two distinct colored rectangles
        img = Image.new("RGB", (200, 200), color=(240, 240, 240))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 80, 80], fill=(20, 20, 20))
        draw.rectangle([110, 110, 170, 170], fill=(50, 50, 50))

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        # Patch predict_vit to return mock_vit_probs
        from unittest.mock import patch
        with patch("app.ml.detector.predict_vit", return_value=mock_vit_probs):
            with TestClient(app) as client:
                resp = client.post(
                    "/api/detect",
                    files={"file": ("test_multi.jpg", buf, "image/jpeg")},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert "total_objects" in data
        assert data["total_objects"] >= 1
        assert "objects" in data
        first_obj = data["objects"][0]
        assert "bbox" in first_obj
        assert "x1" in first_obj["bbox"]
        assert "detection_confidence" in first_obj
        assert "material_category" in first_obj
        assert "recycling_status" in first_obj
        assert "annotated_image_base64" in data
        assert data["annotated_image_base64"].startswith("data:image/jpeg;base64,")
        assert "recycling_summary" in data
    finally:
        app.dependency_overrides.pop(get_model_registry, None)
