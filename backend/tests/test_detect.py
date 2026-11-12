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

        # Verify summary invariant: sum of status counts == total_objects
        sc = data["recycling_summary"]["status_counts"]
        total_sc = sum(sc.values())
        assert total_sc == data["total_objects"]
    finally:
        app.dependency_overrides.pop(get_model_registry, None)


def test_organic_waste_v1_compatibility_and_compostable():
    """Verify that organic waste identified by CLIP V2 is handled with V1 compatibility."""
    from app.ml.detector import detect_and_classify_objects
    from unittest.mock import patch, MagicMock

    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 90, 90], fill=(30, 30, 30))

    mock_vit_model = MagicMock()
    mock_vit_model.config.num_labels = 6
    mock_vit_proc = MagicMock()

    mock_vit_probs = {
        "cardboard": 0.05,
        "glass": 0.05,
        "metal": 0.05,
        "paper": 0.05,
        "plastic": 0.10,
        "trash": 0.70,
    }
    mock_clip_probs = {
        "cardboard": 0.02,
        "glass": 0.02,
        "metal": 0.02,
        "paper": 0.02,
        "plastic": 0.02,
        "trash": 0.05,
        "organic": 0.85,
    }

    with patch("app.ml.detector.predict_vit", return_value=mock_vit_probs):
        with patch("app.ml.detector.predict_clip_v2", return_value=mock_clip_probs):
            res = detect_and_classify_objects(
                img,
                vit_model=mock_vit_model,
                vit_processor=mock_vit_proc,
                clip_model=MagicMock(),
                clip_processor=MagicMock(),
            )

    assert res.total_objects >= 1
    obj = res.objects[0]
    assert obj.material_category == "organic"
    assert obj.recycling_status == "COMPOSTABLE"
    assert obj.model_coverage.vit_supported is False
    assert obj.model_coverage.clip_supported is True
    assert "Green" in obj.recycling_decision.disposal_bin


def test_model_disagreement_triggers_review():
    """Verify that when ViT and CLIP disagree, confidence_status is marked ambiguous."""
    from app.ml.detector import detect_and_classify_objects
    from unittest.mock import patch, MagicMock

    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([25, 25, 85, 85], fill=(40, 40, 40))

    mock_vit_model = MagicMock()
    mock_vit_model.config.num_labels = 6
    mock_vit_proc = MagicMock()

    mock_vit_probs = {
        "cardboard": 0.05,
        "glass": 0.05,
        "metal": 0.05,
        "paper": 0.05,
        "plastic": 0.75,
        "trash": 0.05,
    }
    mock_clip_probs = {
        "cardboard": 0.05,
        "glass": 0.05,
        "metal": 0.75,
        "paper": 0.05,
        "plastic": 0.05,
        "trash": 0.05,
    }

    with patch("app.ml.detector.predict_vit", return_value=mock_vit_probs):
        with patch("app.ml.detector.predict_clip_v2", return_value=mock_clip_probs):
            res = detect_and_classify_objects(
                img,
                vit_model=mock_vit_model,
                vit_processor=mock_vit_proc,
                clip_model=MagicMock(),
                clip_processor=MagicMock(),
            )

    assert res.total_objects >= 1
    obj = res.objects[0]
    assert obj.model_agreement is False
    assert obj.confidence_status == "ambiguous"
    # Verify probability normalization
    assert abs(sum(obj.class_probabilities.values()) - 1.0) < 0.01


def test_waste_object_detector_nms():
    """Verify Non-Maximum Suppression suppresses heavily overlapping bounding boxes."""
    from app.ml.detector import WasteObjectDetector, calculate_iou

    detector = WasteObjectDetector()
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Draw two distinct separated objects
    draw.rectangle([30, 30, 100, 100], fill=(20, 20, 20))
    draw.rectangle([180, 180, 260, 260], fill=(20, 20, 20))

    boxes = detector.detect_objects(img)
    assert len(boxes) >= 2
    # Verify no two boxes have IoU > 0.40
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            box_i = (boxes[i][0].x1, boxes[i][0].y1, boxes[i][0].x2, boxes[i][0].y2)
            box_j = (boxes[j][0].x1, boxes[j][0].y1, boxes[j][0].x2, boxes[j][0].y2)
            iou = calculate_iou(box_i, box_j)
            assert iou <= 0.40
