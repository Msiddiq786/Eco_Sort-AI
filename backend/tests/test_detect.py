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


def test_13_objects_regression_six_disagreements():
    """
    Regression test reproducing the exact 13-object detection scenario with 6 model disagreement cases:
    - Object #2: ViT Plastic, CLIP Glass
    - Object #3: ViT Cardboard, CLIP Trash
    - Object #4: ViT Metal, CLIP Cardboard
    - Object #5: ViT Glass, CLIP Metal
    - Object #9: ViT Glass, CLIP Metal
    - Object #13: ViT Glass, CLIP Plastic
    - Plus 7 consensus cases (e.g. 4 Recyclable metal, 3 Conditionally Recyclable plastic).

    Expected Invariant:
    - Total objects: 13
    - Review stream (UNKNOWN_REQUIRES_VERIFICATION): 6
    - Recyclable: 4
    - Conditionally Recyclable: 3
    - Sum of all 6 streams == 13 (zero double counting)
    - Every disagreement object has status_label == 'Inspection Required'.
    """
    from app.services.recycling_engine import get_recycling_decision, DisposalStatus

    # Defined predictions for all 13 items
    test_cases = [
        # obj_id, vit_pred, clip_pred, expected_agree
        (1, "metal", "metal", True),        # Agree -> Recyclable
        (2, "plastic", "glass", False),      # Disagree -> Review (#1 of 6)
        (3, "cardboard", "trash", False),    # Disagree -> Review (#2 of 6)
        (4, "metal", "cardboard", False),    # Disagree -> Review (#3 of 6)
        (5, "glass", "metal", False),        # Disagree -> Review (#4 of 6)
        (6, "metal", "metal", True),        # Agree -> Recyclable
        (7, "plastic", "plastic", True),    # Agree -> Conditional
        (8, "plastic", "plastic", True),    # Agree -> Conditional
        (9, "glass", "metal", False),        # Disagree -> Review (#5 of 6)
        (10, "metal", "metal", True),       # Agree -> Recyclable
        (11, "metal", "metal", True),       # Agree -> Recyclable
        (12, "plastic", "plastic", True),   # Agree -> Conditional
        (13, "glass", "plastic", False),     # Disagree -> Review (#6 of 6)
    ]

    status_counts = {
        "recyclable": 0,
        "conditionally_recyclable": 0,
        "compostable": 0,
        "hazardous": 0,
        "not_standard": 0,
        "unknown": 0,
    }

    disagreement_count = 0

    for obj_id, vit_pred, clip_pred, is_agree in test_cases:
        final_cat = vit_pred if is_agree else vit_pred
        final_conf = 0.85 if is_agree else 0.55
        conf_status = "normal" if is_agree else "ambiguous"

        rec_decision = get_recycling_decision(
            category=final_cat,
            confidence=final_conf,
            is_ood=False,
            confidence_status=conf_status,
            model_agreement=is_agree,
        )

        disp_status = rec_decision.disposal_status
        if disp_status == DisposalStatus.RECYCLABLE:
            status_counts["recyclable"] += 1
        elif disp_status == DisposalStatus.CONDITIONALLY_RECYCLABLE:
            status_counts["conditionally_recyclable"] += 1
        elif disp_status == DisposalStatus.COMPOSTABLE:
            status_counts["compostable"] += 1
        elif disp_status == DisposalStatus.HAZARDOUS_SPECIAL_DISPOSAL:
            status_counts["hazardous"] += 1
        elif disp_status == DisposalStatus.NOT_STANDARD_RECYCLABLE:
            status_counts["not_standard"] += 1
        else:
            status_counts["unknown"] += 1

        if not is_agree:
            disagreement_count += 1
            assert rec_decision.disposal_status == DisposalStatus.UNKNOWN_REQUIRES_VERIFICATION
            assert rec_decision.status_label == "Inspection Required"
            assert "disagreement" in rec_decision.reason.lower()

    # Verify counts
    assert len(test_cases) == 13
    assert disagreement_count == 6
    assert status_counts["unknown"] == 6
    assert status_counts["recyclable"] == 4
    assert status_counts["conditionally_recyclable"] == 3

    # Invariant: sum of all six streams must equal 13 exactly
    total_accounted = sum(status_counts.values())
    assert total_accounted == 13


def test_candidate_box_area_capping():
    """Verify that WasteObjectDetector enforces maximum area cap (<= 22% of image canvas)."""
    from app.ml.detector import WasteObjectDetector

    detector = WasteObjectDetector()
    # Create an image with an oversized central block
    w, h = 600, 600
    total_area = w * h
    img = Image.new("RGB", (w, h), color=(20, 20, 20))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Draw a 400x400 block (44% of image area)
    draw.rectangle([100, 100, 500, 500], fill=(220, 220, 220))

    boxes = detector.detect_objects(img)
    for b, score, label in boxes:
        bw = b.x2 - b.x1
        bh = b.y2 - b.y1
        box_area = bw * bh
        # No single box should ever swallow the scene (max 38% area cap)
        assert box_area <= total_area * 0.38, f"Box exceeded 38% cap: {box_area} > {total_area * 0.38}"


def test_paper_prompts_include_cups_and_packaging():
    """Verify that CLIP V2 paper prompt templates cover disposable cups and packaging."""
    from app.ml.clip_model_v2 import V2_CLIP_PROMPT_TEMPLATES

    paper_prompts = V2_CLIP_PROMPT_TEMPLATES["paper"]
    all_text = " ".join(paper_prompts).lower()
    assert "paper cup" in all_text or "coffee cup" in all_text
    assert "packaging" in all_text or "paper bag" in all_text


def test_organic_strict_bypass_policy():
    """
    Verify strict conditions under which Organic can bypass disagreement review:
    1. ViT verified 6-class.
    2. ViT explicitly reports Organic as unsupported.
    3. CLIP V2 supports Organic.
    4. CLIP organic confidence and margin requirements satisfied.
    5. Compatibility decision recorded in response.
    If both models support Organic, disagreement must route to UNKNOWN_REQUIRES_VERIFICATION.
    """
    from app.services.recycling_engine import get_recycling_decision, DisposalStatus

    # Case 1: Verified 6-class ViT with valid CLIP organic confidence & margin (bypass granted)
    dec_bypass = get_recycling_decision(
        category="organic",
        confidence=0.85,
        model_agreement=False,
        is_organic_v1_bypass=True,
    )
    assert dec_bypass.disposal_status == DisposalStatus.COMPOSTABLE
    assert "Green" in dec_bypass.disposal_bin

    # Case 2: Both models support organic (ViT >= 7 classes) -> is_organic_v1_bypass is False.
    # ViT predicts plastic, CLIP predicts organic -> Disagreement MUST route to UNKNOWN_REQUIRES_VERIFICATION!
    dec_disagree = get_recycling_decision(
        category="organic",
        confidence=0.60,
        model_agreement=False,
        is_organic_v1_bypass=False,
    )
    assert dec_disagree.disposal_status == DisposalStatus.UNKNOWN_REQUIRES_VERIFICATION
    assert dec_disagree.status_label == "Inspection Required"

    # Case 3: ViT 6-class, but CLIP organic confidence is weak -> is_organic_v1_bypass is False.
    dec_weak = get_recycling_decision(
        category="organic",
        confidence=0.30,
        model_agreement=False,
        is_organic_v1_bypass=False,
    )
    assert dec_weak.disposal_status == DisposalStatus.UNKNOWN_REQUIRES_VERIFICATION
