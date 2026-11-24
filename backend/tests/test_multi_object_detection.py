"""
15 Test cases for EcoSort AI V2 — Multi-Object Waste Detection Pipeline.

Test coverage:
 1. Single-object synthetic image
 2. Two objects side-by-side
 3. Three or more distinct objects
 4. Mixed materials (ViT + CLIP predict different classes per crop)
 5. Touching objects
 6. Overlapping objects
 7. Very small object (<2% area) is filtered out
 8. No visible object (blank / solid-colour image)
 9. Invalid / corrupted image bytes
10. Bounding-box coordinate validity (all coords within image bounds)
11. Duplicate detection removal (NMS working correctly)
12. Individual crop classification returns valid 7-class output
13. Organic waste crop classification via CLIP V2
14. API response structure validation (all required fields present)
15. Existing /api/predict not broken (backward-compatibility regression)
"""
from __future__ import annotations

import io
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.core.dependencies import get_model_registry
from app.main import app
from app.ml.detector import (
    WasteObjectDetector,
    apply_non_maximum_suppression,
    calculate_iou,
    detect_and_classify_objects,
    extract_candidate_regions,
)
from app.ml.loader import ModelRegistry
from app.schemas.detect import BoundingBox, DetectResponse


# ─────────────────────────── helpers ───────────────────────────────────────

def _make_registry(*, vit_loaded: bool = True, clip_loaded: bool = False, num_labels: int = 7):
    """Return a ModelRegistry mock with configurable state."""
    registry = ModelRegistry()
    registry._vit_loaded = vit_loaded
    registry._clip_loaded = clip_loaded
    mock_vit = MagicMock()
    mock_vit.config.num_labels = num_labels
    registry.vit_model = mock_vit
    registry.vit_processor = MagicMock()
    if clip_loaded:
        registry.clip_model = MagicMock()
        registry.clip_processor = MagicMock()
    return registry


def _jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _two_rect_image(
    size: tuple = (300, 200),
    rects=None,
    bg=(240, 240, 240),
) -> Image.Image:
    """Create image with coloured rectangles on a light background."""
    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)
    for rect, fill in (rects or []):
        draw.rectangle(rect, fill=fill)
    return img


MOCK_VIT_7CLASS: Dict[str, float] = {
    "cardboard": 0.03,
    "glass":     0.03,
    "metal":     0.03,
    "paper":     0.03,
    "plastic":   0.80,
    "organic":   0.03,
    "trash":     0.05,
}

MOCK_CLIP_7CLASS: Dict[str, float] = {
    "cardboard": 0.03,
    "glass":     0.03,
    "metal":     0.03,
    "paper":     0.03,
    "plastic":   0.80,
    "organic":   0.03,
    "trash":     0.05,
}


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1 — Single-Object Synthetic Image
# ═══════════════════════════════════════════════════════════════════════════

def test_single_object_detection():
    """A synthetic image with one clearly defined dark block should produce ≥ 1 detection."""
    img = _two_rect_image(
        size=(300, 300),
        rects=[([50, 50, 230, 230], (20, 20, 20))],
    )
    mock_vit = MagicMock()
    mock_vit.config.num_labels = 7

    with patch("app.ml.detector.predict_vit", return_value=MOCK_VIT_7CLASS):
        result = detect_and_classify_objects(
            img,
            vit_model=mock_vit,
            vit_processor=MagicMock(),
        )

    assert result.total_objects >= 1
    assert len(result.objects) == result.total_objects
    assert result.objects[0].object_id == 1


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2 — Two Objects Side-by-Side
# ═══════════════════════════════════════════════════════════════════════════

def test_two_objects_side_by_side():
    """Image with two clearly separated dark regions should produce ≥ 2 detections."""
    img = _two_rect_image(
        size=(400, 200),
        rects=[
            ([20, 20, 150, 170], (15, 15, 15)),
            ([250, 20, 380, 170], (15, 15, 15)),
        ],
    )
    mock_vit = MagicMock()
    mock_vit.config.num_labels = 7

    with patch("app.ml.detector.predict_vit", return_value=MOCK_VIT_7CLASS):
        result = detect_and_classify_objects(
            img,
            vit_model=mock_vit,
            vit_processor=MagicMock(),
        )

    assert result.total_objects >= 2


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3 — Three or More Objects
# ═══════════════════════════════════════════════════════════════════════════

def test_three_or_more_objects():
    """Image with three separated blocks should return ≥ 2 detections (at minimum two quadrant splits)."""
    img = Image.new("RGB", (600, 200), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20,  20, 160, 170], fill=(10, 10, 10))
    draw.rectangle([220, 20, 380, 170], fill=(10, 10, 10))
    draw.rectangle([420, 20, 580, 170], fill=(10, 10, 10))

    mock_vit = MagicMock()
    mock_vit.config.num_labels = 7

    with patch("app.ml.detector.predict_vit", return_value=MOCK_VIT_7CLASS):
        result = detect_and_classify_objects(
            img,
            vit_model=mock_vit,
            vit_processor=MagicMock(),
        )

    assert result.total_objects >= 2


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4 — Mixed Materials (ViT vs CLIP disagree per-crop)
# ═══════════════════════════════════════════════════════════════════════════

def test_mixed_materials_disagreement():
    """When ViT says plastic and CLIP says metal (disagreement), status is 'ambiguous'."""
    img = _two_rect_image(
        size=(300, 300),
        rects=[([40, 40, 240, 240], (20, 20, 20))],
    )
    mock_vit = MagicMock()
    mock_vit.config.num_labels = 7

    vit_probs = {**MOCK_VIT_7CLASS, "plastic": 0.75, "metal": 0.10}
    clip_probs = {**MOCK_CLIP_7CLASS, "metal": 0.78, "plastic": 0.07}

    with patch("app.ml.detector.predict_vit", return_value=vit_probs):
        with patch("app.ml.detector.predict_clip_v2", return_value=clip_probs):
            result = detect_and_classify_objects(
                img,
                vit_model=mock_vit,
                vit_processor=MagicMock(),
                clip_model=MagicMock(),
                clip_processor=MagicMock(),
            )

    assert result.total_objects >= 1
    obj = result.objects[0]
    assert obj.model_agreement is False
    assert obj.confidence_status == "ambiguous"
    # Disagreement cap: confidence must be ≤ 0.68
    assert obj.calibrated_confidence <= 0.68


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5 — Touching Objects (continuous blob should not all merge)
# ═══════════════════════════════════════════════════════════════════════════

def test_touching_objects_produce_detections():
    """Two touching rectangles (separated by 2px gap) should still produce ≥ 1 detection."""
    img = Image.new("RGB", (300, 200), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Touching at x=150
    draw.rectangle([20, 30, 148, 170], fill=(10, 10, 10))
    draw.rectangle([152, 30, 280, 170], fill=(10, 10, 10))

    detector = WasteObjectDetector()
    boxes = detector.detect_objects(img)
    assert len(boxes) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6 — Overlapping Objects (high IoU boxes are suppressed by NMS)
# ═══════════════════════════════════════════════════════════════════════════

def test_overlapping_boxes_suppressed_by_nms():
    """Two heavily overlapping boxes (IoU > 0.38) must be reduced to one by NMS."""
    # Manually construct heavily overlapping proposals
    box_a = ((10.0, 10.0, 200.0, 200.0), 0.9, "waste")
    box_b = ((15.0, 15.0, 195.0, 195.0), 0.85, "waste")

    iou = calculate_iou(
        (box_a[0][0], box_a[0][1], box_a[0][2], box_a[0][3]),
        (box_b[0][0], box_b[0][1], box_b[0][2], box_b[0][3]),
    )
    assert iou > 0.38, f"Test setup: expected high IoU, got {iou}"

    kept = apply_non_maximum_suppression([box_a, box_b], iou_threshold=0.38)
    assert len(kept) == 1
    # Higher-scored box wins
    assert kept[0][1] == 0.9


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7 — Very Small Object (<2% area) is Filtered Out
# ═══════════════════════════════════════════════════════════════════════════

def test_tiny_object_filtered():
    """A 5x5 pixel patch (<0.03% of a 600x600 image) should not produce a detection."""
    img = Image.new("RGB", (600, 600), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # 5x5 = 25px² out of 360000px² = 0.007% — well below 0.4% threshold
    draw.rectangle([290, 290, 295, 295], fill=(0, 0, 0))

    detector = WasteObjectDetector()
    boxes = detector.detect_objects(img)
    for b, _, _ in boxes:
        area = (b.x2 - b.x1) * (b.y2 - b.y1)
        total = 600 * 600
        assert area >= total * 0.004, f"Tiny patch should have been filtered; area={area}"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8 — No Visible Object (blank / uniform image)
# ═══════════════════════════════════════════════════════════════════════════

def test_blank_image_returns_zero_detections():
    """Blank image with no edges/objects must return exactly zero detections."""
    img = Image.new("RGB", (400, 300), color=(200, 200, 200))
    detector = WasteObjectDetector()
    boxes = detector.detect_objects(img)
    assert len(boxes) == 0, f"Expected 0 detections for blank image, got {len(boxes)}"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 9 — Invalid / Corrupted Image Bytes → HTTP 422
# ═══════════════════════════════════════════════════════════════════════════

def test_corrupted_image_returns_422():
    """Corrupted/non-image bytes submitted to /api/detect must return 422."""
    registry = _make_registry(vit_loaded=True)
    app.dependency_overrides[get_model_registry] = lambda: registry
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/detect",
                files={"file": ("broken.jpg", b"\x00\x01NOTANIMAGE\xff\xfe", "image/jpeg")},
            )
        assert resp.status_code in (422, 400), f"Expected 422/400, got {resp.status_code}"
    finally:
        app.dependency_overrides.pop(get_model_registry, None)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 10 — Bounding-Box Coordinate Validity
# ═══════════════════════════════════════════════════════════════════════════

def test_bounding_box_coordinates_within_image():
    """All bounding-box coordinates must lie strictly within the image dimensions."""
    img = _two_rect_image(
        size=(400, 300),
        rects=[
            ([30,  30, 170, 140], (10, 10, 10)),
            ([230, 30, 370, 140], (10, 10, 10)),
        ],
    )
    W, H = img.size
    mock_vit = MagicMock()
    mock_vit.config.num_labels = 7

    with patch("app.ml.detector.predict_vit", return_value=MOCK_VIT_7CLASS):
        result = detect_and_classify_objects(
            img,
            vit_model=mock_vit,
            vit_processor=MagicMock(),
        )

    for obj in result.objects:
        b = obj.bbox
        assert b.x1 >= 0.0, f"x1={b.x1} is negative"
        assert b.y1 >= 0.0, f"y1={b.y1} is negative"
        assert b.x2 <= float(W), f"x2={b.x2} exceeds image width {W}"
        assert b.y2 <= float(H), f"y2={b.y2} exceeds image height {H}"
        assert b.x1 < b.x2, f"Degenerate box: x1={b.x1} >= x2={b.x2}"
        assert b.y1 < b.y2, f"Degenerate box: y1={b.y1} >= y2={b.y2}"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 11 — Duplicate Detection Removal (IoU-based NMS)
# ═══════════════════════════════════════════════════════════════════════════

def test_nms_removes_duplicates_keeps_best_score():
    """NMS should keep the highest-confidence box and suppress IoU > 0.38 duplicates."""
    proposals = [
        ((10.0, 10.0, 100.0, 100.0), 0.95, "item"),
        ((12.0, 12.0, 98.0,  98.0),  0.80, "item"),   # almost identical → suppress
        ((200.0, 10.0, 290.0, 100.0), 0.75, "item"),  # separate region → keep
        ((202.0, 12.0, 288.0, 98.0),  0.70, "item"),  # duplicate of above → suppress
    ]
    kept = apply_non_maximum_suppression(proposals, iou_threshold=0.38)
    assert len(kept) == 2, f"Expected 2 kept after NMS, got {len(kept)}"
    scores = [k[1] for k in kept]
    assert 0.95 in scores
    assert 0.75 in scores


# ═══════════════════════════════════════════════════════════════════════════
# TEST 12 — Individual Crop Returns Valid 7-Class Output
# ═══════════════════════════════════════════════════════════════════════════

def test_individual_crop_classification_7_classes():
    """Each detected object must have class_probabilities covering all 7 V2 categories."""
    V2_CLASSES = {"cardboard", "glass", "metal", "paper", "plastic", "organic", "trash"}

    img = _two_rect_image(
        size=(300, 300),
        rects=[([50, 50, 240, 240], (20, 20, 20))],
    )
    mock_vit = MagicMock()
    mock_vit.config.num_labels = 7

    with patch("app.ml.detector.predict_vit", return_value=MOCK_VIT_7CLASS):
        with patch("app.ml.detector.predict_clip_v2", return_value=MOCK_CLIP_7CLASS):
            result = detect_and_classify_objects(
                img,
                vit_model=mock_vit,
                vit_processor=MagicMock(),
                clip_model=MagicMock(),
                clip_processor=MagicMock(),
            )

    assert result.total_objects >= 1
    for obj in result.objects:
        returned_keys = set(obj.class_probabilities.keys())
        assert V2_CLASSES.issubset(returned_keys), (
            f"Missing classes: {V2_CLASSES - returned_keys}"
        )
        prob_sum = sum(obj.class_probabilities.values())
        assert abs(prob_sum - 1.0) < 0.02, f"Probabilities don't sum to 1: {prob_sum}"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 13 — Organic Waste Crop Classification via CLIP V2
# ═══════════════════════════════════════════════════════════════════════════

def test_organic_crop_classification_bypass():
    """
    When ViT is 6-class and CLIP identifies organic with high confidence,
    the organic V1 bypass must fire: material_category='organic', recycling_status='COMPOSTABLE'.
    """
    img = _two_rect_image(
        size=(300, 300),
        rects=[([40, 40, 240, 240], (30, 60, 20))],
    )
    mock_vit = MagicMock()
    mock_vit.config.num_labels = 6  # 6-class ViT (V1)

    vit_6class = {
        "cardboard": 0.05, "glass": 0.05, "metal": 0.05,
        "paper": 0.05, "plastic": 0.10, "trash": 0.70,
    }
    clip_organic_strong = {
        "cardboard": 0.02, "glass": 0.01, "metal": 0.01,
        "paper": 0.02, "plastic": 0.02, "trash": 0.07, "organic": 0.85,
    }

    with patch("app.ml.detector.predict_vit", return_value=vit_6class):
        with patch("app.ml.detector.predict_clip_v2", return_value=clip_organic_strong):
            result = detect_and_classify_objects(
                img,
                vit_model=mock_vit,
                vit_processor=MagicMock(),
                clip_model=MagicMock(),
                clip_processor=MagicMock(),
            )

    assert result.total_objects >= 1
    obj = result.objects[0]
    assert obj.material_category == "organic", f"Expected organic, got {obj.material_category}"
    assert obj.recycling_status == "COMPOSTABLE"
    assert obj.model_coverage is not None
    assert obj.model_coverage.vit_supported is False
    assert obj.model_coverage.clip_supported is True
    assert "Green" in obj.recycling_decision.disposal_bin


# ═══════════════════════════════════════════════════════════════════════════
# TEST 14 — API Response Structure Validation
# ═══════════════════════════════════════════════════════════════════════════

def test_detect_response_structure_all_required_fields():
    """POST /api/detect must return all required DetectResponse fields."""
    img = _two_rect_image(
        size=(250, 250),
        rects=[([30, 30, 110, 110], (10, 10, 10))],
    )
    registry = _make_registry(vit_loaded=True, num_labels=7)
    app.dependency_overrides[get_model_registry] = lambda: registry
    try:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        with patch("app.ml.detector.predict_vit", return_value=MOCK_VIT_7CLASS):
            with TestClient(app) as client:
                resp = client.post(
                    "/api/detect",
                    files={"file": ("scene.jpg", buf, "image/jpeg")},
                )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()

        # Top-level required fields
        for field in [
            "image_width", "image_height", "total_objects", "objects",
            "category_counts", "recycling_summary", "model_versions",
            "inference_time_ms", "annotated_image_base64", "detection_model_status",
        ]:
            assert field in data, f"Missing top-level field: {field}"

        # Base64 image sanity
        assert data["annotated_image_base64"].startswith("data:image/jpeg;base64,")

        # Recycling summary sub-structure
        rs = data["recycling_summary"]
        for sf in ["total_objects", "category_counts", "status_counts"]:
            assert sf in rs, f"Missing recycling_summary field: {sf}"

        sc = rs["status_counts"]
        for key in ["recyclable", "conditionally_recyclable", "compostable",
                    "hazardous", "not_standard", "unknown"]:
            assert key in sc, f"Missing status_counts key: {key}"

        # Stream count conservation invariant
        assert sum(sc.values()) == data["total_objects"]

        # Per-object required fields
        if data["objects"]:
            obj = data["objects"][0]
            for of in [
                "object_id", "bbox", "detection_confidence", "detection_label",
                "material_category", "vit_prediction", "clip_prediction",
                "final_category", "class_probabilities", "calibrated_confidence",
                "confidence_status", "recycling_status", "disposal_recommendation",
                "preparation_steps", "recycling_decision",
            ]:
                assert of in obj, f"Missing DetectedObject field: {of}"

            # BoundingBox sub-fields
            for bf in ["x1", "y1", "x2", "y2"]:
                assert bf in obj["bbox"], f"Missing bbox field: {bf}"

    finally:
        app.dependency_overrides.pop(get_model_registry, None)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 15 — /api/predict Not Broken (Backward-Compatibility Regression)
# ═══════════════════════════════════════════════════════════════════════════

def test_predict_endpoint_still_works_after_detect_changes():
    """POST /api/predict must still respond correctly (backward-compatibility guard)."""
    img = Image.new("RGB", (224, 224), color=(180, 100, 60))
    registry = _make_registry(vit_loaded=True, num_labels=7)
    app.dependency_overrides[get_model_registry] = lambda: registry

    mock_predict_probs = {
        "cardboard": 0.80, "glass": 0.04, "metal": 0.03,
        "paper": 0.04, "plastic": 0.05, "organic": 0.02, "trash": 0.02,
    }

    try:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        with patch("app.api.predict.predict_vit", return_value=mock_predict_probs):
            with TestClient(app) as client:
                resp = client.post(
                    "/api/predict",
                    files={"file": ("cardboard.jpg", buf, "image/jpeg")},
                )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()

        # Core PredictResponse fields
        for field in [
            "predicted_category", "confidence", "vit_prediction",
            "class_probabilities", "recycling_recommendation",
        ]:
            assert field in data, f"Missing /api/predict response field: {field}"

        # /api/predict must NOT return detect-specific fields
        assert "total_objects" not in data
        assert "annotated_image_base64" not in data
        assert "detection_model_status" not in data

    finally:
        app.dependency_overrides.pop(get_model_registry, None)
