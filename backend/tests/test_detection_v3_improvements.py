"""
Regression and acceptance tests for EcoSort AI V3 Multi-Object Detection Improvements:
- CWD-independent detector initialization
- Non-merging containment suppression invariant (never merge 2+ distinct kept items)
- Edge-ghost bottle duplicate suppression
- Aspect-ratio-preserving crop padding
- Normalized bbox coordinates & independent Detection / ViT / CLIP / Ensemble scores
- Paper vs Cardboard disambiguation under ViT overconfidence
- TACO/COCO -> YOLO dataset export validation
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from app.ml.detector import (
    WasteObjectDetector,
    _prepare_crop_for_classification,
    apply_containment_and_nms,
    detect_and_classify_objects,
)
from training.prepare_detection_dataset import export_coco_to_yolo, validate_coco_annotations


def test_cwd_independence_finds_yolo_detector(monkeypatch, tmp_path):
    """Changing CWD (e.g. to backend/ or tmp) must still resolve local YOLO weights."""
    monkeypatch.chdir(tmp_path)
    detector = WasteObjectDetector()
    assert detector.model_status != "fallback_multiscale_proposal", (
        f"Expected cached YOLO detector even from {tmp_path}, got {detector.model_status}"
    )


def test_containment_nms_never_merges_two_distinct_objects():
    """An oversized box enclosing >= 2 already kept objects must be rejected, not merge them."""
    proposals = [
        ((10.0, 20.0, 110.0, 300.0), 0.92, "coco_bottle"),
        ((130.0, 25.0, 230.0, 300.0), 0.89, "coco_bottle"),
        ((5.0, 15.0, 235.0, 305.0), 0.95, "morph_obj"),  # Giant merged box
    ]
    kept = apply_containment_and_nms(proposals, total_area=500 * 500)
    assert len(kept) == 2
    boxes = [k[0] for k in kept]
    assert (10.0, 20.0, 110.0, 300.0) in boxes
    assert (130.0, 25.0, 230.0, 300.0) in boxes


def test_containment_nms_suppresses_edge_ghost_duplicate():
    """Low-confidence shifted bottle box touching the right boundary of an existing bottle is suppressed."""
    proposals = [
        ((889.0, 296.0, 989.0, 676.0), 0.90, "coco_bottle"),
        ((953.0, 309.0, 1024.0, 672.0), 0.52, "coco_bottle"),
    ]
    kept = apply_containment_and_nms(
        proposals,
        total_area=1024 * 687,
        image_width=1024.0,
        image_height=687.0,
    )
    assert len(kept) == 1
    assert kept[0][0] == (889.0, 296.0, 989.0, 676.0)


def test_aspect_preserving_crop_padding():
    """Elongated crops (>1.35 ratio) are padded to square; near-square crops remain unchanged."""
    tall_bottle = Image.new("RGB", (80, 320), color=(40, 130, 60))
    padded = _prepare_crop_for_classification(tall_bottle)
    assert padded.size == (320, 320)

    square_tray = Image.new("RGB", (220, 200), color=(230, 230, 230))
    untouched = _prepare_crop_for_classification(square_tray)
    assert untouched.size == (220, 200)


def test_normalized_bbox_and_independent_scores_in_response():
    """DetectResponse populates nx1..ny2 in [0,1] and independent ViT, CLIP, Detection, Ensemble scores."""
    img = Image.new("RGB", (400, 200), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 160, 160], fill=(30, 90, 160))

    vit_probs = {"cardboard": 0.05, "glass": 0.85, "metal": 0.03, "paper": 0.02, "plastic": 0.03, "trash": 0.02}
    clip_probs = {"cardboard": 0.02, "glass": 0.90, "metal": 0.02, "paper": 0.02, "plastic": 0.02, "organic": 0.01, "trash": 0.01}

    mock_vit = MagicMock()
    mock_vit.config.num_labels = 6

    with patch("app.ml.detector.predict_vit", return_value=vit_probs), \
         patch("app.ml.detector.predict_clip_v2", return_value=clip_probs):
        resp = detect_and_classify_objects(
            img, vit_model=mock_vit, vit_processor=MagicMock(),
            clip_model=MagicMock(), clip_processor=MagicMock(),
        )

    assert resp.total_objects >= 1
    obj = resp.objects[0]
    assert 0.0 <= obj.bbox.nx1 < obj.bbox.nx2 <= 1.0
    assert 0.0 <= obj.bbox.ny1 < obj.bbox.ny2 <= 1.0
    assert obj.vit_confidence == 0.85
    assert obj.clip_confidence == 0.90
    assert obj.raw_ensemble_confidence is not None
    assert obj.vit_probabilities is not None
    assert obj.clip_probabilities is not None


def test_paper_not_overridden_by_overconfident_cardboard_vit():
    """When ViT guesses cardboard (0.85) but CLIP strongly sees paper (0.92), balanced crop fusion preserves paper."""
    img = Image.new("RGB", (300, 300), color=(240, 240, 240))
    ImageDraw.Draw(img).rectangle([50, 50, 220, 220], fill=(220, 220, 220))
    # Add texture contrast so detector triggers on synthetic crop
    for x in range(60, 210, 15):
        ImageDraw.Draw(img).line([(x, 60), (x + 10, 210)], fill=(120, 120, 120), width=3)

    vit_probs = {"cardboard": 0.85, "glass": 0.02, "metal": 0.02, "paper": 0.07, "plastic": 0.02, "trash": 0.02}
    clip_probs = {"cardboard": 0.03, "glass": 0.01, "metal": 0.01, "paper": 0.92, "plastic": 0.01, "organic": 0.01, "trash": 0.01}

    mock_vit = MagicMock()
    mock_vit.config.num_labels = 6

    with patch("app.ml.detector.predict_vit", return_value=vit_probs), \
         patch("app.ml.detector.predict_clip_v2", return_value=clip_probs):
        resp = detect_and_classify_objects(
            img, vit_model=mock_vit, vit_processor=MagicMock(),
            clip_model=MagicMock(), clip_processor=MagicMock(),
        )

    assert resp.total_objects >= 1
    obj = resp.objects[0]
    assert obj.material_category == "paper"
    assert obj.model_agreement is False
    assert obj.confidence_status == "ambiguous"


def test_coco_yolo_dataset_export_roundtrip(tmp_path: Path):
    """Dataset preparation validates COCO JSON and exports normalized YOLO labels + data.yaml."""
    img_dir = tmp_path / "raw_imgs"
    img_dir.mkdir()
    img1 = img_dir / "scene1.jpg"
    img2 = img_dir / "scene2.jpg"
    Image.new("RGB", (200, 100), color=(100, 100, 100)).save(img1)
    Image.new("RGB", (200, 100), color=(150, 120, 90)).save(img2)

    coco = {
        "images": [
            {"id": 1, "file_name": "scene1.jpg", "width": 200, "height": 100},
            {"id": 2, "file_name": "scene2.jpg", "width": 200, "height": 100},
        ],
        "categories": [
            {"id": 1, "name": "Glass bottle"},
            {"id": 2, "name": "Foam tray"},
            {"id": 3, "name": "Paper bag"},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [20, 10, 60, 80]},
            {"id": 11, "image_id": 2, "category_id": 2, "bbox": [10, 10, 80, 60]},
            {"id": 12, "image_id": 2, "category_id": 3, "bbox": [110, 20, 60, 60]},
        ],
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(coco), encoding="utf-8")

    summary = validate_coco_annotations(json_path, img_dir)
    assert summary["valid_annotations"] == 3
    assert summary["ecosort_category_distribution"]["glass"] == 1
    assert summary["ecosort_category_distribution"]["trash"] == 1
    assert summary["ecosort_category_distribution"]["paper"] == 1

    out_dir = tmp_path / "yolo_out"
    yaml_path = export_coco_to_yolo(json_path, img_dir, output_dir=out_dir)
    assert yaml_path.exists()
    lbl_files = list(out_dir.rglob("*.txt"))
    assert len(lbl_files) == 2
