"""
Tests for EcoSort AI V3 Sprint Features:
1. Batched ViT + CLIP crop inference and test monkey-patch compatibility
2. Multi-object /api/detect SQLite history persistence + GET/DELETE /api/history/detections
3. Active-learning /api/feedback submission & archival
4. Resin Code (#1-#7) & hazard analysis module
5. HF TACO ingestion & feedback crop merging helpers in prepare_detection_dataset.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.core.dependencies import get_model_registry
from app.main import app
from app.ml.clip_model_v2 import V2_WASTE_CLASSES, predict_clip_v2_batch
from app.ml.detector import detect_and_classify_objects
from app.ml.resin_ocr import analyze_resin_and_hazards
from app.ml.vit_model import predict_vit_batch
from app.schemas.detect import BoundingBox


def _make_jpeg_bytes(color=(30, 140, 210), size=(240, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_fake_vit(num_classes: int = 6):
    calls = {"count": 0}

    class FakeViTModel:
        config = SimpleNamespace(num_labels=num_classes)

        def __call__(self, pixel_values=None, **kwargs):
            calls["count"] += 1
            batch_n = pixel_values.shape[0] if pixel_values is not None else 1
            logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 2.5, 0.1]] * batch_n, dtype=torch.float32)
            return SimpleNamespace(logits=logits)

    class FakeViTProcessor:
        def __call__(self, images=None, return_tensors="pt"):
            imgs = images if isinstance(images, list) else [images]
            return SimpleNamespace(
                pixel_values=torch.zeros((len(imgs), 3, 224, 224), dtype=torch.float32),
                to=lambda device: {"pixel_values": torch.zeros((len(imgs), 3, 224, 224), dtype=torch.float32)},
            )

    return FakeViTModel(), FakeViTProcessor(), calls


def _make_fake_clip():
    calls = {"image_calls": 0}

    class FakeCLIPModel:
        device = torch.device("cpu")
        logit_scale = torch.tensor(np.log(20.0), dtype=torch.float32)

        def get_text_features(self, **kwargs):
            n = kwargs["input_ids"].shape[0]
            torch.manual_seed(7)
            feats = torch.randn((n, 16), dtype=torch.float32)
            return feats / feats.norm(dim=-1, keepdim=True)

        def get_image_features(self, pixel_values=None, **kwargs):
            calls["image_calls"] += 1
            batch_n = pixel_values.shape[0] if pixel_values is not None else 1
            feats = torch.ones((batch_n, 16), dtype=torch.float32)
            feats[:, 0] = 3.0
            return feats / feats.norm(dim=-1, keepdim=True)

    class FakeCLIPProcessor:
        def __call__(self, text=None, images=None, return_tensors="pt", padding=True):
            out = {}
            if text is not None:
                out["input_ids"] = torch.ones((len(text), 4), dtype=torch.long)
            if images is not None:
                imgs = images if isinstance(images, list) else [images]
                out["pixel_values"] = torch.zeros((len(imgs), 3, 224, 224), dtype=torch.float32)
            return out

    return FakeCLIPModel(), FakeCLIPProcessor(), calls


def test_predict_vit_batch_single_forward_pass():
    model, proc, calls = _make_fake_vit()
    assert predict_vit_batch(model, proc, []) == []

    imgs = [Image.new("RGB", (64, 64), c) for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
    res = predict_vit_batch(model, proc, imgs)
    assert calls["count"] == 1
    assert len(res) == 3
    for probs in res:
        assert abs(sum(probs.values()) - 1.0) < 1e-4
        assert max(probs, key=probs.get) == "plastic"


def test_predict_clip_v2_batch_single_forward_pass():
    model, proc, calls = _make_fake_clip()
    assert predict_clip_v2_batch(model, proc, []) == []

    imgs = [Image.new("RGB", (64, 64), c) for c in [(200, 100, 50), (50, 150, 200)]]
    res = predict_clip_v2_batch(model, proc, imgs)
    assert calls["image_calls"] == 1
    assert len(res) == 2
    for item in res:
        assert set(item.probabilities.keys()) == set(V2_WASTE_CLASSES)
        assert abs(sum(item.probabilities.values()) - 1.0) < 1e-5


def test_detect_and_classify_uses_batched_inference_and_respects_patches():
    vit_model, vit_proc, vit_calls = _make_fake_vit()
    clip_model, clip_proc, clip_calls = _make_fake_clip()
    img = Image.new("RGB", (320, 240), color=(240, 240, 240))

    fake_boxes = [
        (BoundingBox(x1=10, y1=10, x2=100, y2=120, nx1=0.03, ny1=0.04, nx2=0.31, ny2=0.5), 0.91, "bottle"),
        (BoundingBox(x1=150, y1=20, x2=280, y2=180, nx1=0.47, ny1=0.08, nx2=0.87, ny2=0.75), 0.88, "can"),
    ]

    with patch("app.ml.detector.get_detector") as mock_det:
        mock_det.return_value.model_status = "yolo_test"
        mock_det.return_value.detect_objects.return_value = fake_boxes

        resp = detect_and_classify_objects(
            pil_image=img,
            vit_model=vit_model,
            vit_processor=vit_proc,
            clip_model=clip_model,
            clip_processor=clip_proc,
        )

    assert resp.total_objects == 2
    assert vit_calls["count"] == 1
    # 1 batched call for crop classification + up to 1 batched call for resin/hazard screening
    assert clip_calls["image_calls"] <= 2

    # Verify monkey-patching predict_vit still works seamlessly in unit tests
    patched_vit_calls = {"count": 0}

    def custom_vit(m, p, crop):
        patched_vit_calls["count"] += 1
        return {"cardboard": 0.8, "glass": 0.04, "metal": 0.04, "paper": 0.04, "plastic": 0.04, "trash": 0.04}

    with (
        patch("app.ml.detector.get_detector") as mock_det,
        patch("app.ml.detector.predict_vit", side_effect=custom_vit),
    ):
        mock_det.return_value.model_status = "yolo_test"
        mock_det.return_value.detect_objects.return_value = fake_boxes
        resp_patched = detect_and_classify_objects(
            pil_image=img,
            vit_model=vit_model,
            vit_processor=vit_proc,
            clip_model=None,
            clip_processor=None,
        )
    assert patched_vit_calls["count"] == 2
    assert resp_patched.objects[0].material_category == "cardboard"


def test_detect_persistence_history_and_feedback_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "FEEDBACK_DIR", tmp_path / "feedback")

    vit_model, vit_proc, _ = _make_fake_vit()
    fake_registry = SimpleNamespace(
        ready=True,
        vit_model=vit_model,
        vit_processor=vit_proc,
        clip_model=None,
        clip_processor=None,
        clip_loaded=False,
    )
    app.dependency_overrides[get_model_registry] = lambda: fake_registry

    fake_boxes = [
        (BoundingBox(x1=20, y1=20, x2=140, y2=160, nx1=0.08, ny1=0.1, nx2=0.58, ny2=0.8), 0.92, "plastic"),
    ]
    raw_bytes = _make_jpeg_bytes()

    try:
        with TestClient(app) as client:
            with patch("app.ml.detector.get_detector") as mock_det:
                mock_det.return_value.model_status = "yolo_test"
                mock_det.return_value.detect_objects.return_value = fake_boxes
                det_res = client.post(
                    "/api/detect",
                    files={"file": ("mixed_audit.jpg", raw_bytes, "image/jpeg")},
                )
            assert det_res.status_code == 200
            det_data = det_res.json()
            scan_id = det_data["scan_id"]
            assert isinstance(scan_id, int)
            assert det_data["total_objects"] == 1
            assert det_data["objects"][0]["resin_info"]["resin_code"] == "Unverified Plastic"

            # Verify persisted scan shows up in /api/history/detections
            hist_res = client.get("/api/history/detections")
            assert hist_res.status_code == 200
            hist_data = hist_res.json()
            assert any(item["id"] == scan_id for item in hist_data["items"])

            # Submit active-learning feedback correction with crop base64
            crop_b64 = det_data["objects"][0]["crop_image_base64"]
            fb_res = client.post(
                "/api/feedback",
                json={
                    "predicted_category": "plastic",
                    "corrected_category": "glass",
                    "detection_record_id": scan_id,
                    "object_id": 1,
                    "source": "multi_object",
                    "notes": "Green glass bottle misread",
                    "image_base64": crop_b64,
                },
            )
            assert fb_res.status_code == 201
            fb_data = fb_res.json()
            assert fb_data["corrected_category"] == "glass"
            assert Path(fb_data["saved_crop_path"]).is_file()
            assert Path(fb_data["saved_crop_path"]).parent.name == "glass"

            # List feedback summary
            fb_stats = client.get("/api/feedback")
            assert fb_stats.status_code == 200
            assert fb_stats.json()["by_corrected_category"].get("glass", 0) >= 1

            # Invalid category is rejected
            bad_fb = client.post(
                "/api/feedback",
                json={"predicted_category": "plastic", "corrected_category": "plutonium"},
            )
            assert bad_fb.status_code == 422

            # Delete persisted detection audit
            del_res = client.delete(f"/api/history/detections/{scan_id}")
            assert del_res.status_code == 204
    finally:
        app.dependency_overrides.clear()


def test_resin_ocr_category_gating_and_clip_classification():
    img = Image.new("RGB", (128, 128), color=(210, 230, 250))
    assert analyze_resin_and_hazards(img, "organic") is None
    assert analyze_resin_and_hazards(img, "cardboard") is None

    # No CLIP & no Tesseract -> returns safe Unverified Plastic fallback instead of crashing
    fallback = analyze_resin_and_hazards(img, "plastic", clip_model=None, clip_processor=None)
    assert fallback is not None
    assert fallback["resin_code"] == "Unverified Plastic"

    # Mocked CLIP returning PET #1 dominant confidence
    model, proc, _ = _make_fake_clip()
    res_clip = analyze_resin_and_hazards(img, "plastic", clip_model=model, clip_processor=proc)
    assert res_clip is not None
    assert res_clip["resin_code"] in {"Unverified Plastic", "#1 PET", "#2 HDPE", "#3 PVC", "#4 LDPE", "#5 PP", "#6 PS", "#7 Other"}


def test_prepare_detection_dataset_coco_export_and_feedback_merge(tmp_path):
    training_dir = Path(__file__).resolve().parent.parent / "training"
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    from prepare_detection_dataset import export_coco_to_yolo, merge_feedback_crops_into_train

    raw_root = tmp_path / "taco_raw"
    (raw_root / "batch_1").mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (200, 160), (50 * i + 20, 100, 150)).save(raw_root / "batch_1" / f"img_{i}.jpg")

    coco = {
        "images": [
            {"id": i + 1, "file_name": f"batch_1/img_{i}.jpg", "width": 200, "height": 160}
            for i in range(4)
        ],
        "categories": [
            {"id": 1, "name": "Glass bottle"},
            {"id": 2, "name": "Clear plastic bottle"},
        ],
        "annotations": [
            {"id": i + 1, "image_id": i + 1, "category_id": 1 if i % 2 == 0 else 2, "bbox": [20, 20, 80, 100]}
            for i in range(4)
        ],
    }
    json_path = raw_root / "annotations.json"
    json_path.write_text(json.dumps(coco), encoding="utf-8")

    out_yolo = tmp_path / "yolo_out"
    yaml_path = export_coco_to_yolo(json_path, raw_root, output_dir=out_yolo)
    assert yaml_path.is_file()

    # Create mock feedback crop and merge into train split
    fb_dir = tmp_path / "feedback" / "glass"
    fb_dir.mkdir(parents=True)
    Image.new("RGB", (96, 96), (20, 160, 40)).save(fb_dir / "fb_glass1.jpg")
    added = merge_feedback_crops_into_train(tmp_path / "feedback", output_dir=out_yolo)
    assert added == 1
    assert (out_yolo / "train" / "labels" / "feedback_glass_fb_glass1.txt").is_file()
