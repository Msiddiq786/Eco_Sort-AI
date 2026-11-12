"""
Tests for Single-Image Classification Pipeline (EcoSort AI V2.1 — /api/predict).

Verifies:
1. Banana peel -> Organic
2. Apple core -> Organic
3. Watermelon rind -> Organic
4. Vegetable scraps -> Organic
5. Clean cardboard -> Cardboard
6. Plastic bottle -> Plastic
7. Aluminum can -> Metal
8. Glass bottle -> Glass
9. Contaminated packaging -> Trash / Uncertain review
10. Ordinary non-organic trash -> Trash
11. V1 6-class compatibility (ViT reports 'Not supported (V1 6-class)')
12. CLIP V2 7-class mapping (cardboard, glass, metal, paper, plastic, organic, trash)
13. Partial model coverage flag
14. False consensus prevention
15. Low-confidence organic prediction routes to verification
16. Organic recycling decision is COMPOSTABLE with Green Bin
17. Live CLIP prompt classification behavior on organic patterns
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.dependencies import get_model_registry
from app.main import app
from app.ml.clip_model_v2 import CLIPClassificationResultV2, V2_WASTE_CLASSES
from app.ml.loader import ModelRegistry
from app.services.recycling_engine import DisposalStatus


def _create_image_bytes(color: tuple[int, int, int] = (100, 150, 80)) -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", (128, 128), color=color).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _build_mock_registry(
    vit_probs: dict[str, float],
    clip_probs: dict[str, float],
    vit_num_classes: int = 6,
) -> ModelRegistry:
    mock_reg = ModelRegistry()
    mock_reg._vit_loaded = True
    mock_reg._clip_loaded = True

    mock_vit_model = MagicMock()
    mock_vit_model.config = MagicMock()
    mock_vit_model.config.num_labels = vit_num_classes

    mock_reg.vit_model = mock_vit_model
    mock_reg.vit_processor = MagicMock()
    mock_reg.clip_model = MagicMock()
    mock_reg.clip_processor = MagicMock()

    return mock_reg


# ── 1. Banana Peel -> Organic ──────────────────────────────────────────────────
def test_banana_peel_predicts_organic():
    vit_p = {"cardboard": 0.05, "glass": 0.02, "metal": 0.03, "paper": 0.10, "plastic": 0.10, "trash": 0.70}
    clip_p = {
        "cardboard": 0.02, "glass": 0.01, "metal": 0.01, "paper": 0.03,
        "plastic": 0.03, "organic": 0.85, "trash": 0.05,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("organic", 0.85, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("banana_peel.jpg", _create_image_bytes(), "image/jpeg")})
            assert res.status_code == 200
            data = res.json()
            assert data["predicted_category"] == "organic"
            assert data["vit_prediction"] == "Not supported (V1 6-class)"
            assert data["clip_prediction"] == "organic"
            assert data["model_coverage"]["coverage_type"] == "partial"
            assert data["model_coverage"]["compatibility_decision"] == "organic_v1_clip_bypass"
            assert data["recycling_decision"]["disposal_status"] == DisposalStatus.COMPOSTABLE
            assert "green" in data["recycling_decision"]["disposal_bin"].lower()
    finally:
        app.dependency_overrides.clear()


# ── 2. Apple Core -> Organic ───────────────────────────────────────────────────
def test_apple_core_predicts_organic():
    vit_p = {"cardboard": 0.03, "glass": 0.02, "metal": 0.05, "paper": 0.15, "plastic": 0.05, "trash": 0.70}
    clip_p = {
        "cardboard": 0.01, "glass": 0.01, "metal": 0.01, "paper": 0.02,
        "plastic": 0.02, "organic": 0.88, "trash": 0.05,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("organic", 0.88, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("apple_core.jpg", _create_image_bytes(), "image/jpeg")})
            assert res.status_code == 200
            data = res.json()
            assert data["predicted_category"] == "organic"
            assert data["recycling_decision"]["disposal_status"] == DisposalStatus.COMPOSTABLE
    finally:
        app.dependency_overrides.clear()


# ── 3. Watermelon Rind -> Organic ──────────────────────────────────────────────
def test_watermelon_rind_predicts_organic():
    vit_p = {"cardboard": 0.02, "glass": 0.03, "metal": 0.05, "paper": 0.08, "plastic": 0.12, "trash": 0.70}
    clip_p = {
        "cardboard": 0.02, "glass": 0.01, "metal": 0.01, "paper": 0.01,
        "plastic": 0.05, "organic": 0.80, "trash": 0.10,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("organic", 0.80, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("watermelon_rind.jpg", _create_image_bytes(), "image/jpeg")})
            assert res.status_code == 200
            data = res.json()
            assert data["predicted_category"] == "organic"
            assert data["confidence"] >= 0.70
            assert data["recycling_decision"]["disposal_status"] == DisposalStatus.COMPOSTABLE
    finally:
        app.dependency_overrides.clear()


# ── 4. Vegetable Scraps -> Organic ─────────────────────────────────────────────
def test_vegetable_scraps_predicts_organic():
    vit_p = {"cardboard": 0.04, "glass": 0.01, "metal": 0.05, "paper": 0.10, "plastic": 0.10, "trash": 0.70}
    clip_p = {
        "cardboard": 0.01, "glass": 0.01, "metal": 0.01, "paper": 0.02,
        "plastic": 0.03, "organic": 0.86, "trash": 0.06,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("organic", 0.86, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("carrot_scraps.jpg", _create_image_bytes(), "image/jpeg")})
            assert res.status_code == 200
            data = res.json()
            assert data["predicted_category"] == "organic"
            assert "preparation_steps" in data["recycling_decision"]
            assert len(data["recycling_decision"]["preparation_steps"]) > 0
    finally:
        app.dependency_overrides.clear()


# ── 5. Clean Cardboard -> Cardboard ────────────────────────────────────────────
def test_clean_cardboard_predicts_cardboard():
    vit_p = {"cardboard": 0.90, "glass": 0.02, "metal": 0.02, "paper": 0.04, "plastic": 0.01, "trash": 0.01}
    clip_p = {
        "cardboard": 0.85, "glass": 0.02, "metal": 0.02, "paper": 0.05,
        "plastic": 0.02, "organic": 0.02, "trash": 0.02,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("cardboard", 0.85, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("box.jpg", _create_image_bytes(), "image/jpeg")})
            assert res.status_code == 200
            data = res.json()
            assert data["predicted_category"] == "cardboard"
            assert data["model_agreement"] is True
            assert data["model_coverage"]["coverage_type"] == "full"
            assert data["model_coverage"]["vit_supported"] is True
    finally:
        app.dependency_overrides.clear()


# ── 6. Plastic Bottle -> Plastic ───────────────────────────────────────────────
def test_plastic_bottle_predicts_plastic():
    vit_p = {"cardboard": 0.02, "glass": 0.05, "metal": 0.03, "paper": 0.02, "plastic": 0.86, "trash": 0.02}
    clip_p = {
        "cardboard": 0.01, "glass": 0.08, "metal": 0.02, "paper": 0.02,
        "plastic": 0.82, "organic": 0.02, "trash": 0.03,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("plastic", 0.82, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("bottle.jpg", _create_image_bytes(), "image/jpeg")})
            assert res.status_code == 200
            data = res.json()
            assert data["predicted_category"] == "plastic"
            assert data["model_agreement"] is True
            assert data["recycling_decision"]["disposal_status"] == DisposalStatus.CONDITIONALLY_RECYCLABLE
    finally:
        app.dependency_overrides.clear()


# ── 7. Aluminum Can -> Metal ───────────────────────────────────────────────────
def test_aluminum_can_predicts_metal():
    vit_p = {"cardboard": 0.01, "glass": 0.02, "metal": 0.91, "paper": 0.01, "plastic": 0.03, "trash": 0.02}
    clip_p = {
        "cardboard": 0.01, "glass": 0.02, "metal": 0.89, "paper": 0.01,
        "plastic": 0.04, "organic": 0.01, "trash": 0.02,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("metal", 0.89, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("can.jpg", _create_image_bytes(), "image/jpeg")})
            assert res.status_code == 200
            data = res.json()
            assert data["predicted_category"] == "metal"
            assert data["recycling_decision"]["disposal_status"] == DisposalStatus.RECYCLABLE
    finally:
        app.dependency_overrides.clear()


# ── 8. Glass Bottle -> Glass ───────────────────────────────────────────────────
def test_glass_bottle_predicts_glass():
    vit_p = {"cardboard": 0.02, "glass": 0.88, "metal": 0.02, "paper": 0.02, "plastic": 0.04, "trash": 0.02}
    clip_p = {
        "cardboard": 0.01, "glass": 0.85, "metal": 0.03, "paper": 0.01,
        "plastic": 0.06, "organic": 0.01, "trash": 0.03,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("glass", 0.85, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("glass.jpg", _create_image_bytes(), "image/jpeg")})
            assert res.status_code == 200
            data = res.json()
            assert data["predicted_category"] == "glass"
            assert data["model_agreement"] is True
            assert data["recycling_decision"]["disposal_status"] == DisposalStatus.CONDITIONALLY_RECYCLABLE
    finally:
        app.dependency_overrides.clear()


# ── 9. Contaminated Packaging -> Uncertain / Verification ──────────────────────
def test_contaminated_packaging_predicts_disagreement():
    vit_p = {"cardboard": 0.45, "glass": 0.05, "metal": 0.05, "paper": 0.10, "plastic": 0.10, "trash": 0.25}
    clip_p = {
        "cardboard": 0.10, "glass": 0.05, "metal": 0.05, "paper": 0.10,
        "plastic": 0.45, "organic": 0.05, "trash": 0.20,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("plastic", 0.45, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("contaminated.jpg", _create_image_bytes(), "image/jpeg")})
            assert res.status_code == 200
            data = res.json()
            assert data["model_agreement"] is False
            assert data["confidence"] <= 0.68
            assert data["recycling_decision"]["disposal_status"] == DisposalStatus.UNKNOWN_REQUIRES_VERIFICATION
    finally:
        app.dependency_overrides.clear()


# ── 10. Ordinary Non-Organic Trash -> Trash ────────────────────────────────────
def test_ordinary_trash_predicts_trash():
    vit_p = {"cardboard": 0.05, "glass": 0.05, "metal": 0.05, "paper": 0.05, "plastic": 0.05, "trash": 0.75}
    clip_p = {
        "cardboard": 0.05, "glass": 0.05, "metal": 0.05, "paper": 0.05,
        "plastic": 0.05, "organic": 0.05, "trash": 0.70,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("trash", 0.70, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("trash.jpg", _create_image_bytes(), "image/jpeg")})
            assert res.status_code == 200
            data = res.json()
            assert data["predicted_category"] == "trash"
            assert data["model_agreement"] is True
            assert data["recycling_decision"]["disposal_status"] == DisposalStatus.NOT_STANDARD_RECYCLABLE
    finally:
        app.dependency_overrides.clear()


# ── 11. V1 6-class Compatibility ──────────────────────────────────────────────
def test_v1_six_class_compatibility():
    vit_p = {"cardboard": 0.05, "glass": 0.02, "metal": 0.03, "paper": 0.10, "plastic": 0.10, "trash": 0.70}
    clip_p = {
        "cardboard": 0.02, "glass": 0.01, "metal": 0.01, "paper": 0.03,
        "plastic": 0.03, "organic": 0.85, "trash": 0.05,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("organic", 0.85, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("food.jpg", _create_image_bytes(), "image/jpeg")})
            data = res.json()
            assert data["vit_prediction"] == "Not supported (V1 6-class)"
            assert data["model_coverage"]["vit_supported"] is False
    finally:
        app.dependency_overrides.clear()


# ── 12. CLIP V2 7-class Mapping ────────────────────────────────────────────────
def test_clip_v2_seven_class_mapping_completeness():
    vit_p = {"cardboard": 0.1, "glass": 0.1, "metal": 0.1, "paper": 0.1, "plastic": 0.1, "trash": 0.5}
    clip_p = {c: 1.0 / 7.0 for c in V2_WASTE_CLASSES}
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("cardboard", 0.20, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("test.jpg", _create_image_bytes(), "image/jpeg")})
            data = res.json()
            class_probs = data["class_probabilities"]
            for c in V2_WASTE_CLASSES:
                assert c in class_probs
                assert isinstance(class_probs[c], float)
    finally:
        app.dependency_overrides.clear()


# ── 13. Partial Model Coverage ─────────────────────────────────────────────────
def test_partial_model_coverage_reported():
    vit_p = {"cardboard": 0.05, "glass": 0.05, "metal": 0.05, "paper": 0.05, "plastic": 0.05, "trash": 0.75}
    clip_p = {
        "cardboard": 0.02, "glass": 0.02, "metal": 0.02, "paper": 0.02,
        "plastic": 0.02, "organic": 0.85, "trash": 0.05,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("organic", 0.85, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("peel.jpg", _create_image_bytes(), "image/jpeg")})
            data = res.json()
            cov = data["model_coverage"]
            assert cov["coverage_type"] == "partial"
            assert cov["vit_supported"] is False
            assert cov["clip_supported"] is True
            assert cov["ensemble_supported"] is False
            assert "6 classes" in cov["unsupported_reason"]
    finally:
        app.dependency_overrides.clear()


# ── 14. False Consensus Prevention ─────────────────────────────────────────────
def test_false_consensus_prevention_on_incomplete_coverage():
    """
    When ViT is 6-class and CLIP predicts Organic, even if ViT defaulted to Trash,
    the pipeline MUST NOT declare Model Consensus!
    """
    vit_p = {"cardboard": 0.05, "glass": 0.02, "metal": 0.03, "paper": 0.10, "plastic": 0.10, "trash": 0.70}
    clip_p = {
        "cardboard": 0.02, "glass": 0.01, "metal": 0.01, "paper": 0.03,
        "plastic": 0.03, "organic": 0.85, "trash": 0.05,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("organic", 0.85, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("food.jpg", _create_image_bytes(), "image/jpeg")})
            data = res.json()
            assert data["model_agreement"] is False
    finally:
        app.dependency_overrides.clear()


# ── 15. Low-Confidence Organic Routes to Verification ──────────────────────────
def test_low_confidence_organic_routes_to_verification():
    """When CLIP organic confidence is below 0.40 or narrow margin, route to UNKNOWN_REQUIRES_VERIFICATION."""
    vit_p = {"cardboard": 0.15, "glass": 0.15, "metal": 0.15, "paper": 0.15, "plastic": 0.15, "trash": 0.25}
    # Organic is highest at 0.35, but below 0.40 threshold and narrow margin
    clip_p = {
        "cardboard": 0.10, "glass": 0.10, "metal": 0.05, "paper": 0.10,
        "plastic": 0.10, "organic": 0.35, "trash": 0.20,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("organic", 0.35, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("vague_organic.jpg", _create_image_bytes(), "image/jpeg")})
            data = res.json()
            assert data["predicted_category"] == "unknown"
            assert data["recycling_decision"]["disposal_status"] == DisposalStatus.UNKNOWN_REQUIRES_VERIFICATION
            assert data["model_coverage"]["compatibility_decision"] == "organic_v1_weak_confidence"
    finally:
        app.dependency_overrides.clear()


# ── 16. Organic Recycling Decision is COMPOSTABLE ──────────────────────────────
def test_organic_recycling_decision_compostable():
    vit_p = {"cardboard": 0.05, "glass": 0.02, "metal": 0.03, "paper": 0.10, "plastic": 0.10, "trash": 0.70}
    clip_p = {
        "cardboard": 0.02, "glass": 0.01, "metal": 0.01, "paper": 0.03,
        "plastic": 0.03, "organic": 0.88, "trash": 0.02,
    }
    reg = _build_mock_registry(vit_p, clip_p, vit_num_classes=6)

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("organic", 0.88, clip_p)):
            with TestClient(app) as client:
                res = client.post("/api/predict", files={"file": ("fruit.jpg", _create_image_bytes(), "image/jpeg")})
            data = res.json()
            dec = data["recycling_decision"]
            assert dec["disposal_status"] == DisposalStatus.COMPOSTABLE
            assert dec["status_label"] == "Compostable"
            assert "Green" in dec["disposal_bin"]
            assert any("compost" in step.lower() for step in dec["preparation_steps"])
    finally:
        app.dependency_overrides.clear()


# ── 17. Live CLIP V2 Prompt Classifier Verification ───────────────────────────
def test_live_clip_prompt_templates_coverage():
    """Verify that all required food scrap items are covered in V2_CLIP_PROMPT_TEMPLATES."""
    from app.ml.clip_model_v2 import V2_CLIP_PROMPT_TEMPLATES

    organic_templates = " ".join(V2_CLIP_PROMPT_TEMPLATES["organic"]).lower()

    required_keywords = [
        "banana peel",
        "apple core",
        "watermelon rind",
        "citrus",
        "vegetable",
        "carrot",
        "fruit",
        "coffee grounds",
        "eggshells",
        "food scraps",
    ]

    for kw in required_keywords:
        assert kw in organic_templates, f"Missing required keyword '{kw}' in organic prompt templates"
