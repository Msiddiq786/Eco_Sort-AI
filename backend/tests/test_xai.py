"""
Tests for Explainable AI (XAI) module and /api/explain & /api/predict?include_explanation=true.
"""
from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.core.dependencies import get_model_registry
from app.main import app
from app.ml.clip_model_v2 import CLIPClassificationResultV2
from app.ml.loader import ModelRegistry
from app.ml.xai import (
    _attention_rollout,
    build_xai_explanation,
    explain_clip_scores,
    generate_attention_heatmap,
)


def _create_image_bytes(size: tuple[int, int] = (224, 224)) -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(60, 140, 90)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _build_mock_registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg._vit_loaded = True
    reg._clip_loaded = True
    vit_model = MagicMock()
    vit_model.config = MagicMock()
    vit_model.config.num_labels = 6
    reg.vit_model = vit_model
    reg.vit_processor = MagicMock()
    reg.clip_model = MagicMock()
    reg.clip_processor = MagicMock()
    return reg


def test_attention_rollout_shape_valid():
    """Attention rollout over 4 layers of (12 heads, 197, 197) produces 196 normalized patch scores."""
    rng = np.random.default_rng(42)
    attentions = [rng.dirichlet(np.ones(197), size=(12, 197)).astype(np.float32) for _ in range(4)]

    rollout = _attention_rollout(attentions, head_fusion="mean", discard_ratio=0.9)

    assert rollout.shape == (196,)
    assert rollout.dtype == np.float32
    assert float(rollout.min()) >= 0.0
    assert float(rollout.max()) <= 1.0 + 1e-6


def test_generate_attention_heatmap_returns_base64():
    """generate_attention_heatmap returns a valid base64 PNG and 2D rollout array."""
    rng = np.random.default_rng(7)
    attentions = [rng.dirichlet(np.ones(197), size=(12, 197)).astype(np.float32) for _ in range(2)]
    pil_img = Image.new("RGB", (224, 224), color=(120, 180, 90))

    with patch("app.ml.xai._extract_attention_maps", return_value=(attentions, 14, 14)):
        b64_str, rollout_2d = generate_attention_heatmap(
            model=MagicMock(),
            processor=MagicMock(),
            pil_image=pil_img,
        )

    assert isinstance(b64_str, str) and len(b64_str) > 0
    assert rollout_2d.shape == (14, 14)

    decoded = base64.b64decode(b64_str)
    img_out = Image.open(io.BytesIO(decoded))
    assert img_out.format == "PNG"
    assert img_out.size == (224, 224)


def test_xai_heatmap_failure_does_not_crash_prediction():
    """If attention extraction fails during /api/predict?include_explanation=true, prediction still succeeds."""
    vit_p = {"cardboard": 0.80, "glass": 0.04, "metal": 0.04, "paper": 0.06, "plastic": 0.04, "trash": 0.02}
    clip_p = {
        "cardboard": 0.82, "glass": 0.03, "metal": 0.03, "paper": 0.05,
        "plastic": 0.04, "organic": 0.01, "trash": 0.02,
    }
    reg = _build_mock_registry()

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.predict.predict_vit", return_value=vit_p), \
             patch("app.api.predict.predict_clip_v2", return_value=CLIPClassificationResultV2("cardboard", 0.82, clip_p)), \
             patch("app.ml.xai.generate_attention_heatmap", side_effect=RuntimeError("attention unsupported")):
            with TestClient(app) as client:
                res = client.post(
                    "/api/predict?include_explanation=true",
                    files={"file": ("box.jpg", _create_image_bytes(), "image/jpeg")},
                )
        assert res.status_code == 200
        data = res.json()
        assert data["predicted_category"] == "cardboard"
        assert data["xai_explanation"] is not None
        assert data["xai_explanation"]["attention_heatmap_base64"] is None
        assert data["xai_explanation"]["models_agree"] is True
    finally:
        app.dependency_overrides.clear()


def test_explain_clip_scores_ranked_correctly():
    """explain_clip_scores returns categories sorted descending with 1-based ranks."""
    scores = {
        "cardboard": 0.05,
        "glass": 0.62,
        "metal": 0.18,
        "paper": 0.03,
        "plastic": 0.10,
        "organic": 0.01,
        "trash": 0.01,
    }
    ranked = explain_clip_scores(scores, top_k=7)
    assert len(ranked) == 7
    assert ranked[0]["category"] == "glass"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["percentage"] == 62.0
    assert ranked[1]["category"] == "metal"
    assert ranked[1]["rank"] == 2
    assert [r["score"] for r in ranked] == sorted((r["score"] for r in ranked), reverse=True)


def test_build_xai_explanation_structure():
    """build_xai_explanation populates all required fields for agreement and disagreement cases."""
    vit_p = {"cardboard": 0.10, "glass": 0.70, "metal": 0.10, "paper": 0.04, "plastic": 0.04, "trash": 0.02}
    clip_p = {"cardboard": 0.05, "glass": 0.15, "metal": 0.65, "paper": 0.05, "plastic": 0.05, "organic": 0.02, "trash": 0.03}
    ens_p = {"cardboard": 0.085, "glass": 0.535, "metal": 0.265, "paper": 0.043, "plastic": 0.043, "organic": 0.006, "trash": 0.023}

    payload = build_xai_explanation(
        predicted_category="glass",
        confidence=0.535,
        vit_probs=vit_p,
        clip_probs=clip_p,
        ensemble_probs=ens_p,
        attention_heatmap_b64="ZmFrZV9wbmc=",
        vit_weight=0.70,
        clip_weight=0.30,
    )

    assert payload["predicted_category"] == "glass"
    assert payload["models_agree"] is False
    assert payload["attention_heatmap_base64"] == "ZmFrZV9wbmc="
    assert payload["attention_method"] == "attention_rollout"
    assert payload["vit_explanation"]["predicted"] == "glass"
    assert len(payload["vit_explanation"]["top3"]) == 3
    assert payload["clip_explanation"]["predicted"] == "metal"
    assert len(payload["clip_explanation"]["ranked_categories"]) == 7
    assert "0.7" in payload["ensemble_explanation"]["formula"]
    assert len(payload["ensemble_explanation"]["top3"]) == 3


def test_post_explain_endpoint():
    """POST /api/explain returns structured XAI payload without storing to DB."""
    vit_p = {"cardboard": 0.04, "glass": 0.04, "metal": 0.82, "paper": 0.04, "plastic": 0.04, "trash": 0.02}
    clip_p = {
        "cardboard": 0.02, "glass": 0.03, "metal": 0.80, "paper": 0.04,
        "plastic": 0.06, "organic": 0.02, "trash": 0.03,
    }
    reg = _build_mock_registry()

    app.dependency_overrides[get_model_registry] = lambda: reg
    try:
        with patch("app.api.explain.predict_vit", return_value=vit_p), \
             patch("app.api.explain.predict_clip_v2", return_value=CLIPClassificationResultV2("metal", 0.80, clip_p)), \
             patch("app.api.explain.generate_attention_heatmap", return_value=("YWJj", np.zeros((14, 14)))):
            with TestClient(app) as client:
                res = client.post(
                    "/api/explain?include_heatmap=true",
                    files={"file": ("can.jpg", _create_image_bytes(), "image/jpeg")},
                )
        assert res.status_code == 200
        data = res.json()
        assert data["predicted_category"] == "metal"
        assert data["models_agree"] is True
        assert data["attention_heatmap_base64"] == "YWJj"
    finally:
        app.dependency_overrides.clear()
