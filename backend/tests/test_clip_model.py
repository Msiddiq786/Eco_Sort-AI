"""
Tests for Phase 5 — CLIP Foundation Model Integration.

Verifies:
- Model configuration (prompts, classes, checkpoint identifier)
- Reusable class prompts (exact matching with specifications)
- Zero-shot classification inference logic (softmax, normalization, argmax)
- Diagnostic flags and probability shapes
- Error handling (uninitialized models, invalid inputs, offline failures)
- ModelRegistry integration and lifecycle
- API endpoints (GET /api/model-info and POST /api/predict/clip)

All tests run completely offline without downloading real weights.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.ml.clip_model import (
    CLIP_PROMPTS,
    CLIPClassificationResult,
    load_clip,
    predict_clip,
)
from app.ml.loader import ModelRegistry, model_registry


# ── Fixtures and Mocks ────────────────────────────────────────────────────────


@pytest.fixture
def dummy_rgb_image():
    """Create a simple 64x64 RGB PIL Image."""
    return Image.new("RGB", (64, 64), color=(100, 150, 200))


class DummyCLIPOutput:
    """Mock CLIPModel forward output with logits_per_image."""

    def __init__(self, logits: torch.Tensor):
        self.logits_per_image = logits


def create_mock_clip(logits: list[float] | torch.Tensor):
    """
    Create a paired (mock_model, mock_processor) that outputs designated logits.
    """
    if not isinstance(logits, torch.Tensor):
        tensor_logits = torch.tensor([logits], dtype=torch.float32)
    else:
        tensor_logits = logits

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.return_value = DummyCLIPOutput(tensor_logits)

    mock_processor = MagicMock()
    # Processor returns a dict of dummy tensors
    mock_processor.return_value = {
        "input_ids": torch.zeros((6, 10), dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
    }

    return mock_model, mock_processor


# ── 1. Configuration & Prompts Tests ──────────────────────────────────────────


def test_clip_required_classes():
    """Verify all 6 waste classes are configured in the model."""
    expected_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    assert sorted(settings.WASTE_CLASSES) == sorted(expected_classes)
    assert len(CLIP_PROMPTS) == 6
    for cls in expected_classes:
        assert cls in CLIP_PROMPTS, f"Class '{cls}' missing from CLIP_PROMPTS"


def test_clip_exact_prompt_strings():
    """Verify that class prompts match the Phase 5 specification exactly."""
    exact_expected = {
        "cardboard": "a photo of a cardboard box or cardboard waste",
        "glass": "a photo of a glass bottle or glass container",
        "metal": "a photo of a metal can or metallic object",
        "paper": "a photo of paper waste or newspaper",
        "plastic": "a photo of a plastic bottle or plastic bag",
        "trash": "a photo of general household trash or garbage",
    }

    for cls, expected_prompt in exact_expected.items():
        assert (
            CLIP_PROMPTS[cls] == expected_prompt
        ), f"Prompt for '{cls}' does not match expected exact text."


def test_clip_settings_configuration():
    """Verify settings contains valid CLIP checkpoint and flags."""
    assert settings.CLIP_CHECKPOINT == "openai/clip-vit-base-patch32"
    assert settings.CLIP_MODEL_NAME == "openai/clip-vit-base-patch32"
    assert settings.CLIP_ENABLED is True


# ── 2. Inference & Zero-Shot Classification Tests ─────────────────────────────


def test_clip_predict_output_structure(dummy_rgb_image):
    """Verify that predict_clip produces the expected dictionary/dataclass structure."""
    mock_model, mock_processor = create_mock_clip([1.0, 2.0, 3.0, 0.5, 0.2, 0.1])

    result = predict_clip(mock_model, mock_processor, dummy_rgb_image)

    # Must be dict-compatible and instance of CLIPClassificationResult
    assert isinstance(result, CLIPClassificationResult)
    assert isinstance(result, dict)
    assert "predicted_category" in result
    assert "confidence" in result
    assert "probabilities" in result

    # Check attribute access
    assert hasattr(result, "predicted_category")
    assert hasattr(result, "confidence")
    assert hasattr(result, "probabilities")

    # Check probabilities dictionary
    probs = result["probabilities"]
    assert len(probs) == 6
    for cls in settings.WASTE_CLASSES:
        assert cls in probs
        assert isinstance(probs[cls], float)


def test_clip_probability_normalization(dummy_rgb_image):
    """Probabilities across all 6 classes must sum approximately to 1.0."""
    arbitrary_logits = [-2.5, 0.3, 4.1, 1.8, -0.9, 0.0]
    mock_model, mock_processor = create_mock_clip(arbitrary_logits)

    result = predict_clip(mock_model, mock_processor, dummy_rgb_image)
    probs = result["probabilities"]

    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected ~1.0"

    for cls, p in probs.items():
        assert 0.0 <= p <= 1.0, f"Probability {p} for {cls} is outside [0, 1]"


def test_clip_argmax_prediction_consistency(dummy_rgb_image):
    """
    The predicted_category must strictly correspond to the highest probability,
    and confidence must equal that highest probability.
    """
    # Class order: cardboard, glass, metal, paper, plastic, trash
    # Test case 1: Metal (index 2) has highest logit
    metal_logits = [0.1, 0.2, 10.0, 0.4, 0.3, 0.1]
    mock_model, mock_processor = create_mock_clip(metal_logits)
    result = predict_clip(mock_model, mock_processor, dummy_rgb_image)

    assert result["predicted_category"] == "metal"
    assert result["confidence"] == result["probabilities"]["metal"]
    assert result["confidence"] == max(result["probabilities"].values())

    # Test case 2: Plastic (index 4) has highest logit
    plastic_logits = [0.1, 0.2, 0.3, 0.4, 8.5, 0.1]
    mock_model, mock_processor = create_mock_clip(plastic_logits)
    result2 = predict_clip(mock_model, mock_processor, dummy_rgb_image)

    assert result2["predicted_category"] == "plastic"
    assert result2["confidence"] == result2["probabilities"]["plastic"]
    assert result2["confidence"] == max(result2["probabilities"].values())


def test_clip_result_to_dict(dummy_rgb_image):
    """Verify result.to_dict() returns a clean dictionary."""
    mock_model, mock_processor = create_mock_clip([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = predict_clip(mock_model, mock_processor, dummy_rgb_image)
    d = result.to_dict()

    assert isinstance(d, dict)
    assert d["predicted_category"] == "trash"
    assert "probabilities" in d
    assert "confidence" in d


# ── 3. Error Handling Tests ───────────────────────────────────────────────────


def test_clip_predict_uninitialized_model(dummy_rgb_image):
    """Inference must fail clearly if model or processor is None."""
    mock_model, mock_processor = create_mock_clip([1.0] * 6)

    with pytest.raises(RuntimeError, match="not initialized"):
        predict_clip(None, mock_processor, dummy_rgb_image)

    with pytest.raises(RuntimeError, match="not initialized"):
        predict_clip(mock_model, None, dummy_rgb_image)


def test_clip_predict_none_image():
    """Inference must fail clearly if image is None."""
    mock_model, mock_processor = create_mock_clip([1.0] * 6)

    with pytest.raises(ValueError, match="Input image cannot be None"):
        predict_clip(mock_model, mock_processor, None)


def test_clip_predict_empty_prompts(dummy_rgb_image):
    """Inference must fail if candidate prompts dictionary is empty."""
    mock_model, mock_processor = create_mock_clip([1.0] * 6)

    with pytest.raises(ValueError, match="cannot be empty"):
        predict_clip(mock_model, mock_processor, dummy_rgb_image, candidate_prompts={})


# ── 4. Model Loading & Registry Integration Tests ─────────────────────────────


def test_load_clip_offline_with_mock():
    """Verify load_clip loads model and processor and sets eval mode."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_proc = MagicMock()

    with patch("transformers.CLIPModel.from_pretrained", return_value=mock_model) as mock_m_load, \
         patch("transformers.CLIPProcessor.from_pretrained", return_value=mock_proc) as mock_p_load:

        model, proc = load_clip("openai/clip-vit-base-patch32", device="cpu")

        assert model is mock_model
        assert proc is mock_proc
        mock_model.eval.assert_called_once()
        mock_m_load.assert_called_once()
        mock_p_load.assert_called_once()


def test_load_clip_failure_propagates():
    """Verify that failure in load_clip raises an exception."""
    with patch("transformers.CLIPModel.from_pretrained", side_effect=OSError("Checkpoint not found")):
        with pytest.raises(OSError, match="Checkpoint not found"):
            load_clip("nonexistent/checkpoint")


def test_model_registry_load_clip_success():
    """Verify ModelRegistry successfully tracks loaded state when load_clip succeeds."""
    registry = ModelRegistry()
    mock_m = MagicMock()
    mock_p = MagicMock()

    with patch("app.ml.clip_model.load_clip", return_value=(mock_m, mock_p)):
        registry._load_clip()

        assert registry.clip_loaded is True
        assert registry.clip_model is mock_m
        assert registry.clip_processor is mock_p


def test_model_registry_load_clip_failure_graceful():
    """Verify ModelRegistry gracefully catches loading errors and does not claim loaded."""
    registry = ModelRegistry()

    with patch("app.ml.clip_model.load_clip", side_effect=RuntimeError("Download failed")):
        registry._load_clip()

        assert registry.clip_loaded is False
        assert registry.clip_model is None
        assert registry.clip_processor is None


def test_model_registry_clip_disabled():
    """When CLIP is disabled in settings, ModelRegistry should not attempt to load it."""
    registry = ModelRegistry()

    with patch("app.core.config.settings.CLIP_ENABLED", False):
        with patch("app.ml.clip_model.load_clip") as mock_load:
            registry._load_clip()

            mock_load.assert_not_called()
            assert registry.clip_loaded is False


# ── 5. API Integration Tests ──────────────────────────────────────────────────


def test_model_info_api_exposes_clip_fields():
    """GET /api/model-info must expose CLIP checkpoint, model name, and accurate load status."""
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/api/model-info")

    assert resp.status_code == 200
    data = resp.json()

    assert "clip_checkpoint" in data
    assert data["clip_checkpoint"] == "openai/clip-vit-base-patch32"
    assert "clip_model_name" in data
    assert data["clip_model_name"] == "openai/clip-vit-base-patch32"
    assert "clip_loaded" in data
    assert isinstance(data["clip_loaded"], bool)
    assert "clip_enabled" in data
    assert data["clip_enabled"] is True

    # ViT and ensemble fields must also remain present
    assert "vit_checkpoint" in data
    assert "ensemble_vit_weight" in data
    assert "ensemble_clip_weight" in data


def test_predict_clip_endpoint_503_when_unloaded():
    """POST /api/predict/clip must return HTTP 503 if CLIP is not loaded."""
    from app.main import app
    from app.core.dependencies import get_model_registry

    unloaded_registry = ModelRegistry()
    unloaded_registry._clip_loaded = False
    unloaded_registry.clip_model = None
    unloaded_registry.clip_processor = None

    app.dependency_overrides[get_model_registry] = lambda: unloaded_registry
    try:
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(200, 200, 200)).save(buf, format="JPEG")
        buf.seek(0)

        with TestClient(app) as client:
            resp = client.post(
                "/api/predict/clip",
                files={"file": ("test.jpg", buf, "image/jpeg")},
            )

        assert resp.status_code == 503
        assert "not loaded" in resp.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_model_registry, None)


def test_predict_clip_endpoint_success():
    """POST /api/predict/clip returns 200 with 6-class zero-shot probabilities when loaded."""
    from app.main import app
    from app.core.dependencies import get_model_registry

    mock_m, mock_p = create_mock_clip([5.0, 1.0, 0.5, 0.2, 0.1, 0.0])  # Cardboard top
    loaded_registry = ModelRegistry()
    loaded_registry._clip_loaded = True
    loaded_registry.clip_model = mock_m
    loaded_registry.clip_processor = mock_p

    app.dependency_overrides[get_model_registry] = lambda: loaded_registry
    try:
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(100, 150, 200)).save(buf, format="JPEG")
        buf.seek(0)

        with TestClient(app) as client:
            resp = client.post(
                "/api/predict/clip",
                files={"file": ("test.jpg", buf, "image/jpeg")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_category"] == "cardboard"
        assert data["confidence"] > 0.5
        assert "probabilities" in data
        assert len(data["probabilities"]) == 6
        probs_sum = sum(data["probabilities"].values())
        assert abs(probs_sum - 1.0) < 1e-4
    finally:
        app.dependency_overrides.pop(get_model_registry, None)
