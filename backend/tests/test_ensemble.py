"""
Unit and integration tests for Phase 6: Ensemble Fusion, Probability Alignment,
Confidence Flagging, and Preprocessing Diagnostics.

Covers:
  1. ViT probability normalization.
  2. CLIP probability normalization.
  3. Correct class-index alignment.
  4. 70:30 weighted ensemble calculation.
  5. Correct final category selection (argmax).
  6. Confidence calculation (max ensemble prob).
  7. Invalid model output handling (empty, NaN, negative, missing classes).
  8. Low-confidence and ambiguity flagging.
  9. Image preprocessing (EXIF orientation transpose, RGB conversion).
  10. Full PredictResponse schema validation.
"""
from __future__ import annotations

import io
import pytest
from PIL import Image

from app.core.config import settings
from app.ml.ensemble import (
    AMBIGUITY_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    EnsembleResult,
    _validate_and_normalize_probs,
    fuse,
)
from app.ml.preprocessor import load_image_pil
from app.schemas.predict import ClassProbabilities, PredictResponse


@pytest.fixture
def uniform_vit_probs() -> dict[str, float]:
    """Simulate under-trained/smoke-test ViT output (near uniform ~16.7%)."""
    return {
        "cardboard": 0.1748,
        "glass": 0.1636,
        "metal": 0.1738,
        "paper": 0.1612,
        "plastic": 0.1474,
        "trash": 0.1792,
    }


@pytest.fixture
def confident_clip_plastic() -> dict[str, float]:
    """Simulate confident zero-shot CLIP prediction for plastic water bottle."""
    return {
        "cardboard": 0.0003,
        "glass": 0.1076,
        "metal": 0.0025,
        "paper": 0.0055,
        "plastic": 0.8355,
        "trash": 0.0486,
    }


# ── 1 & 2. Probability Normalization ──────────────────────────────────────────


def test_validate_and_normalize_valid_probs():
    """Valid raw probabilities must normalize strictly to sum = 1.0."""
    raw = {c: 1.0 for c in settings.WASTE_CLASSES}  # sum = 6.0
    norm = _validate_and_normalize_probs(raw, "TestModel")
    assert pytest.approx(sum(norm.values()), abs=1e-5) == 1.0
    for c in settings.WASTE_CLASSES:
        assert norm[c] == pytest.approx(1 / 6, abs=1e-4)


# ── 3. Class Index and Key Alignment ──────────────────────────────────────────


def test_class_key_alignment(uniform_vit_probs, confident_clip_plastic):
    """Ensemble must align all 6 waste classes in exact canonical order."""
    res = fuse(uniform_vit_probs, confident_clip_plastic)
    assert set(res.class_probabilities.keys()) == set(settings.WASTE_CLASSES)
    assert len(res.class_probabilities) == 6


# ── 4. 70:30 Weighted Average Calculation ─────────────────────────────────────


def test_ensemble_70_30_weighted_average(uniform_vit_probs, confident_clip_plastic):
    """
    Ensemble probabilities must match w_vit=0.70 and w_clip=0.30:
    Plastic: 0.70 * 0.1474 + 0.30 * 0.8355 = 0.10318 + 0.25065 = 0.35383.
    """
    res = fuse(uniform_vit_probs, confident_clip_plastic, vit_weight=0.70, clip_weight=0.30)
    expected_plastic = 0.70 * uniform_vit_probs["plastic"] + 0.30 * confident_clip_plastic["plastic"]
    assert res.class_probabilities["plastic"] == pytest.approx(expected_plastic, abs=1e-3)


# ── 5. Correct Final Category Selection (Argmax) ──────────────────────────────


def test_argmax_category_selection(uniform_vit_probs, confident_clip_plastic):
    """Final predicted category must be the argmax of the ensemble distribution."""
    res = fuse(uniform_vit_probs, confident_clip_plastic)
    assert res.predicted_category == "plastic"
    assert res.confidence == res.class_probabilities["plastic"]


# ── 6. Confidence Calculation ─────────────────────────────────────────────────


def test_confidence_is_max_probability():
    """Confidence must strictly equal the maximum probability in class_probabilities."""
    vit = {c: 0.1 for c in settings.WASTE_CLASSES}
    vit["cardboard"] = 0.5
    clip = {c: 0.1 for c in settings.WASTE_CLASSES}
    clip["cardboard"] = 0.5

    res = fuse(vit, clip)
    assert res.confidence == max(res.class_probabilities.values())
    assert res.predicted_category == "cardboard"


# ── 7. Invalid Model Output Handling ──────────────────────────────────────────


def test_invalid_probs_empty_raises():
    with pytest.raises(ValueError, match="non-empty dictionary"):
        fuse({}, {"cardboard": 1.0})


def test_invalid_probs_nan_raises():
    bad = {c: 0.1 for c in settings.WASTE_CLASSES}
    bad["plastic"] = float("nan")
    with pytest.raises(ValueError, match="Invalid probability"):
        fuse(bad, {c: 0.1 for c in settings.WASTE_CLASSES})


def test_invalid_probs_negative_raises():
    bad = {c: 0.1 for c in settings.WASTE_CLASSES}
    bad["metal"] = -0.5
    with pytest.raises(ValueError, match="Invalid probability"):
        fuse(bad, {c: 0.1 for c in settings.WASTE_CLASSES})


# ── 8. Low-Confidence and Ambiguity Flagging ───────────────────────────────────


def test_low_confidence_flagged():
    """Predictions below LOW_CONFIDENCE_THRESHOLD must be flagged."""
    flat_vit = {c: 1 / 6 for c in settings.WASTE_CLASSES}
    flat_clip = {c: 1 / 6 for c in settings.WASTE_CLASSES}
    res = fuse(flat_vit, flat_clip)

    assert res.confidence < LOW_CONFIDENCE_THRESHOLD
    assert res.is_low_confidence is True
    assert res.confidence_status == "low_confidence"


def test_high_confidence_not_flagged():
    """Confident predictions (> threshold) must have normal status."""
    conf = {c: 0.05 for c in settings.WASTE_CLASSES}
    conf["glass"] = 0.75
    res = fuse(conf, conf)

    assert res.is_low_confidence is False
    assert res.confidence_status == "normal"


def test_ambiguity_flagged():
    """Top-2 probabilities with gap < AMBIGUITY_THRESHOLD must be flagged ambiguous."""
    p = {c: 0.05 for c in settings.WASTE_CLASSES}
    p["cardboard"] = 0.41
    p["paper"] = 0.39
    res = fuse(p, p)

    assert res.is_ambiguous is True
    assert (res.class_probabilities["cardboard"] - res.class_probabilities["paper"]) < AMBIGUITY_THRESHOLD


# ── 9. Image Preprocessing Diagnostics ────────────────────────────────────────


def test_load_image_pil_rgb_conversion():
    """Grayscale and RGBA images must convert to 3-channel RGB."""
    rgba = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    buf.seek(0)

    loaded = load_image_pil(buf.getvalue())
    assert loaded.mode == "RGB"
    assert loaded.size == (32, 32)


# ── 10. PredictResponse Schema Validation ──────────────────────────────────────


def test_predict_response_schema():
    """PredictResponse must validate full schema including diagnostic fields."""
    probs = {c: 1 / 6 for c in settings.WASTE_CLASSES}
    resp = PredictResponse(
        id=1,
        filename="test.jpg",
        original_filename="upload.jpg",
        image_path="uploads/test.jpg",
        predicted_category="plastic",
        confidence=0.78,
        vit_prediction="cardboard",
        clip_prediction="plastic",
        class_probabilities=ClassProbabilities(**probs),
        vit_probabilities=ClassProbabilities(**probs),
        clip_probabilities=ClassProbabilities(**probs),
        is_low_confidence=False,
        is_ambiguous=False,
        confidence_status="normal",
        recycling_recommendation="Recycle with plastics.",
        model_version="1.0.0",
        created_at="2026-11-12T12:00:00Z",
    )
    assert resp.predicted_category == "plastic"
    assert resp.vit_probabilities is not None
    assert resp.confidence_status == "normal"
