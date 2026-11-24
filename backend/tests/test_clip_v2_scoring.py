"""
Tests for Improved CLIP V2 Class-Centroid Scoring & Taxonomy Disambiguation.

Verifies:
1. Exactly seven category scores are produced.
2. Probabilities sum to 1.0 (within float precision).
3. Prompt aggregation and centroid caching are deterministic.
4. Prompt count does not unintentionally bias a category.
5. Identical image inputs produce consistent, reproducible outputs.
6. Cardboard vs Paper disambiguation on real test image.
7. Organic classification performance.
8. Recycling recommendation engine integration.
"""
from __future__ import annotations

import io
from pathlib import Path
import pytest
from PIL import Image
import torch

from app.ml.clip_model import load_clip
from app.ml.clip_model_v2 import (
    V2_WASTE_CLASSES,
    V2_CLIP_PROMPT_TEMPLATES,
    get_or_build_class_centroids,
    clear_clip_v2_cache,
    predict_clip_v2,
    CLIPClassificationResultV2,
)
from app.services.recycling_engine import get_recycling_decision, DisposalStatus

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def clip_components():
    model, processor = load_clip()
    return model, processor


def test_seven_category_scores_produced(clip_components):
    """Verify that exactly 7 category probabilities are produced matching V2_WASTE_CLASSES."""
    model, processor = clip_components
    dummy_img = Image.new("RGB", (224, 224), color=(200, 180, 150))

    result = predict_clip_v2(model, processor, dummy_img)
    assert isinstance(result, CLIPClassificationResultV2)
    assert len(result.probabilities) == 7
    for cls in V2_WASTE_CLASSES:
        assert cls in result.probabilities
        assert 0.0 <= result.probabilities[cls] <= 1.0
    assert result.predicted_category in V2_WASTE_CLASSES


def test_probabilities_sum_to_one(clip_components):
    """Verify that class probabilities sum to approximately 1.0."""
    model, processor = clip_components
    dummy_img = Image.new("RGB", (224, 224), color=(100, 150, 200))

    result = predict_clip_v2(model, processor, dummy_img)
    prob_sum = sum(result.probabilities.values())
    assert abs(prob_sum - 1.0) < 1e-4
    assert abs(result.confidence - result.probabilities[result.predicted_category]) < 1e-6


def test_prompt_aggregation_is_deterministic(clip_components):
    """Verify that centroid computation and prediction are strictly deterministic across runs."""
    model, processor = clip_components
    dummy_img = Image.new("RGB", (224, 224), color=(120, 120, 120))

    clear_clip_v2_cache()
    c1 = get_or_build_class_centroids(model, processor)
    c2 = get_or_build_class_centroids(model, processor)
    assert torch.allclose(c1, c2, atol=1e-6), "Centroids must be identical across calls"

    r1 = predict_clip_v2(model, processor, dummy_img)
    r2 = predict_clip_v2(model, processor, dummy_img)

    assert r1.predicted_category == r2.predicted_category
    assert abs(r1.confidence - r2.confidence) < 1e-6
    for cls in V2_WASTE_CLASSES:
        assert abs(r1.probabilities[cls] - r2.probabilities[cls]) < 1e-6


def test_prompt_count_does_not_bias_category(clip_components):
    """
    Verify that class centroid scoring eliminates prompt-count bias.
    Each class centroid is unit-normalized (L2 norm = 1.0), regardless of prompt count.
    """
    model, processor = clip_components
    centroids = get_or_build_class_centroids(model, processor)

    assert centroids.shape == (7, 512), f"Expected shape (7, 512), got {centroids.shape}"
    norms = centroids.norm(dim=-1)
    for idx, norm_val in enumerate(norms):
        assert abs(norm_val.item() - 1.0) < 1e-5, (
            f"Class {V2_WASTE_CLASSES[idx]} centroid is not unit normalized: {norm_val.item()}"
        )


def test_identical_image_consistency(clip_components):
    """Verify identical image bytes loaded into separate PIL instances produce consistent scores."""
    model, processor = clip_components
    img_data = io.BytesIO()
    original = Image.new("RGB", (224, 224), color=(180, 120, 60))
    original.save(img_data, format="JPEG")

    img_data.seek(0)
    img_a = Image.open(img_data)
    img_data.seek(0)
    img_b = Image.open(img_data)

    res_a = predict_clip_v2(model, processor, img_a)
    res_b = predict_clip_v2(model, processor, img_b)

    assert res_a.predicted_category == res_b.predicted_category
    assert abs(res_a.confidence - res_b.confidence) < 1e-6


def test_cardboard_disambiguation_on_test_image(clip_components):
    """
    Verify the fix for the cardboard image:
    Cardboard must be predicted with high probability, NOT confused with paper.
    """
    model, processor = clip_components
    img_path = ROOT_DIR / "uploads" / "af1bfb110b5b49a29b10dd40c4d22035_61h-5NsO8KL._AC_UF1000_1000_QL80_.jpg"
    if not img_path.exists():
        pytest.skip(f"Test image not found at {img_path}")

    img = Image.open(img_path)
    result = predict_clip_v2(model, processor, img)

    assert result.predicted_category == "cardboard", (
        f"Expected 'cardboard', got '{result.predicted_category}' (Paper={result.probabilities.get('paper')})"
    )
    assert result.probabilities["cardboard"] > 0.80, (
        f"Cardboard probability should be high (>80%), got {result.probabilities['cardboard']}"
    )
    assert result.probabilities["paper"] < 0.05, (
        f"Paper probability should be low (<5%), got {result.probabilities['paper']}"
    )


def test_organic_image_classification(clip_components):
    """Verify organic waste images are correctly detected as organic."""
    model, processor = clip_components
    org_folder = ROOT_DIR / "data" / "val" / "organic"
    if not org_folder.is_dir():
        pytest.skip(f"Organic validation folder not found at {org_folder}")

    org_files = sorted(org_folder.glob("*.jpg"))
    if not org_files:
        pytest.skip("No organic test images found")

    correct = 0
    test_subset = org_files[:5]
    for p in test_subset:
        img = Image.open(p)
        res = predict_clip_v2(model, processor, img)
        if res.predicted_category == "organic":
            correct += 1

    assert correct >= 4, f"Expected at least 4/5 organic images correctly predicted, got {correct}/5"


def test_recycling_recommendations_integration(clip_components):
    """Verify recycling decision engine handles both agreement and disagreement correctly."""
    decision_agree = get_recycling_decision(
        category="cardboard",
        confidence=0.96,
        is_ood=False,
        confidence_status="normal",
        model_agreement=True,
    )
    assert decision_agree.disposal_status == DisposalStatus.CONDITIONALLY_RECYCLABLE
    assert decision_agree.status_label == "Conditionally Recyclable"
    assert "Blue" in decision_agree.disposal_bin or "recycl" in decision_agree.disposal_bin.lower()

    decision_disagree = get_recycling_decision(
        category="cardboard",
        confidence=0.68,
        is_ood=False,
        confidence_status="ambiguous",
        model_agreement=False,
    )
    assert decision_disagree.disposal_status == DisposalStatus.UNKNOWN_REQUIRES_VERIFICATION
    assert decision_disagree.status_label == "Inspection Required"
