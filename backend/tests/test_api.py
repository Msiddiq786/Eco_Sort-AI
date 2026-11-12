"""
Backend tests — API health and ensemble logic.

Run:
    cd smart-waste-segregation/backend
    pytest tests/ -v
"""
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ── Ensemble unit tests ───────────────────────────────────────────────


def test_ensemble_fuse_basic():
    """Fuse should return the class with the highest weighted probability."""
    from app.ml.ensemble import fuse

    vit_probs = {
        "cardboard": 0.8, "glass": 0.05, "metal": 0.05,
        "paper": 0.05, "plastic": 0.03, "trash": 0.02,
    }
    clip_probs = {
        "cardboard": 0.6, "glass": 0.1, "metal": 0.1,
        "paper": 0.1, "plastic": 0.05, "trash": 0.05,
    }
    result = fuse(vit_probs, clip_probs, vit_weight=0.70, clip_weight=0.30)
    assert result.predicted_category == "cardboard"
    assert result.vit_prediction == "cardboard"
    assert result.clip_prediction == "cardboard"
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-5


def test_ensemble_low_confidence():
    from app.ml.ensemble import fuse

    # Uniform distribution → low confidence
    uniform = {cls: 1 / 6 for cls in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]}
    result = fuse(uniform, uniform)
    assert result.is_low_confidence is True


def test_ensemble_weight_normalisation():
    """Non-unit weights should be normalised automatically."""
    from app.ml.ensemble import fuse

    probs = {"cardboard": 0.5, "glass": 0.1, "metal": 0.1,
             "paper": 0.1, "plastic": 0.1, "trash": 0.1}
    result = fuse(probs, probs, vit_weight=7, clip_weight=3)
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-5


# ── API integration tests ─────────────────────────────────────────────


def test_health_endpoint():
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_model_info_endpoint():
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "waste_classes" in data
    assert len(data["waste_classes"]) == 6


def test_predict_no_model_returns_503():
    """
    Without a trained checkpoint the endpoint must return 503, never fake predictions.
    """
    from app.main import app
    from app.ml.loader import model_registry

    img_buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_buf, format="JPEG")
    img_buf.seek(0)

    with TestClient(app) as client:
        # Ensure model is marked not loaded inside client lifespan context
        model_registry._vit_loaded = False
        resp = client.post(
            "/api/predict",
            files={"file": ("test.jpg", img_buf, "image/jpeg")},
        )
    assert resp.status_code == 503
    assert "not loaded" in resp.json()["detail"].lower()
