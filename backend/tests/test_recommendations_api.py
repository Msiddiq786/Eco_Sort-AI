"""
Tests for Smart Bin Recommendation API (/api/recommendations) and recycling_engine.get_bin_recommendation.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


REQUIRED_FIELDS = {
    "category",
    "bin_color",
    "bin_label",
    "stream",
    "disposal_status",
    "status_label",
    "action",
    "preparation_steps",
    "eco_tip",
    "regional_caveats",
}


def test_get_recommendation_cardboard():
    with TestClient(app) as client:
        res = client.get("/api/recommendations/cardboard")
    assert res.status_code == 200
    data = res.json()
    assert data["category"] == "cardboard"
    assert "Blue Bin" in data["bin_label"]
    assert data["stream"] == "Dry Recyclable"
    assert len(data["preparation_steps"]) >= 1


def test_get_recommendation_organic():
    with TestClient(app) as client:
        res = client.get("/api/recommendations/organic")
    assert res.status_code == 200
    data = res.json()
    assert data["category"] == "organic"
    assert data["disposal_status"] == "COMPOSTABLE"
    assert "Green Bin" in data["bin_label"]
    assert data["stream"] == "Wet / Compostable"


def test_get_recommendation_unknown_category():
    with TestClient(app) as client:
        res_unknown = client.get("/api/recommendations/unknown")
        res_missing = client.get("/api/recommendations/unrecognized_xyz")
    assert res_unknown.status_code == 200
    assert res_unknown.json()["disposal_status"] == "UNKNOWN_REQUIRES_VERIFICATION"
    assert res_missing.status_code == 404


def test_recommendation_structure_has_required_fields():
    with TestClient(app) as client:
        res = client.get("/api/recommendations")
    assert res.status_code == 200
    data = res.json()
    categories = data["categories"]
    assert len(categories) >= 7
    found_names = {c["category"] for c in categories}
    for expected in ("cardboard", "glass", "metal", "paper", "plastic", "organic", "trash"):
        assert expected in found_names

    for item in categories:
        assert REQUIRED_FIELDS.issubset(item.keys())
        assert isinstance(item["preparation_steps"], list)
        assert item["bin_color"].startswith("#")
