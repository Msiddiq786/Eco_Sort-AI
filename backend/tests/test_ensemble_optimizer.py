"""
Tests for Adaptive Ensemble Optimizer service and /api/ensemble endpoints.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.ml.clip_model_v2 import CLIPClassificationResultV2
from app.ml.loader import ModelRegistry
from app.services.ensemble_optimizer import (
    DEFAULT_CLIP_WEIGHT,
    DEFAULT_VIT_WEIGHT,
    WEIGHT_GRID,
    _load_validation_data,
    load_ensemble_config,
    run_grid_search,
    save_ensemble_config,
)


def _make_mock_registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg._vit_loaded = True
    reg._clip_loaded = True
    reg.vit_model = MagicMock()
    reg.vit_processor = MagicMock()
    reg.clip_model = MagicMock()
    reg.clip_processor = MagicMock()
    return reg


def test_weight_grid_covers_full_range():
    """Weight grid covers 0.00 to 1.00 in 0.10 steps (11 weights)."""
    assert len(WEIGHT_GRID) == 11
    assert WEIGHT_GRID[0] == 0.0
    assert WEIGHT_GRID[-1] == 1.0
    assert 0.7 in WEIGHT_GRID
    assert 0.9 in WEIGHT_GRID


@pytest.mark.asyncio
async def test_best_weight_selected_by_macro_f1(tmp_path: Path):
    """
    Grid search selects the weight pair with highest validation Macro F1.
    Constructs 2 synthetic classes where ViT=0.9/CLIP=0.1 achieves 1.0 F1
    while CLIP-heavy splits drop F1.
    """
    for cls, color in [("cardboard", (180, 130, 80)), ("glass", (50, 180, 90))]:
        cls_dir = tmp_path / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=color).save(cls_dir / f"{cls}_1.jpg")

    def fake_vit(_model, _proc, img):
        r, g, _ = img.getpixel((0, 0))
        if r > g:
            return {"cardboard": 0.80, "glass": 0.20}
        return {"cardboard": 0.20, "glass": 0.80}

    def fake_clip(_model, _proc, img):
        # Strongly biased toward cardboard so only vit_weight >= 0.6 predicts glass accurately
        return CLIPClassificationResultV2("cardboard", 0.90, {"cardboard": 0.90, "glass": 0.10})

    reg = _make_mock_registry()
    with patch("app.ml.vit_model.predict_vit", side_effect=fake_vit), \
         patch("app.ml.clip_model_v2.predict_clip_v2", side_effect=fake_clip):
        result = await run_grid_search(reg, val_dir=tmp_path)

    assert result["num_samples"] == 2
    assert len(result["all_results"]) == 11
    assert result["best_macro_f1"] == 1.0
    assert result["best_vit_weight"] >= 0.6
    assert abs(result["best_vit_weight"] + result["best_clip_weight"] - 1.0) < 1e-6


def test_config_saved_and_loaded(tmp_path: Path):
    """save_ensemble_config persists JSON and load_ensemble_config reads it back."""
    cfg_file = tmp_path / "ensemble_config.json"

    # Before saving -> defaults
    default_cfg = load_ensemble_config(config_path=cfg_file)
    assert default_cfg["vit_weight"] == 0.70
    assert default_cfg["clip_weight"] == 0.30
    assert default_cfg["source"] == "default"

    saved_payload = {
        "vit_weight": 0.90,
        "clip_weight": 0.10,
        "macro_f1": 0.9117,
        "accuracy": 0.9294,
        "source": "validated",
        "optimized_at": "2026-11-24T12:00:00Z",
    }
    save_ensemble_config(saved_payload, config_path=cfg_file)

    loaded = load_ensemble_config(config_path=cfg_file)
    assert loaded["vit_weight"] == 0.90
    assert loaded["clip_weight"] == 0.10
    assert loaded["macro_f1"] == 0.9117
    assert loaded["source"] == "validated"


def test_validation_data_not_test_data(tmp_path: Path):
    """Missing validation folder raises clear FileNotFoundError; test dir is never used."""
    missing = tmp_path / "nonexistent_val"
    with pytest.raises(FileNotFoundError, match="Validation directory not found"):
        _load_validation_data(missing)

    import inspect
    src = inspect.getsource(run_grid_search)
    assert '"val"' in src
    assert '"test"' not in src


def test_baseline_70_30_preserved():
    """Default ensemble baseline constants remain 0.70 ViT / 0.30 CLIP and API endpoints respond."""
    assert DEFAULT_VIT_WEIGHT == 0.70
    assert DEFAULT_CLIP_WEIGHT == 0.30

    with TestClient(app) as client:
        cfg_res = client.get("/api/ensemble/config")
        assert cfg_res.status_code == 200
        cfg_data = cfg_res.json()
        assert pytest.approx(cfg_data["vit_weight"] + cfg_data["clip_weight"], abs=1e-4) == 1.0
        assert cfg_data["source"] in ("default", "validated")

        eval_res = client.get("/api/ensemble/evaluation")
        assert eval_res.status_code == 200
        eval_data = eval_res.json()
        assert "available" in eval_data
