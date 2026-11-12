"""
GET /api/metrics    — load evaluation metrics from disk.
GET /api/train-log  — load training log from disk.

Both return 404 when the respective files don't exist yet (before training/eval).
"""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

def _get_results_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / "evaluation" / "results",
        Path("backend/evaluation/results"),
        Path("evaluation/results"),
        Path(__file__).resolve().parents[3] / "backend" / "evaluation" / "results",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


@router.get("/metrics")
async def get_metrics():
    results_dir = _get_results_dir()
    path = results_dir / "metrics.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Run evaluation/evaluate_model.py first.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/train-log")
async def get_train_log():
    results_dir = _get_results_dir()
    path = results_dir / "training_history.json"
    if not path.exists():
        path = results_dir / "train_log.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Training log not found. Run training/train_vit.py first.",
        )
    return json.loads(path.read_text(encoding="utf-8"))
