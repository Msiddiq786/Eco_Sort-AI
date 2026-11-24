"""
Ensemble configuration and optimization API endpoints — EcoSort AI V3.

GET  /api/ensemble/config      — current weights + source
GET  /api/ensemble/evaluation  — last grid-search results (if available)
POST /api/ensemble/optimize    — trigger validation-only grid search, save best weights
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.dependencies import get_model_registry
from app.services.ensemble_optimizer import (
    load_ensemble_config,
    load_saved_grid_evaluation,
    run_grid_search,
    save_ensemble_config,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store for the last grid-search result (survives within one server run)
_last_grid_result: Optional[Dict[str, Any]] = None
_optimization_running: bool = False


# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------

class EnsembleConfigResponse(BaseModel):
    vit_weight: float = Field(..., ge=0.0, le=1.0)
    clip_weight: float = Field(..., ge=0.0, le=1.0)
    macro_f1: Optional[float] = None
    accuracy: Optional[float] = None
    source: str  # "default" | "validated"
    optimized_at: Optional[str] = None


class GridResultEntry(BaseModel):
    vit_weight: float
    clip_weight: float
    macro_f1: float
    accuracy: float


class EnsembleEvaluationResponse(BaseModel):
    available: bool
    best_vit_weight: Optional[float] = None
    best_clip_weight: Optional[float] = None
    best_macro_f1: Optional[float] = None
    best_accuracy: Optional[float] = None
    baseline_vit_weight: Optional[float] = None
    baseline_clip_weight: Optional[float] = None
    baseline_macro_f1: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    num_samples: Optional[int] = None
    all_results: Optional[List[GridResultEntry]] = None


class OptimizeResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# GET /api/ensemble/config
# ---------------------------------------------------------------------------

@router.get(
    "/ensemble/config",
    response_model=EnsembleConfigResponse,
    summary="Get current ensemble weight configuration",
)
async def get_ensemble_config():
    """
    Returns the current ViT/CLIP ensemble weights and their source
    (default 70/30 or last validated configuration).
    """
    cfg = load_ensemble_config()
    return EnsembleConfigResponse(
        vit_weight=cfg.get("vit_weight", 0.70),
        clip_weight=cfg.get("clip_weight", 0.30),
        macro_f1=cfg.get("macro_f1"),
        accuracy=cfg.get("accuracy"),
        source=cfg.get("source", "default"),
        optimized_at=cfg.get("optimized_at"),
    )


# ---------------------------------------------------------------------------
# GET /api/ensemble/evaluation
# ---------------------------------------------------------------------------

@router.get(
    "/ensemble/evaluation",
    response_model=EnsembleEvaluationResponse,
    summary="Get last grid-search evaluation results",
)
async def get_ensemble_evaluation():
    """
    Returns the results of the most recently completed grid search,
    falling back to saved offline validation analysis if present.
    """
    global _last_grid_result
    r = _last_grid_result
    if r is None:
        r = load_saved_grid_evaluation()
    if r is None or r.get("error") or not r.get("all_results"):
        return EnsembleEvaluationResponse(available=False)

    return EnsembleEvaluationResponse(
        available=True,
        best_vit_weight=r.get("best_vit_weight"),
        best_clip_weight=r.get("best_clip_weight"),
        best_macro_f1=r.get("best_macro_f1"),
        best_accuracy=r.get("best_accuracy"),
        baseline_vit_weight=r.get("baseline_vit_weight"),
        baseline_clip_weight=r.get("baseline_clip_weight"),
        baseline_macro_f1=r.get("baseline_macro_f1"),
        baseline_accuracy=r.get("baseline_accuracy"),
        num_samples=r.get("num_samples"),
        all_results=[GridResultEntry(**e) for e in (r.get("all_results") or [])],
    )


# ---------------------------------------------------------------------------
# POST /api/ensemble/optimize
# ---------------------------------------------------------------------------

@router.post(
    "/ensemble/optimize",
    response_model=OptimizeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger validation-only ensemble weight grid search",
)
async def optimize_ensemble(
    background_tasks: BackgroundTasks,
    registry=Depends(get_model_registry),
):
    """
    Launches an async grid search over ViT/CLIP weight combinations using
    the validation dataset at backend/data/val/.

    The search runs in a background task. Poll GET /api/ensemble/evaluation
    to retrieve results after it completes.

    Validation data must exist at backend/data/val/<class>/<images>.
    """
    global _optimization_running

    if not registry.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Cannot run grid search.",
        )

    if _optimization_running:
        return OptimizeResponse(
            status="already_running",
            message="An optimization is already in progress. Poll GET /api/ensemble/evaluation for results.",
        )

    background_tasks.add_task(_run_optimization_task, registry)
    return OptimizeResponse(
        status="started",
        message=(
            "Grid search started in the background. "
            "Poll GET /api/ensemble/evaluation to retrieve results when complete. "
            "Requires validation images at backend/data/val/<class_name>/."
        ),
    )


async def _run_optimization_task(registry) -> None:
    global _last_grid_result, _optimization_running
    _optimization_running = True
    try:
        result = await run_grid_search(registry)
        _last_grid_result = result

        # Persist best weights
        cfg = {
            "vit_weight": result["best_vit_weight"],
            "clip_weight": result["best_clip_weight"],
            "macro_f1": result["best_macro_f1"],
            "accuracy": result["best_accuracy"],
            "source": "validated",
            "optimized_at": datetime.now(timezone.utc).isoformat(),
        }
        save_ensemble_config(cfg)
        logger.info(
            "Optimization complete: best ViT=%.2f CLIP=%.2f F1=%.4f",
            cfg["vit_weight"], cfg["clip_weight"], cfg["macro_f1"],
        )
    except FileNotFoundError as exc:
        logger.error("Optimization failed — validation data missing: %s", exc)
        _last_grid_result = {
            "error": str(exc),
            "available": False,
        }
    except Exception as exc:
        logger.exception("Optimization task failed: %s", exc)
        _last_grid_result = {
            "error": str(exc),
            "available": False,
        }
    finally:
        _optimization_running = False
