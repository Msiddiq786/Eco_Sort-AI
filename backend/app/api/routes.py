"""
API route registration.
"""
from fastapi import APIRouter

from app.api import detect, health, history, metrics, predict, stats

router = APIRouter()

router.include_router(health.router, tags=["Health"])
router.include_router(stats.router, tags=["Stats"])
router.include_router(predict.router, tags=["Predict"])
router.include_router(detect.router, tags=["Detect"])
router.include_router(history.router, tags=["History"])
router.include_router(metrics.router, tags=["Metrics"])
