"""
API route registration.
"""
from fastapi import APIRouter

from app.api import detect, ensemble, explain, feedback, health, history, metrics, predict, recommendations, stats

router = APIRouter()

router.include_router(health.router, tags=["Health"])
router.include_router(stats.router, tags=["Stats"])
router.include_router(predict.router, tags=["Predict"])
router.include_router(detect.router, tags=["Detect"])
router.include_router(history.router, tags=["History"])
router.include_router(feedback.router, tags=["Feedback"])
router.include_router(metrics.router, tags=["Metrics"])
router.include_router(explain.router, tags=["XAI"])
router.include_router(ensemble.router, tags=["Ensemble"])
router.include_router(recommendations.router, tags=["Recommendations"])
