"""
GET /api/stats — category distribution and totals derived from the database.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.schemas.history import CategoryStat, StatsResponse
from app.services.prediction_store import get_stats

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
async def statistics(db: AsyncSession = Depends(get_db)):
    """
    Return real database-derived statistics.
    Returns zero / empty values when no predictions exist.
    Never returns synthetic data.
    """
    data = await get_stats(db)
    return StatsResponse(
        total_predictions=data["total"],
        category_distribution=[CategoryStat(**d) for d in data["distribution"]],
        average_confidence=data["avg_confidence"],
        recyclable_count=data["recyclable_count"],
    )
