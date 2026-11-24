"""
Smart Bin Recommendation API — EcoSort AI V3.

GET /api/recommendations/{category}
    Returns a structured smart-bin recommendation for a waste category,
    including bin color, preparation steps, eco tips, and safety info.

GET /api/recommendations
    Returns recommendations for all supported waste categories.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.recycling_engine import MATERIAL_RULES, get_bin_recommendation

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_CATEGORIES = list(MATERIAL_RULES.keys())


class BinRecommendation(BaseModel):
    category: str
    bin_color: str
    bin_label: str
    stream: str
    disposal_status: str
    status_label: str
    action: str
    preparation_steps: List[str]
    safety_info: Optional[str] = None
    eco_tip: str
    regional_caveats: str


class AllRecommendationsResponse(BaseModel):
    categories: List[BinRecommendation]


def _build_recommendation(category: str) -> BinRecommendation:
    try:
        data = get_bin_recommendation(category)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Category '{category}' is not recognised. "
                f"Supported categories: {', '.join(SUPPORTED_CATEGORIES)}."
            ),
        )
    return BinRecommendation(**data)


@router.get(
    "/recommendations/{category}",
    response_model=BinRecommendation,
    summary="Get smart bin recommendation for a specific waste category",
)
async def get_recommendation_by_category(category: str):
    """
    Returns a structured smart-bin recommendation for the given waste category.

    Supported categories: cardboard, glass, metal, paper, plastic, organic, trash, hazardous, unknown.
    """
    return _build_recommendation(category)


@router.get(
    "/recommendations",
    response_model=AllRecommendationsResponse,
    summary="Get smart bin recommendations for all waste categories",
)
async def get_all_recommendations():
    """
    Returns structured disposal recommendations for every supported waste category.
    """
    recommendations = [_build_recommendation(cat) for cat in SUPPORTED_CATEGORIES]
    return AllRecommendationsResponse(categories=recommendations)
