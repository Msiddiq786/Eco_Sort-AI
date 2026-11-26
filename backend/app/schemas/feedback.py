"""
Pydantic schemas for Active-Learning User Correction Feedback endpoints.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

ALLOWED_FEEDBACK_CATEGORIES: frozenset[str] = frozenset(
    {"cardboard", "glass", "metal", "paper", "plastic", "organic", "trash"}
)


class FeedbackRequest(BaseModel):
    predicted_category: str = Field(..., min_length=1, max_length=50)
    corrected_category: str = Field(..., min_length=1, max_length=50)
    prediction_id: Optional[int] = None
    detection_record_id: Optional[int] = None
    object_id: Optional[int] = None
    source: str = Field(default="single", max_length=40)
    notes: Optional[str] = Field(default=None, max_length=500)
    image_base64: Optional[str] = Field(default=None, description="Optional Data URL or Base64 crop image.")

    @field_validator("predicted_category", "corrected_category")
    @classmethod
    def _normalize_category(cls, value: str) -> str:
        norm = value.strip().lower()
        if norm not in ALLOWED_FEEDBACK_CATEGORIES:
            raise ValueError(
                f"Invalid category '{value}'. Allowed: {sorted(ALLOWED_FEEDBACK_CATEGORIES)}"
            )
        return norm


class FeedbackResponse(BaseModel):
    id: int
    predicted_category: str
    corrected_category: str
    source: str
    saved_crop_path: Optional[str] = None
    created_at: datetime
    message: str


class FeedbackListItem(BaseModel):
    id: int
    prediction_id: Optional[int] = None
    detection_record_id: Optional[int] = None
    object_id: Optional[int] = None
    predicted_category: str
    corrected_category: str
    source: str
    notes: Optional[str] = None
    saved_crop_path: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class FeedbackStatsResponse(BaseModel):
    total_corrections: int
    by_corrected_category: Dict[str, int]
    recent: List[FeedbackListItem]
