"""
Pydantic schemas for history and stats endpoints.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


class HistoryItem(BaseModel):
    id: int
    filename: str
    original_filename: str
    image_path: str
    predicted_category: str
    confidence: float
    vit_prediction: str
    clip_prediction: str
    class_probabilities: Dict[str, float]
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedHistory(BaseModel):
    items: List[HistoryItem]
    total: int
    page: int
    page_size: int


class CategoryStat(BaseModel):
    category: str
    count: int
    percentage: float
    is_recyclable: bool = False


class StatsResponse(BaseModel):
    total_predictions: int
    category_distribution: List[CategoryStat]
    average_confidence: float
    recyclable_count: int


class DetectionObjectSummary(BaseModel):
    object_id: int
    material_category: str
    calibrated_confidence: float
    vit_prediction: str
    clip_prediction: str
    recycling_status: str
    model_agreement: bool = True
    resin_code: Optional[str] = None


class DetectionHistoryItem(BaseModel):
    id: int
    filename: str
    original_filename: str
    image_path: str
    total_objects: int
    category_counts: Dict[str, int]
    status_counts: Dict[str, int]
    objects: List[DetectionObjectSummary]
    inference_time_ms: float
    detection_model_status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedDetectionHistory(BaseModel):
    items: List[DetectionHistoryItem]
    total: int
    page: int
    page_size: int
