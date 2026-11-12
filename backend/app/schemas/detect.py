"""
Pydantic schemas for Multi-Object Waste Detection (EcoSort AI V2).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.predict import RecyclingDecisionSchema


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")


class ModelCoverage(BaseModel):
    vit_supported: bool = True
    clip_supported: bool = True
    ensemble_supported: bool = True
    unsupported_reason: Optional[str] = None
    compatibility_decision: Optional[str] = None
    coverage_type: str = "full"


class DetectedObject(BaseModel):
    object_id: int
    bbox: BoundingBox
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    detection_label: str
    material_category: str
    vit_prediction: str
    clip_prediction: str
    final_category: str
    class_probabilities: Dict[str, float]
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_status: str
    is_out_of_distribution: bool = False
    model_agreement: bool = True
    model_coverage: Optional[ModelCoverage] = None
    crop_width: Optional[int] = None
    crop_height: Optional[int] = None
    recycling_status: str
    disposal_recommendation: str
    preparation_steps: List[str]
    recycling_decision: RecyclingDecisionSchema


class RecyclingStatusCounts(BaseModel):
    recyclable: int = 0
    conditionally_recyclable: int = 0
    compostable: int = 0
    hazardous: int = 0
    not_standard: int = 0
    unknown: int = 0


class RecyclingSummary(BaseModel):
    total_objects: int
    category_counts: Dict[str, int]
    status_counts: RecyclingStatusCounts


class DetectResponse(BaseModel):
    image_width: int
    image_height: int
    total_objects: int
    objects: List[DetectedObject]
    category_counts: Dict[str, int]
    recycling_summary: RecyclingSummary
    model_versions: Dict[str, str]
    inference_time_ms: float
    annotated_image_base64: str
    detection_model_status: str
