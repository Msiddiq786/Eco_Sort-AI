"""
Pydantic schemas for Multi-Object Waste Detection (EcoSort AI V2).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, computed_field

from app.schemas.predict import RecyclingDecisionSchema, ResinInfoSchema


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left coordinate (pixels)")
    y1: float = Field(..., description="Top coordinate (pixels)")
    x2: float = Field(..., description="Right coordinate (pixels)")
    y2: float = Field(..., description="Bottom coordinate (pixels)")
    nx1: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized left coordinate [0, 1]")
    ny1: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized top coordinate [0, 1]")
    nx2: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized right coordinate [0, 1]")
    ny2: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized bottom coordinate [0, 1]")

    @computed_field
    @property
    def x(self) -> float:
        return self.x1

    @computed_field
    @property
    def y(self) -> float:
        return self.y1

    @computed_field
    @property
    def width(self) -> float:
        return round(max(0.0, self.x2 - self.x1), 2)

    @computed_field
    @property
    def height(self) -> float:
        return round(max(0.0, self.y2 - self.y1), 2)


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
    vit_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    clip_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    raw_ensemble_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    final_category: str
    class_probabilities: Dict[str, float]
    vit_probabilities: Optional[Dict[str, float]] = None
    clip_probabilities: Optional[Dict[str, float]] = None
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_status: str
    is_out_of_distribution: bool = False
    model_agreement: bool = True
    model_coverage: Optional[ModelCoverage] = None
    detector_source: Optional[str] = None
    crop_width: Optional[int] = None
    crop_height: Optional[int] = None
    crop_image_base64: Optional[str] = None
    recycling_status: str
    disposal_recommendation: str
    preparation_steps: List[str]
    recycling_decision: RecyclingDecisionSchema
    resin_info: Optional[ResinInfoSchema] = None

    @computed_field
    @property
    def category(self) -> str:
        return self.material_category

    @computed_field
    @property
    def classification_confidence(self) -> float:
        return self.calibrated_confidence

    @computed_field
    @property
    def probabilities(self) -> Dict[str, float]:
        return self.class_probabilities

    @computed_field
    @property
    def recommendation(self) -> Dict[str, Any]:
        return self.recycling_decision.model_dump()


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
    scan_id: Optional[int] = None
    image_width: int
    image_height: int
    total_objects: int
    objects: List[DetectedObject]
    category_counts: Dict[str, int]
    recycling_summary: RecyclingSummary
    model_versions: Dict[str, str]
    inference_time_ms: float
    annotated_image_base64: str
    clean_image_base64: Optional[str] = None
    detection_model_status: str

    @computed_field
    @property
    def success(self) -> bool:
        return True

    @computed_field
    @property
    def image(self) -> Dict[str, int]:
        return {"width": self.image_width, "height": self.image_height}
