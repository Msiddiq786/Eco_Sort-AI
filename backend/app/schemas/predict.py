"""
Pydantic schemas for prediction request / response.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ClassProbabilities(BaseModel):
    cardboard: float
    glass: float
    metal: float
    paper: float
    plastic: float
    trash: float


class RecyclingDecisionSchema(BaseModel):
    disposal_status: str
    status_label: str
    action: str
    preparation_steps: list[str]
    disposal_bin: str
    reason: str
    safety_info: Optional[str] = None
    regional_caveats: str


class PredictResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    image_path: str
    predicted_category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    calibrated_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    vit_prediction: str
    clip_prediction: str
    class_probabilities: ClassProbabilities
    calibrated_probabilities: Optional[ClassProbabilities] = None
    vit_probabilities: Optional[ClassProbabilities] = None
    clip_probabilities: Optional[ClassProbabilities] = None
    is_low_confidence: bool = False
    is_ambiguous: bool = False
    is_out_of_distribution: bool = False
    confidence_status: str = "normal"
    ood_warning: Optional[str] = None
    recycling_recommendation: str
    recycling_decision: Optional[RecyclingDecisionSchema] = None
    model_version: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelInfoResponse(BaseModel):
    vit_checkpoint: str
    clip_checkpoint: str
    vit_model_name: Optional[str] = None
    clip_model_name: Optional[str] = None
    clip_enabled: bool = True
    ensemble_vit_weight: float
    ensemble_clip_weight: float
    temperature: float = 0.35
    uncalibrated_ece: float = 0.3897
    calibrated_ece: float = 0.0666
    waste_classes: list[str]
    model_version: str
    vit_loaded: bool
    clip_loaded: bool


class CLIPPredictResponse(BaseModel):
    predicted_category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: ClassProbabilities
