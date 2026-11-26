"""ORM models for the SQLite database."""
import json
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class Prediction(Base):
    """Stores every waste classification result."""

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_predicted_category", "predicted_category"),
        Index("ix_predictions_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Safe UUID-prefixed filename stored on disk
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # Original name supplied by the user (for display)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # Path relative to the uploads directory
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    predicted_category: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    vit_prediction: Mapped[str] = mapped_column(String(50), nullable=False)
    clip_prediction: Mapped[str] = mapped_column(String(50), nullable=False)
    # JSON-encoded dict of {class_name: probability}
    class_probabilities: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def probabilities_dict(self) -> dict:
        return json.loads(self.class_probabilities)


class EnsembleConfig(Base):
    """Stores active and historical validated ensemble weight configurations."""

    __tablename__ = "ensemble_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vit_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.70)
    clip_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.30)
    macro_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(30), nullable=False, default="default")
    optimized_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class XAIResult(Base):
    """Stores XAI metadata and reasoning generated for predictions."""

    __tablename__ = "xai_results"
    __table_args__ = (
        Index("ix_xai_results_prediction_id", "prediction_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False
    )
    attention_method: Mapped[str] = mapped_column(
        String(50), nullable=False, default="attention_rollout"
    )
    models_agree: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    reasoning_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DetectionRecord(Base):
    """Stores persisted multi-object /api/detect scan summaries for History & Audits."""

    __tablename__ = "detection_records"
    __table_args__ = (
        Index("ix_detection_records_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    total_objects: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    category_counts: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    status_counts: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    objects_summary: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    inference_time_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    detection_model_status: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class FeedbackCorrection(Base):
    """Stores user-confirmed category corrections for active-learning curation."""

    __tablename__ = "feedback_corrections"
    __table_args__ = (
        Index("ix_feedback_corrected_category", "corrected_category"),
        Index("ix_feedback_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True
    )
    detection_record_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("detection_records.id", ondelete="SET NULL"), nullable=True
    )
    object_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    predicted_category: Mapped[str] = mapped_column(String(50), nullable=False)
    corrected_category: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(40), nullable=False, default="single")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    saved_crop_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
