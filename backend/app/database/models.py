"""ORM models for the SQLite database."""
import json
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, func
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
