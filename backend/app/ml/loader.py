"""
ML model loader — loads ViT and CLIP exactly once at application startup.

If the fine-tuned ViT checkpoint is missing the application will still start,
but POST /api/predict will return HTTP 503 with a clear message.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelRegistry:
    """Central store for all loaded ML models."""

    vit_model: Optional[object] = field(default=None, repr=False)
    vit_processor: Optional[object] = field(default=None, repr=False)
    clip_model: Optional[object] = field(default=None, repr=False)
    clip_processor: Optional[object] = field(default=None, repr=False)

    _vit_loaded: bool = False
    _clip_loaded: bool = False

    def load(self) -> None:
        """Load both models. Called once during FastAPI lifespan startup."""
        self._load_clip()
        self._load_vit()

    def _load_vit(self) -> None:
        from app.ml.vit_model import load_vit

        try:
            self.vit_model, self.vit_processor = load_vit()
            self._vit_loaded = True
            logger.info("ViT model loaded successfully.")
        except FileNotFoundError as exc:
            logger.warning("Fine-tuned ViT checkpoint not found: %s", exc)
            logger.warning(
                "Run backend/training/train_vit.py to train the model, "
                "then restart the server."
            )
            self._vit_loaded = False
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error loading ViT: %s", exc)
            self._vit_loaded = False

    def _load_clip(self) -> None:
        from app.core.config import settings
        from app.ml.clip_model import load_clip

        if not getattr(settings, "CLIP_ENABLED", True):
            logger.info("CLIP foundation model is disabled in settings.")
            self._clip_loaded = False
            self.clip_model = None
            self.clip_processor = None
            return

        try:
            self.clip_model, self.clip_processor = load_clip()
            self._clip_loaded = True
            logger.info("CLIP model loaded successfully.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Error loading CLIP model: %s", exc)
            self._clip_loaded = False
            self.clip_model = None
            self.clip_processor = None

    def unload(self) -> None:
        self.vit_model = None
        self.vit_processor = None
        self.clip_model = None
        self.clip_processor = None
        self._vit_loaded = False
        self._clip_loaded = False
        logger.info("Models unloaded.")

    @property
    def vit_loaded(self) -> bool:
        return self._vit_loaded

    @property
    def clip_loaded(self) -> bool:
        return self._clip_loaded

    @property
    def ready(self) -> bool:
        """True only when the fine-tuned ViT is available."""
        return self._vit_loaded


model_registry = ModelRegistry()
