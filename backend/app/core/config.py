"""
Application configuration loaded from environment variables / .env file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


def _load_yaml_config() -> Dict[str, Any]:
    """Helper to read training/config.yaml if available to sync model config."""
    candidates = [
        _BACKEND_DIR / "training" / "config.yaml",
        Path("training/config.yaml"),
        Path("backend/training/config.yaml"),
    ]
    for p in candidates:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


def _parse_str_list(value: Any) -> List[str]:
    """Accept either JSON arrays or comma-separated strings from environment variables."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                pass
        return [item.strip() for item in raw.split(",") if item.strip()]
    return list(value) if value else []


_yaml_cfg = _load_yaml_config()
_clip_cfg = _yaml_cfg.get("clip", {})
_default_clip_checkpoint = _clip_cfg.get("model_name", "openai/clip-vit-base-patch32")
_default_clip_enabled = _clip_cfg.get("enabled", True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", str(_BACKEND_DIR / ".env")),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────
    ENVIRONMENT: str = "production"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = Field(default=8000, validation_alias=AliasChoices("PORT", "APP_PORT"))
    DEBUG: bool = False
    ENABLE_DOCS: bool = True

    # ── CORS ──────────────────────────────────────────────
    CORS_ORIGINS: Annotated[List[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    CORS_ORIGIN_REGEX: Optional[str] = None

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _validate_cors_origins(cls, v: Any) -> List[str]:
        return _parse_str_list(v)

    # ── Database ──────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./waste_segregation.db"

    # ── File upload ───────────────────────────────────────
    UPLOAD_DIR: Path = Path("uploads")
    FEEDBACK_DIR: Path = _BACKEND_DIR / "data" / "feedback"
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: Annotated[List[str], NoDecode] = [
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
    ]

    @field_validator("ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def _validate_allowed_extensions(cls, v: Any) -> List[str]:
        return [ext.lower().lstrip(".") for ext in _parse_str_list(v)]

    # ── ML Model paths & optional Hub download ────────────
    MODELS_DIR: Path = _BACKEND_DIR / "models"
    VIT_CHECKPOINT: str = "google/vit-base-patch16-224-in21k"
    VIT_MODEL_NAME: str = "google/vit-base-patch16-224-in21k"
    VIT_FINETUNED_PATH: Path = _BACKEND_DIR / "models" / "vit_waste_best.pt"
    CLIP_CHECKPOINT: str = _default_clip_checkpoint
    CLIP_MODEL_NAME: str = _default_clip_checkpoint
    CLIP_ENABLED: bool = _default_clip_enabled

    AUTO_DOWNLOAD_MODELS: bool = True
    VIT_HF_REPO_ID: Optional[str] = None
    VIT_HF_SUBFOLDER: Optional[str] = None
    TRASH_SEG_HF_REPO_ID: str = "turhancan97/yolov8-segment-trash-detection"
    HF_TOKEN: Optional[str] = None

    # ── Ensemble ──────────────────────────────────────────
    ENSEMBLE_VIT_WEIGHT: float = 0.70
    ENSEMBLE_CLIP_WEIGHT: float = 0.30

    # ── Classification labels (must match training order) ─
    WASTE_CLASSES: List[str] = [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash",
    ]

    MODEL_VERSION: str = "1.0.0"


settings = Settings()
