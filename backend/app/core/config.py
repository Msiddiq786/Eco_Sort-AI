"""
Application configuration loaded from environment variables / .env file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import yaml

from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml_config() -> Dict[str, Any]:
    """Helper to read training/config.yaml if available to sync model config."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "training" / "config.yaml",
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


_yaml_cfg = _load_yaml_config()
_clip_cfg = _yaml_cfg.get("clip", {})
_default_clip_checkpoint = _clip_cfg.get("model_name", "openai/clip-vit-base-patch32")
_default_clip_enabled = _clip_cfg.get("enabled", True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False

    # ── CORS ──────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # ── Database ──────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./waste_segregation.db"

    # ── File upload ───────────────────────────────────────
    UPLOAD_DIR: Path = Path("uploads")
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp", "bmp"]

    # ── ML Model paths ────────────────────────────────────
    MODELS_DIR: Path = Path("models")
    VIT_CHECKPOINT: str = "google/vit-base-patch16-224-in21k"
    VIT_MODEL_NAME: str = "google/vit-base-patch16-224-in21k"
    VIT_FINETUNED_PATH: Path = Path("models/vit_waste_best.pt")
    CLIP_CHECKPOINT: str = _default_clip_checkpoint
    CLIP_MODEL_NAME: str = _default_clip_checkpoint
    CLIP_ENABLED: bool = _default_clip_enabled

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
