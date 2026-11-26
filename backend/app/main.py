"""
Smart Waste Segregation — FastAPI Application Entry Point
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import health_check
from app.api.routes import router
from app.core.config import settings
from app.core.dependencies import get_model_registry
from app.database.connection import init_db
from app.ml.loader import model_registry

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB tables and load ML models once at startup; release on shutdown."""
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()
    logger.info("Database initialized successfully.")
    model_registry.load()
    logger.info(
        "Model startup complete (vit_loaded=%s, clip_loaded=%s).",
        model_registry.vit_loaded,
        model_registry.clip_loaded,
    )
    yield
    model_registry.unload()


def create_app() -> FastAPI:
    docs_url = "/api/docs" if settings.ENABLE_DOCS else None
    redoc_url = "/api/redoc" if settings.ENABLE_DOCS else None
    openapi_url = "/api/openapi.json" if settings.ENABLE_DOCS else None

    app = FastAPI(
        title="Smart Waste Segregation API",
        description=(
            "Waste classification using a ViT + CLIP ensemble. "
            "Accepts images and returns the predicted waste category "
            "with recycling recommendations."
        ),
        version="1.0.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_origin_regex=settings.CORS_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "service": "EcoSort AI API",
            "status": "ok",
            "version": "1.0.0",
            "health": "/health",
            "api_health": "/api/health",
            "docs": docs_url,
        }

    @app.get("/health", tags=["Health"])
    async def root_health(registry=Depends(get_model_registry)):
        return await health_check(registry=registry)

    app.include_router(router, prefix="/api")
    return app


app = create_app()
