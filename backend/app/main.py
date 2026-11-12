"""
Smart Waste Segregation — FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.database.connection import init_db
from app.ml.loader import model_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models once at startup; release on shutdown."""
    await init_db()
    model_registry.load()
    yield
    model_registry.unload()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Smart Waste Segregation API",
        description=(
            "Waste classification using a ViT + CLIP ensemble. "
            "Accepts images and returns the predicted waste category "
            "with recycling recommendations."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")
    return app


app = create_app()
