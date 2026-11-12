"""
FastAPI dependency providers.
"""
from app.ml.loader import model_registry


def get_model_registry():
    """Inject the global model registry into route handlers."""
    return model_registry
