"""
ViT inference wrapper.

The fine-tuned model is loaded from the local checkpoint saved during training.
If the checkpoint does not exist, FileNotFoundError is raised so that the loader
can mark the model as unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

try:
    import torch
    from transformers import ViTForImageClassification, ViTImageProcessor
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    no_grad = torch.no_grad
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore
    ViTForImageClassification = None  # type: ignore
    ViTImageProcessor = None  # type: ignore
    DEVICE = "cpu"
    def no_grad():  # type: ignore
        def decorator(f):
            return f
        return decorator

from app.core.config import settings

logger = logging.getLogger(__name__)


def load_vit() -> Tuple[object, object]:
    """
    Load the fine-tuned ViT model from disk.

    Supports both:
      1. HuggingFace save_pretrained() directory at models/vit/best/ (Phase 4)
      2. Direct PyTorch checkpoint at settings.VIT_FINETUNED_PATH

    Raises:
        FileNotFoundError: if the checkpoint does not exist or PyTorch is not available.
    """
    if not HAS_TORCH:
        raise FileNotFoundError(
            "PyTorch and Hugging Face Transformers are not yet installed in this environment. "
            "Please install dependencies from backend/requirements.txt."
        )

    # Check for HuggingFace saved checkpoint directory first (Phase 4 standard)
    candidate_dirs = [
        Path("models/vit/best"),
        Path("backend/models/vit/best"),
        settings.MODELS_DIR / "vit" / "best",
        Path(__file__).resolve().parent.parent.parent / "models" / "vit" / "best",
    ]
    hf_dir = None
    for d in candidate_dirs:
        if d.exists() and (d / "config.json").exists():
            hf_dir = d
            break

    if hf_dir is not None:
        logger.info("Loading fine-tuned ViT from HuggingFace checkpoint '%s'...", hf_dir)
        model = ViTForImageClassification.from_pretrained(str(hf_dir))
        model.to(DEVICE)
        model.eval()
        try:
            processor = ViTImageProcessor.from_pretrained(str(hf_dir))
        except Exception:
            processor = ViTImageProcessor.from_pretrained(settings.VIT_CHECKPOINT)
        logger.info("ViT loaded successfully on %s.", DEVICE)
        return model, processor

    checkpoint_path: Path = settings.VIT_FINETUNED_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned ViT checkpoint not found at '{candidate_dirs[0]}' or '{checkpoint_path}'. "
            "Please run backend/training/train_vit.py first."
        )

    logger.info("Loading ViT from %s ...", checkpoint_path)
    model = ViTForImageClassification.from_pretrained(
        str(checkpoint_path.parent),
        num_labels=len(settings.WASTE_CLASSES),
        id2label={i: c for i, c in enumerate(settings.WASTE_CLASSES)},
        label2id={c: i for i, c in enumerate(settings.WASTE_CLASSES)},
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)
    model.eval()

    processor = ViTImageProcessor.from_pretrained(settings.VIT_CHECKPOINT)
    logger.info("ViT loaded on %s.", DEVICE)
    return model, processor


@no_grad()
def predict_vit(
    model: object,
    processor: object,
    pil_image,
) -> Dict[str, float]:
    """
    Run ViT inference on a PIL image.

    Returns:
        dict mapping each waste class to its softmax probability.
    """
    if not HAS_TORCH or model is None or processor is None:
        raise RuntimeError("ViT model runtime is not initialized.")

    inputs = processor(images=pil_image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
    return {cls: round(prob, 6) for cls, prob in zip(settings.WASTE_CLASSES, probs)}
