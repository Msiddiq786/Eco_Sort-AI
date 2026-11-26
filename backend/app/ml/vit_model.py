"""
ViT inference wrapper.

The fine-tuned model is loaded from the local checkpoint saved during training.
If the checkpoint does not exist, FileNotFoundError is raised so that the loader
can mark the model as unavailable.
"""
from __future__ import annotations

import logging
import os
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

    # Optional remote Hugging Face Hub download for stateless cloud builds (Render/Railway)
    is_offline = (
        os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
        or os.getenv("TRANSFORMERS_OFFLINE", "").lower() in {"1", "true", "yes"}
    )
    if (
        hf_dir is None
        and not settings.VIT_FINETUNED_PATH.exists()
        and getattr(settings, "AUTO_DOWNLOAD_MODELS", True)
        and getattr(settings, "VIT_HF_REPO_ID", None)
        and not is_offline
    ):
        try:
            from huggingface_hub import snapshot_download

            target_dir = settings.MODELS_DIR / "vit" / "best"
            target_dir.mkdir(parents=True, exist_ok=True)
            sub = getattr(settings, "VIT_HF_SUBFOLDER", None)
            patterns = [f"{sub}/*"] if sub else ["*.json", "*.safetensors", "*.bin", "*.txt"]
            logger.info("Downloading ViT checkpoint from HF Hub repo '%s'...", settings.VIT_HF_REPO_ID)
            dl_dir = snapshot_download(
                repo_id=settings.VIT_HF_REPO_ID,
                local_dir=str(target_dir),
                allow_patterns=patterns,
                token=getattr(settings, "HF_TOKEN", None),
            )
            dl_path = Path(dl_dir) / sub if sub else Path(dl_dir)
            if (dl_path / "config.json").exists():
                hf_dir = dl_path
        except Exception as exc:
            logger.warning("Failed downloading ViT checkpoint from HF Hub: %s", exc)

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


@no_grad()
def predict_vit_batch(
    model: object,
    processor: object,
    pil_images: list,
) -> list[Dict[str, float]]:
    """
    Run batched ViT inference across multiple PIL crops in a single forward pass.

    Returns:
        List of dicts mapping each waste class to its softmax probability.
    """
    if not pil_images:
        return []
    if not HAS_TORCH or model is None or processor is None:
        raise RuntimeError("ViT model runtime is not initialized.")
    if len(pil_images) == 1:
        return [predict_vit(model, processor, pil_images[0])]

    inputs = processor(images=list(pil_images), return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    probs_tensor = torch.softmax(outputs.logits, dim=-1).cpu()
    if probs_tensor.ndim == 1:
        probs_tensor = probs_tensor.unsqueeze(0)

    return [
        {cls: round(float(prob), 6) for cls, prob in zip(settings.WASTE_CLASSES, row)}
        for row in probs_tensor.tolist()
    ]
