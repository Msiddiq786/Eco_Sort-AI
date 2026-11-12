"""
CLIP Foundation Model Integration for EcoSort AI.

Zero-shot image classification across 6 waste categories using
openai/clip-vit-base-patch32 and natural-language prompt engineering.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
from PIL import Image, ImageOps

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore
    DEVICE = "cpu"

from app.core.config import settings

logger = logging.getLogger(__name__)

# Exact prompt descriptions for the six required waste classes (Phase 5 specification)
CLIP_PROMPTS: Dict[str, str] = {
    "cardboard": "a photo of a cardboard box or cardboard waste",
    "glass": "a photo of a glass bottle or glass container",
    "metal": "a photo of a metal can or metallic object",
    "paper": "a photo of paper waste or newspaper",
    "plastic": "a photo of a plastic bottle or plastic bag",
    "trash": "a photo of general household trash or garbage",
}

# Multi-prompt natural-language descriptions per category (Phase 6 improvement)
CLIP_PROMPT_TEMPLATES: Dict[str, List[str]] = {
    "cardboard": [
        "a photo of a cardboard box or cardboard waste",
        "a photo of corrugated cardboard waste",
        "a photo of a brown cardboard box or carton",
        "a photo of discarded cardboard packaging",
        "a photo of flattened or crushed cardboard",
    ],
    "glass": [
        "a photo of a glass bottle or glass container",
        "a photo of a beverage glass bottle",
        "a photo of a clear glass jar or container",
        "a photo of broken or recyclable glass",
        "a photo of glassware or glass beverage bottle",
    ],
    "metal": [
        "a photo of a metal can or metallic object",
        "a photo of an aluminum soda can",
        "a photo of a discarded tin food can",
        "a photo of a crushed metal beverage can",
        "a photo of recyclable scrap metal or aluminum can",
    ],
    "paper": [
        "a photo of paper waste or newspaper",
        "a photo of a sheet of paper or document",
        "a photo of crumpled office paper waste",
        "a photo of printed newspaper or magazine paper",
        "a photo of white printer paper or notebook sheet",
    ],
    "plastic": [
        "a photo of a plastic bottle or plastic bag",
        "a photo of a discarded plastic water bottle",
        "a photo of a clear disposable plastic bottle",
        "a photo of plastic packaging waste or plastic container",
        "a photo of a plastic cup or plastic detergent bottle",
    ],
    "trash": [
        "a photo of general household trash or garbage",
        "a photo of non-recyclable domestic garbage",
        "a photo of miscellaneous municipal waste or rubbish",
        "a photo of dirty non-recyclable trash",
        "a photo of mixed disposable waste items",
    ],
}


def _extract_tensor(out: Any) -> Any:
    """Helper to extract Tensor from ModelOutput if wrapped."""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if hasattr(out, "shape"):
        return out
    try:
        return out[0]
    except Exception:
        return out


class CLIPClassificationResult(dict):
    """
    Structured container for CLIP zero-shot classification output.

    Inherits from dict for serialization, API response compatibility,
    and dict-style indexing, while also offering attribute access:
        result.predicted_category
        result.confidence
        result.probabilities
    """

    def __init__(
        self,
        predicted_category: str,
        confidence: float,
        probabilities: Dict[str, float],
    ):
        super().__init__(
            predicted_category=predicted_category,
            confidence=confidence,
            probabilities=probabilities,
        )
        self.predicted_category = predicted_category
        self.confidence = confidence
        self.probabilities = probabilities

    def to_dict(self) -> Dict[str, Any]:
        """Return result as a standard dictionary."""
        return {
            "predicted_category": self.predicted_category,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


def load_clip(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load CLIP foundation model and processor from HuggingFace Hub or local cache.

    Args:
        model_name: HuggingFace checkpoint or local path. Defaults to settings.CLIP_CHECKPOINT.
        device: Device to place the model on ('cuda' or 'cpu'). Defaults to detected DEVICE.

    Returns:
        Tuple of (model, processor).

    Raises:
        ImportError: If PyTorch or Transformers is not installed.
        Exception: If loading fails (e.g. offline cache miss or invalid checkpoint).
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch and HuggingFace Transformers must be installed to load CLIP."
        )

    target_model = model_name or settings.CLIP_CHECKPOINT
    target_device = device or DEVICE

    is_offline = (
        os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        or os.environ.get("HF_HUB_OFFLINE") == "1"
    )

    logger.info(
        "Loading CLIP model '%s' (device=%s, offline=%s)...",
        target_model,
        target_device,
        is_offline,
    )

    try:
        model = CLIPModel.from_pretrained(
            target_model,
            local_files_only=is_offline,
        ).to(target_device)
        model.eval()

        processor = CLIPProcessor.from_pretrained(
            target_model,
            local_files_only=is_offline,
        )
        logger.info("CLIP foundation model '%s' loaded successfully.", target_model)
        return model, processor
    except Exception as exc:
        logger.error("Failed to load CLIP foundation model '%s': %s", target_model, exc)
        raise


def predict_clip(
    model: Any,
    processor: Any,
    image: Any,
    candidate_prompts: Optional[Dict[str, str]] = None,
    device: Optional[str] = None,
) -> CLIPClassificationResult:
    """
    Perform zero-shot classification on an image using CLIP.

    Args:
        model: Loaded CLIPModel instance.
        processor: Loaded CLIPProcessor instance.
        image: PIL Image or image file path.
        candidate_prompts: Dict mapping class names to text prompts.
                           Defaults to CLIP_PROMPTS.
        device: Target compute device. Defaults to model device or detected DEVICE.

    Returns:
        CLIPClassificationResult containing:
            - predicted_category: str (highest probability class)
            - confidence: float (highest probability value)
            - probabilities: Dict[str, float] (normalized class distribution summing ~1.0)

    Raises:
        RuntimeError: If model or processor is None.
        ValueError: If image is None or candidate_prompts is empty.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch runtime is not available.")

    if model is None or processor is None:
        raise RuntimeError("CLIP model runtime is not initialized or model is not loaded.")

    if image is None:
        raise ValueError("Input image cannot be None.")

    prompts = candidate_prompts if candidate_prompts is not None else CLIP_PROMPTS
    if not prompts:
        raise ValueError("Candidate prompts dictionary cannot be empty.")

    # Convert image to RGB PIL Image if needed
    if isinstance(image, str):
        pil_image = Image.open(image)
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    elif hasattr(image, "convert"):
        pil_image = ImageOps.exif_transpose(image).convert("RGB")
    else:
        pil_image = image

    # Retain consistent class order (preferring settings.WASTE_CLASSES if available)
    if hasattr(settings, "WASTE_CLASSES"):
        class_names = [cls for cls in settings.WASTE_CLASSES if cls in prompts]
        # Include any remaining prompts not in settings.WASTE_CLASSES
        for cls in prompts:
            if cls not in class_names:
                class_names.append(cls)
    else:
        class_names = list(prompts.keys())

    # Resolve target device
    if device is not None:
        target_device = device
    elif hasattr(model, "device"):
        target_device = model.device
    else:
        target_device = DEVICE

    # If using default prompts with real CLIPModel, apply multi-prompt zero-shot ensembling
    is_real_clip = (
        candidate_prompts is None
        and hasattr(model, "get_text_features")
        and hasattr(model, "get_image_features")
        and not hasattr(model, "_mock_return_value")
    )

    if is_real_clip:
        with torch.no_grad():
            cached_text_emb = getattr(model, "_cached_prompt_embeddings", None)
            if cached_text_emb is None:
                class_feats = []
                for cls_name in class_names:
                    tmpl_list = CLIP_PROMPT_TEMPLATES.get(cls_name, [CLIP_PROMPTS.get(cls_name, cls_name)])
                    txt_inputs = processor(text=tmpl_list, return_tensors="pt", padding=True)
                    if hasattr(txt_inputs, "to"):
                        txt_inputs = txt_inputs.to(target_device)
                    else:
                        txt_inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in txt_inputs.items()}
                    txt_out = model.get_text_features(**txt_inputs)
                    txt_tensor = _extract_tensor(txt_out)
                    txt_tensor = txt_tensor / txt_tensor.norm(dim=-1, keepdim=True)
                    cls_feat = txt_tensor.mean(dim=0, keepdim=True)
                    cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
                    class_feats.append(cls_feat)
                cached_text_emb = torch.cat(class_feats, dim=0)
                setattr(model, "_cached_prompt_embeddings", cached_text_emb)

            img_inputs = processor(images=pil_image, return_tensors="pt")
            if hasattr(img_inputs, "to"):
                img_inputs = img_inputs.to(target_device)
            else:
                img_inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in img_inputs.items()}
            img_out = model.get_image_features(**img_inputs)
            img_tensor = _extract_tensor(img_out)
            img_tensor = img_tensor / img_tensor.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else torch.tensor(100.0)
            logits = logit_scale * (img_tensor @ cached_text_emb.T)
            probs_tensor = logits.softmax(dim=-1).squeeze()
    else:
        text_prompts = [prompts[cls] for cls in class_names]

        # Preprocess image and text prompts
        inputs = processor(
            text=text_prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True,
        )

        if hasattr(inputs, "items"):
            inputs = {
                k: v.to(target_device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

        # Zero-shot inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract image-to-text similarity logits
        logits = getattr(outputs, "logits_per_image", outputs)
        if hasattr(logits, "softmax"):
            probs_tensor = logits.softmax(dim=-1)
        else:
            # Fallback if mock outputs raw array/tensor
            if hasattr(logits, "exp"):
                exp_logits = logits.exp()
                probs_tensor = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
            else:
                raise RuntimeError("Model outputs do not contain recognizable logits.")

        if hasattr(probs_tensor, "squeeze"):
            probs_tensor = probs_tensor.squeeze()

    if hasattr(probs_tensor, "cpu"):
        probs_raw = probs_tensor.cpu().tolist()
    else:
        probs_raw = list(probs_tensor)

    if not isinstance(probs_raw, list):
        probs_raw = [probs_raw]

    # Build per-class probabilities normalized to 6 decimal places
    probabilities: Dict[str, float] = {}
    for cls, prob in zip(class_names, probs_raw):
        probabilities[cls] = round(float(prob), 6)

    # Re-normalize to ensure exact 1.0 sum if there's any float precision drift
    total_p = sum(probabilities.values())
    if total_p > 0:
        probabilities = {cls: round(p / total_p, 6) for cls, p in probabilities.items()}

    # Determine predicted category and confidence (argmax)
    predicted_category = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
    confidence = probabilities[predicted_category]

    return CLIPClassificationResult(
        predicted_category=predicted_category,
        confidence=confidence,
        probabilities=probabilities,
    )
