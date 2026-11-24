"""
EcoSort AI V2 — CLIP Zero-Shot Foundation Model Integration (7-Class).

Supports:
0 = cardboard
1 = glass
2 = metal
3 = paper
4 = plastic
5 = organic  <-- V2 CLASS
6 = trash

Improved Architecture:
- Class-Centroid scoring:
  1. Multi-prompt templates encoded per category.
  2. Individual prompt embeddings are L2-normalized.
  3. Prompt embeddings within each category are averaged into a class centroid.
  4. Each class centroid is L2-normalized into unit space.
  5. Image embedding is L2-normalized.
  6. Cosine similarity computed against exactly 7 category centroids (1 logit per class).
  7. Softmax applied over the 7 category logits using model logit_scale.
- Balanced prompt engineering:
  - Exactly 5 distinct, high-separation prompts per class.
  - Removes semantic overlap (e.g. kraft paper removed from paper; distinct corrugated cardboard).
  - Resolves cardboard vs paper confusion and plastic vs glass confusion.
  - Eliminates prompt count bias.
- Precomputed centroid caching for fast, deterministic, thread-safe inference.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import torch

logger = logging.getLogger(__name__)

V2_WASTE_CLASSES: tuple[str, ...] = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "organic",
    "trash",
)

# Empirically validated 5-prompt per class balanced templates
V2_CLIP_PROMPT_TEMPLATES: Dict[str, List[str]] = {
    "cardboard": [
        "a photo of a corrugated cardboard box",
        "a photo of brown corrugated cardboard packaging",
        "a photo of a cardboard shipping carton",
        "a photo of thick corrugated cardboard waste",
        "a photo of flattened brown cardboard box",
    ],
    "glass": [
        "a photo of a glass bottle or glass jar",
        "a photo of transparent glass container or glassware",
        "a photo of clear or green beverage glass bottle",
        "a photo of a wine bottle, beer bottle, or sauce jar made of glass",
        "a photo of heavy rigid glass container for glass recycling",
    ],
    "metal": [
        "a photo of an aluminum soda can or drink can",
        "a photo of a tin food can or steel can",
        "a photo of metallic aluminum can or tin can",
        "a photo of scrap metal or aluminum foil",
        "a photo of metal food can for recycling",
    ],
    "paper": [
        "a photo of office paper, document, or printed sheet",
        "a photo of newspaper, magazine, or paper brochure",
        "a photo of thin sheets of white paper",
        "a photo of a paper cup, paper bag, or paper flyer",
        "a photo of recyclable thin paper waste",
    ],
    "plastic": [
        "a photo of a plastic water bottle or plastic beverage bottle",
        "a photo of a flexible plastic bag or plastic wrapper",
        "a photo of a plastic jug, milk jug, or plastic tub",
        "a photo of a clear PET plastic drink bottle with plastic cap",
        "a photo of lightweight molded plastic packaging or container",
    ],
    "organic": [
        "a photo of banana peels, apple cores, watermelon rind, or citrus peel scraps",
        "a photo of raw vegetable scraps, carrot peelings, and kitchen food cuttings",
        "a photo of compostable food scraps, kitchen waste, and organic leftovers",
        "a photo of coffee grounds, eggshells, and used tea leaves",
        "a photo of fresh fruit leftovers, fruit peels, and biodegradable wet food waste for compost",
    ],
    "trash": [
        "a photo of styrofoam cup or polystyrene foam packaging",
        "a photo of non-recyclable landfill garbage and rubbish",
        "a photo of dirty mixed trash and contaminated waste",
        "a photo of soiled snack packaging, chip bag, or sweet wrapper",
        "a photo of disposable sanitary waste or landfill rubbish",
    ],
}

# Centroid cache: key is (model_id, device_str, tuple_of_classes) -> Tensor [len(classes), D]
_CENTROID_CACHE: Dict[Tuple[int, str, Tuple[str, ...]], torch.Tensor] = {}


@dataclass
class CLIPClassificationResultV2:
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_category": self.predicted_category,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


def clear_clip_v2_cache() -> None:
    """Clear precomputed class centroid cache."""
    global _CENTROID_CACHE
    _CENTROID_CACHE.clear()


def get_or_build_class_centroids(
    model: Any,
    processor: Any,
    classes: tuple[str, ...] = V2_WASTE_CLASSES,
    prompts_dict: Dict[str, List[str]] = V2_CLIP_PROMPT_TEMPLATES,
) -> torch.Tensor:
    """
    Build and cache normalized class-centroid text embeddings for the given classes.
    
    For each class c:
      1. Encode all prompt texts for class c.
      2. Normalize each individual prompt embedding to unit length.
      3. Compute average (mean) embedding across prompts for class c.
      4. Normalize the resulting class centroid vector to unit length.
      
    Returns:
      Tensor of shape [len(classes), embedding_dim] on the model's device.
    """
    device = getattr(model, "device", torch.device("cpu"))
    cache_key = (id(model), str(device), classes)

    if cache_key in _CENTROID_CACHE:
        return _CENTROID_CACHE[cache_key]

    prompt_list: List[str] = []
    prompt_to_class_idx: List[int] = []

    for idx, cls in enumerate(classes):
        templates = prompts_dict.get(cls, [f"a photo of {cls} waste"])
        for tmpl in templates:
            prompt_list.append(tmpl)
            prompt_to_class_idx.append(idx)

    text_inputs = processor(text=prompt_list, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    class_centroids: List[torch.Tensor] = []
    prompt_idx_tensor = torch.tensor(prompt_to_class_idx, device=device)

    for idx in range(len(classes)):
        mask = (prompt_idx_tensor == idx)
        class_embeds = text_features[mask]
        mean_embed = class_embeds.mean(dim=0)
        norm_centroid = mean_embed / mean_embed.norm(dim=-1, keepdim=True)
        class_centroids.append(norm_centroid)

    centroids_tensor = torch.stack(class_centroids, dim=0)  # [len(classes), D]
    _CENTROID_CACHE[cache_key] = centroids_tensor
    return centroids_tensor


def predict_clip_v2(
    model: Any,
    processor: Any,
    image: Image.Image,
    classes: tuple[str, ...] = V2_WASTE_CLASSES,
    use_centroid_scoring: bool = True,
) -> CLIPClassificationResultV2:
    """
    Zero-shot 7-class waste classification using OpenAI CLIP.

    Args:
        model: Loaded HuggingFace CLIPModel instance.
        processor: Loaded HuggingFace CLIPProcessor instance.
        image: PIL Image input.
        classes: Tuple of target class names (default: 7 V2 classes).
        use_centroid_scoring: When True (default), uses class-centroid scoring
                              with 7-class softmax. When False, uses legacy prompt-sum.

    Returns:
        CLIPClassificationResultV2 with predicted_category, confidence, and probabilities.
    """
    if model is None or processor is None:
        raise ValueError("CLIP model and processor must be initialized.")

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")

    rgb_image = image.convert("RGB")
    device = getattr(model, "device", torch.device("cpu"))

    if use_centroid_scoring:
        # ── Class-Centroid Scoring (Validated Method B) ────────────────────────
        centroids = get_or_build_class_centroids(model, processor, classes)

        image_inputs = processor(images=rgb_image, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity against exactly 7 class centroids
            sim = image_features @ centroids.T  # [1, len(classes)]

            # Scale by model temperature parameter (clamped at 100.0)
            if hasattr(model, "logit_scale"):
                logit_scale = model.logit_scale.exp().clamp(max=100.0)
            else:
                logit_scale = torch.tensor(100.0, device=device)

            logits = sim * logit_scale
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        class_probs: Dict[str, float] = {
            cls: float(probs[i]) for i, cls in enumerate(classes)
        }

    else:
        # Legacy fallback
        prompt_list: List[str] = []
        prompt_to_class: List[str] = []
        for cls in classes:
            for tmpl in V2_CLIP_PROMPT_TEMPLATES.get(cls, [f"a photo of {cls} waste"]):
                prompt_list.append(tmpl)
                prompt_to_class.append(cls)

        inputs = processor(text=prompt_list, images=rgb_image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=-1)[0].cpu().numpy()

        class_probs = {c: 0.0 for c in classes}
        for p_val, c_name in zip(probs, prompt_to_class):
            class_probs[c_name] += float(p_val)

    # Normalize so sum = 1.0
    total = sum(class_probs.values())
    if total > 0:
        class_probs = {c: p / total for c, p in class_probs.items()}

    best_cls = max(class_probs, key=class_probs.get)
    best_conf = class_probs[best_cls]

    return CLIPClassificationResultV2(
        predicted_category=best_cls,
        confidence=best_conf,
        probabilities=class_probs,
    )
