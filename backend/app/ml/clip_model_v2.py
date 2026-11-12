"""
EcoSort AI V2 — CLIP Zero-Shot Foundation Model Integration (7-Class).

Supports:
0 = cardboard
1 = glass
2 = metal
3 = paper
4 = plastic
5 = organic  <-- NEW V2 CLASS
6 = trash
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
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

# Multi-prompt ensemble templates for V2 (rich prompts including organic food scraps)
V2_CLIP_PROMPT_TEMPLATES: Dict[str, List[str]] = {
    "cardboard": [
        "a photo of a cardboard box or cardboard waste",
        "a photo of brown corrugated cardboard packaging",
        "a photo of cardboard shipping carton or box",
        "a photo of flattened cardboard for recycling",
    ],
    "glass": [
        "a photo of a glass bottle or glass container",
        "a photo of clear or colored beverage glass bottle",
        "a photo of a transparent glass jar or bottle",
        "a photo of clean container glass for recycling",
    ],
    "metal": [
        "a photo of a metal can or metallic object",
        "a photo of an aluminum soda can or tin food can",
        "a photo of scrap metal or aluminum beverage can",
        "a photo of a clean tin can or aluminum foil",
    ],
    "paper": [
        "a photo of a paper cup or disposable coffee cup",
        "a photo of paper packaging, paper bag, or kraft paper",
        "a photo of paper waste, newspaper, or paper sheets",
        "a photo of printed office paper, magazine, or document",
        "a photo of clean sheets of paper or newsprint",
        "a photo of dry paper for paper recycling",
    ],
    "plastic": [
        "a photo of a plastic bottle or plastic bag",
        "a photo of clear PET plastic drink bottle or container",
        "a photo of plastic jug, bottle, or packaging wrap",
        "a photo of a single-use plastic container or wrapper",
    ],
    "organic": [
        "a photo of a yellow or brown banana peel or banana skins",
        "a photo of an apple core, fruit core, and fruit scraps",
        "a photo of watermelon rind, melon slices, or melon peel scraps",
        "a photo of orange peel, citrus rinds, or lemon peel scraps",
        "a photo of raw vegetable scraps, peelings, and kitchen scraps",
        "a photo of carrot peels, carrot trimmings, and vegetable cuttings",
        "a photo of fruit leftovers, fruit peels, and fruit waste",
        "a photo of coffee grounds and used tea leaves",
        "a photo of eggshells, fruit rinds, or decaying organic food scraps",
        "a photo of compostable food scraps, kitchen waste, and organic leftovers",
        "a photo of biodegradable wet waste for composting",
        "a photo of green vegetable leaves, lettuce salad trimmings, and celery ends",
    ],
    "trash": [
        "a photo of general household landfill trash or refuse",
        "a photo of non-recyclable solid domestic garbage or dirty rubbish",
        "a photo of non-compostable, non-recyclable composite trash, sweepings, or litter",
        "a photo of disposable sanitary waste, dirty packaging, or mixed dry rubbish",
    ],
}


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


def predict_clip_v2(
    model: Any,
    processor: Any,
    image: Image.Image,
    classes: tuple[str, ...] = V2_WASTE_CLASSES,
) -> CLIPClassificationResultV2:
    """
    Zero-shot multi-prompt ensemble classification for V2 7-class system using CLIP.
    """
    if model is None or processor is None:
        raise ValueError("CLIP model and processor must be initialized.")

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")

    rgb_image = image.convert("RGB")
    device = getattr(model, "device", "cpu")

    # Aggregate candidate text prompts
    all_prompts: List[str] = []
    prompt_to_class: List[str] = []

    for cls in classes:
        templates = V2_CLIP_PROMPT_TEMPLATES.get(cls, [f"a photo of {cls} waste"])
        for tmpl in templates:
            all_prompts.append(tmpl)
            prompt_to_class.append(cls)

    inputs = processor(
        text=all_prompts,
        images=rgb_image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: [1, num_prompts]
        probs = torch.softmax(logits_per_image, dim=-1)[0].cpu().numpy()

    # Sum probabilities per class
    class_probs: Dict[str, float] = {c: 0.0 for c in classes}
    for prob, cls in zip(probs, prompt_to_class):
        class_probs[cls] += float(prob)

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
