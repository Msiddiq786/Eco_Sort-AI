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
        "a photo of paper waste or newspaper",
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
        "a photo of a yellow or brown banana peel",
        "a photo of an apple core and fruit scraps",
        "a photo of orange peel, citrus rinds, or fruit rinds",
        "a photo of raw vegetable peelings and kitchen scraps",
        "a photo of organic waste, food scraps, or vegetable peels",
        "a photo of fruit peels, banana skins, or leftover vegetables",
        "a photo of compostable kitchen food waste or coffee grounds",
        "a photo of eggshells, fruit rinds, or decaying organic food",
        "a photo of biodegradable wet waste for composting",
        "a photo of leftover food waste or kitchen compost",
        "a photo of green vegetable leaves and salad cuttings",
    ],
    "trash": [
        "a photo of general household trash or landfill garbage",
        "a photo of non-recyclable solid waste or residual refuse",
        "a photo of dirty residual waste destined for landfill",
        "a photo of unsorted household garbage or trash bin contents",
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
