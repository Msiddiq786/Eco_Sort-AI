"""
EcoSort AI V3 — Plastic Resin Identification Code (#1–#7) & Hazard Analyzer.

Combines OpenCV CLAHE contrast/ROI analysis, optional OCR text regex matching
(if pytesseract + tesseract binary are available in the environment), and
cached CLIP zero-shot polymer/hazard centroids to provide fine-grained polymer
subtype classification and preparation guidance without crashing when external
OCR binaries are absent.
"""
from __future__ import annotations

import logging
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

try:
    import pytesseract  # type: ignore

    HAS_PYTESSERACT = shutil.which("tesseract") is not None
except ImportError:
    pytesseract = None
    HAS_PYTESSERACT = False

logger = logging.getLogger(__name__)

# Official ASTM D7611 / SPI Resin Identification Codes (1-7)
RESIN_CATALOG: Dict[int, Dict[str, Any]] = {
    1: {
        "code": "#1 PET",
        "name": "Polyethylene Terephthalate (PET / PETE)",
        "tier": "Widely Recyclable",
        "guidance": "Empty liquid, flatten bottle, and keep cap on if accepted locally.",
        "prompts": [
            "a clear disposable PET water bottle or soda bottle made of thin plastic",
            "transparent PET beverage bottle with screw cap",
            "clear plastic drinking bottle PET #1",
        ],
    },
    2: {
        "code": "#2 HDPE",
        "name": "High-Density Polyethylene (HDPE)",
        "tier": "Widely Recyclable",
        "guidance": "Rinse thick bottle or jug thoroughly before placing in dry recycling.",
        "prompts": [
            "an opaque plastic milk jug detergent bottle or shampoo bottle made of thick HDPE",
            "colored sturdy HDPE household bottle container",
            "thick opaque plastic bottle HDPE #2",
        ],
    },
    3: {
        "code": "#3 PVC",
        "name": "Polyvinyl Chloride (PVC / V)",
        "tier": "Specialty / Rarely Curbside Recyclable",
        "guidance": "Often excluded from standard curbside recycling; check municipal rules.",
        "prompts": [
            "rigid PVC plastic blister packaging or vinyl pipe or plastic pipe tubing",
            "clear rigid clamshell blister pack PVC plastic",
        ],
    },
    4: {
        "code": "#4 LDPE",
        "name": "Low-Density Polyethylene (LDPE)",
        "tier": "Film Drop-Off / Conditional",
        "guidance": "Thin plastic films and carrier bags tangle sorting machinery; drop at film collection bins if dry.",
        "prompts": [
            "thin flexible plastic grocery bag film wrap or transparent polythene bag",
            "crumpled thin plastic wrapper film pouch LDPE",
            "soft squeezable plastic bag or bubble wrap film",
        ],
    },
    5: {
        "code": "#5 PP",
        "name": "Polypropylene (PP)",
        "tier": "Conditionally Recyclable",
        "guidance": "Rinse food residue completely from container tubs and lids before recycling.",
        "prompts": [
            "clear or semi-transparent rigid plastic food container tray takeout tub or yogurt cup",
            "microwaveable polypropylene plastic meal container or rigid lid",
            "rigid reusable plastic bowl or storage tub PP #5",
        ],
    },
    6: {
        "code": "#6 PS",
        "name": "Polystyrene / Styrofoam (PS / EPS)",
        "tier": "Generally Non-Recyclable (Residual)",
        "guidance": "Expanded foam trays and disposable foam cups are usually rejected in curbside recycling.",
        "prompts": [
            "white expanded polystyrene styrofoam foam food tray meat tray or foam cup",
            "lightweight cellular white styrofoam cup or foam takeout box PS #6",
        ],
    },
    7: {
        "code": "#7 Other",
        "name": "Other / Multi-Layer Composite Plastics",
        "tier": "Non-Standard / Verify Stamp",
        "guidance": "Multi-layer or polycarbonate plastics generally require specialty take-back programs.",
        "prompts": [
            "multi-layer laminated chip pouch or hard polycarbonate bottle composite plastic #7",
            "mixed composite plastic packaging not PET or HDPE",
        ],
    },
}

_RIC_REGEX_PATTERNS: List[Tuple[int, re.Pattern[str]]] = [
    (1, re.compile(r"\b(?:PETE?|PET\s*1)\b", re.IGNORECASE)),
    (2, re.compile(r"\b(?:HDPE|PE-HD)\b", re.IGNORECASE)),
    (3, re.compile(r"\b(?:PVC|PE-V)\b", re.IGNORECASE)),
    (4, re.compile(r"\b(?:LDPE|PE-LD|LLDPE)\b", re.IGNORECASE)),
    (5, re.compile(r"\bPP\b", re.IGNORECASE)),
    (6, re.compile(r"\b(?:PS|EPS|STYROFOAM)\b", re.IGNORECASE)),
    (7, re.compile(r"\b(?:PC|ABS|PLA|OTHER\s*0?7)\b", re.IGNORECASE)),
]

_HAZARD_TEXT_KEYWORDS: Dict[str, str] = {
    "lithium": "Lithium Battery / Fire Hazard",
    "li-ion": "Lithium-Ion Battery Hazard",
    "battery": "Battery / Heavy Metal Hazard",
    "flammable": "Flammable / Aerosol Content",
    "corrosive": "Corrosive Chemical Container",
    "biohazard": "Medical / Biohazard Waste",
    "toxic": "Toxic / Hazardous Residue",
}

_HAZARD_PROMPTS: Dict[str, List[str]] = {
    "benign": [
        "ordinary empty beverage can tin food container or benign dry waste item",
        "common household waste item without electronics or hazardous symbols",
    ],
    "battery_ewaste": [
        "a lithium battery cell power bank electronics circuit board or AA rechargeable battery",
        "discarded electronic device battery pack e-waste",
    ],
    "pressurized_chemical": [
        "a pressurized spray aerosol paint canister chemical cleaner bottle or hazard symbol container",
        "spray paint can or chemical solvent bottle with warning label",
    ],
}

_RESIN_CENTROID_CACHE: Dict[int, Any] = {}
_HAZARD_CENTROID_CACHE: Dict[int, Any] = {}


def _extract_enhanced_text(pil_image: Image.Image) -> str:
    """
    Use OpenCV CLAHE normalization on cropped RGB image to enhance embossed/printed text,
    and extract text via pytesseract only when Tesseract binary + binding are available.
    """
    if not HAS_PYTESSERACT or pytesseract is None:
        return ""
    try:
        gray = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        text = pytesseract.image_to_string(enhanced, config="--psm 6")
        return " ".join(text.split())[:240]
    except Exception as exc:
        logger.debug("Optional pytesseract extraction skipped: %s", exc)
        return ""


def _has_triangular_or_label_roi(pil_image: Image.Image) -> bool:
    """Detect whether an image contains an embossed/printed label region via OpenCV contours."""
    try:
        arr = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm = clahe.apply(gray)
        edges = cv2.Canny(norm, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours[:60]:
            area = cv2.contourArea(cnt)
            if area < 60:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) in (3, 4):
                return True
        return False
    except Exception:
        return False


def _get_resin_centroids(clip_model: Any, clip_processor: Any):
    model_id = id(clip_model)
    if model_id in _RESIN_CENTROID_CACHE:
        return _RESIN_CENTROID_CACHE[model_id]

    device = getattr(clip_model, "device", torch.device("cpu"))
    resin_ids = sorted(RESIN_CATALOG.keys())
    prompts: List[str] = []
    prompt_resin_idx: List[int] = []
    for idx, rid in enumerate(resin_ids):
        for p in RESIN_CATALOG[rid]["prompts"]:
            prompts.append(p)
            prompt_resin_idx.append(idx)

    inputs = clip_processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)

    idx_t = torch.tensor(prompt_resin_idx, device=device)
    centroids = []
    for idx in range(len(resin_ids)):
        mean_emb = feats[idx_t == idx].mean(dim=0)
        centroids.append(mean_emb / mean_emb.norm(dim=-1, keepdim=True))
    stacked = torch.stack(centroids, dim=0)
    _RESIN_CENTROID_CACHE[model_id] = (resin_ids, stacked)
    return resin_ids, stacked


def _get_hazard_centroids(clip_model: Any, clip_processor: Any):
    model_id = id(clip_model)
    if model_id in _HAZARD_CENTROID_CACHE:
        return _HAZARD_CENTROID_CACHE[model_id]

    device = getattr(clip_model, "device", torch.device("cpu"))
    labels = list(_HAZARD_PROMPTS.keys())
    prompts: List[str] = []
    prompt_idx: List[int] = []
    for idx, key in enumerate(labels):
        for p in _HAZARD_PROMPTS[key]:
            prompts.append(p)
            prompt_idx.append(idx)

    inputs = clip_processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)

    idx_t = torch.tensor(prompt_idx, device=device)
    centroids = []
    for idx in range(len(labels)):
        mean_emb = feats[idx_t == idx].mean(dim=0)
        centroids.append(mean_emb / mean_emb.norm(dim=-1, keepdim=True))
    stacked = torch.stack(centroids, dim=0)
    _HAZARD_CENTROID_CACHE[model_id] = (labels, stacked)
    return labels, stacked


def analyze_resin_and_hazards(
    pil_image: Image.Image,
    category: str,
    clip_model: Optional[Any] = None,
    clip_processor: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Analyze plastic items for ASTMD7611 Resin Code (#1 PET - #7 Other) and trash/metal
    items for hazardous indicators (batteries, pressurized aerosols, warning keywords).

    Returns None when category is outside plastic/trash/metal or no actionable telemetry
    is found.
    """
    results = analyze_resin_and_hazards_batch(
        [(pil_image, category)],
        clip_model=clip_model,
        clip_processor=clip_processor,
    )
    return results[0]


def analyze_resin_and_hazards_batch(
    items: List[Tuple[Image.Image, str]],
    clip_model: Optional[Any] = None,
    clip_processor: Optional[Any] = None,
) -> List[Optional[Dict[str, Any]]]:
    """
    Batched Resin Code & Hazard analysis across one or more (pil_image, category) pairs.
    Only performs CLIP vision encoding for eligible plastic/trash/metal items.
    """
    if not items:
        return []

    outputs: List[Optional[Dict[str, Any]]] = [None] * len(items)
    can_use_clip = HAS_TORCH and clip_model is not None and clip_processor is not None

    plastic_indices: List[int] = []
    hazard_indices: List[int] = []
    extracted_texts: Dict[int, str] = {}

    for i, (img, raw_cat) in enumerate(items):
        cat = (raw_cat or "").strip().lower()
        if cat not in ("plastic", "trash", "metal"):
            continue
        txt = _extract_enhanced_text(img)
        if txt:
            extracted_texts[i] = txt
        if cat == "plastic":
            plastic_indices.append(i)
        else:
            hazard_indices.append(i)

    # 1. Plastic Resin Identification (#1 PET .. #7 Other)
    if plastic_indices:
        clip_probs_by_pos: Dict[int, Tuple[int, float, float]] = {}
        if can_use_clip:
            try:
                resin_ids, centroids = _get_resin_centroids(clip_model, clip_processor)
                device = getattr(clip_model, "device", torch.device("cpu"))
                logit_scale = (
                    clip_model.logit_scale.exp().clamp(max=100.0)
                    if hasattr(clip_model, "logit_scale")
                    else torch.tensor(100.0, device=device)
                )
                rgb_imgs = [items[idx][0].convert("RGB") for idx in plastic_indices]
                img_inputs = clip_processor(images=rgb_imgs, return_tensors="pt")
                img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
                with torch.no_grad():
                    img_feats = clip_model.get_image_features(**img_inputs)
                    if hasattr(img_feats, "pooler_output"):
                        img_feats = img_feats.pooler_output
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    logits = (img_feats @ centroids.T) * logit_scale
                    probs_arr = torch.softmax(logits, dim=-1).cpu().numpy()

                for row_i, item_idx in enumerate(plastic_indices):
                    row = probs_arr[row_i]
                    order = np.argsort(-row)
                    top_rid = resin_ids[int(order[0])]
                    top_prob = float(row[int(order[0])])
                    second_prob = float(row[int(order[1])]) if len(order) > 1 else 0.0
                    clip_probs_by_pos[item_idx] = (top_rid, top_prob, top_prob - second_prob)
            except Exception as exc:
                logger.warning("CLIP resin code scoring failed (non-fatal): %s", exc)

        for item_idx in plastic_indices:
            img, _ = items[item_idx]
            txt = extracted_texts.get(item_idx, "")
            matched_ric: Optional[int] = None
            method = "cv_roi_only"
            conf = 0.0

            # OCR Regex match takes direct precedence when present
            if txt:
                for rid, pattern in _RIC_REGEX_PATTERNS:
                    if pattern.search(txt):
                        matched_ric = rid
                        conf = 0.92
                        method = "ocr_regex"
                        break

            # Otherwise use CLIP zero-shot polymer subtype if margin & confidence clear threshold
            if matched_ric is None and item_idx in clip_probs_by_pos:
                top_rid, top_prob, margin = clip_probs_by_pos[item_idx]
                if top_prob >= 0.30 and margin >= 0.04:
                    matched_ric = top_rid
                    conf = round(top_prob, 4)
                    method = "clip_roi"
                else:
                    conf = round(top_prob, 4)
                    method = "clip_uncertain"

            has_stamp_roi = _has_triangular_or_label_roi(img)
            hazard_flags: List[str] = []
            if txt:
                low_t = txt.lower()
                for kw, label in _HAZARD_TEXT_KEYWORDS.items():
                    if kw in low_t:
                        hazard_flags.append(label)

            if matched_ric is not None and matched_ric in RESIN_CATALOG:
                meta = RESIN_CATALOG[matched_ric]
                outputs[item_idx] = {
                    "resin_code": meta["code"],
                    "resin_number": matched_ric,
                    "resin_name": meta["name"],
                    "recyclability_tier": meta["tier"],
                    "confidence": conf,
                    "detected_text": txt or None,
                    "hazard_flags": hazard_flags,
                    "method": f"{method}+cv_roi" if has_stamp_roi and method == "clip_roi" else method,
                    "guidance": meta["guidance"],
                }
            else:
                outputs[item_idx] = {
                    "resin_code": "Unverified Plastic",
                    "resin_number": None,
                    "resin_name": "Plastic Polymer (Stamp Unclear)",
                    "recyclability_tier": "Verify Resin Triangle Stamp",
                    "confidence": conf,
                    "detected_text": txt or None,
                    "hazard_flags": hazard_flags,
                    "method": method,
                    "guidance": "Check the embossed 1-7 recycling triangle underneath before binning.",
                }

    # 2. Metal / Trash Hazard Cue Screening (Battery / Aerosol / Keyword OCR)
    if hazard_indices:
        clip_hazard_by_pos: Dict[int, Tuple[str, float, float]] = {}
        if can_use_clip:
            try:
                labels, h_centroids = _get_hazard_centroids(clip_model, clip_processor)
                device = getattr(clip_model, "device", torch.device("cpu"))
                logit_scale = (
                    clip_model.logit_scale.exp().clamp(max=100.0)
                    if hasattr(clip_model, "logit_scale")
                    else torch.tensor(100.0, device=device)
                )
                rgb_imgs = [items[idx][0].convert("RGB") for idx in hazard_indices]
                img_inputs = clip_processor(images=rgb_imgs, return_tensors="pt")
                img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
                with torch.no_grad():
                    img_feats = clip_model.get_image_features(**img_inputs)
                    if hasattr(img_feats, "pooler_output"):
                        img_feats = img_feats.pooler_output
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    logits = (img_feats @ h_centroids.T) * logit_scale
                    probs_arr = torch.softmax(logits, dim=-1).cpu().numpy()

                for row_i, item_idx in enumerate(hazard_indices):
                    row = probs_arr[row_i]
                    benign_prob = float(row[0])
                    best_h_idx = 1 + int(np.argmax(row[1:])) if len(row) > 1 else 0
                    best_h_prob = float(row[best_h_idx])
                    clip_hazard_by_pos[item_idx] = (
                        labels[best_h_idx],
                        best_h_prob,
                        best_h_prob - benign_prob,
                    )
            except Exception as exc:
                logger.warning("CLIP hazard screening failed (non-fatal): %s", exc)

        for item_idx in hazard_indices:
            txt = extracted_texts.get(item_idx, "")
            flags: List[str] = []
            conf = 0.0
            method = "ocr_keyword"

            if txt:
                low_t = txt.lower()
                for kw, label in _HAZARD_TEXT_KEYWORDS.items():
                    if kw in low_t and label not in flags:
                        flags.append(label)
                        conf = max(conf, 0.88)

            if item_idx in clip_hazard_by_pos:
                h_key, h_prob, margin = clip_hazard_by_pos[item_idx]
                if h_prob >= 0.65 and margin >= 0.25:
                    label = (
                        "Possible Battery / E-Waste Hazard"
                        if h_key == "battery_ewaste"
                        else "Possible Pressurized / Chemical Hazard"
                    )
                    if label not in flags:
                        flags.append(label)
                    conf = max(conf, round(h_prob, 4))
                    method = "clip_hazard_screening"

            if flags or txt:
                outputs[item_idx] = {
                    "resin_code": "Hazard Screening" if flags else None,
                    "resin_number": None,
                    "resin_name": flags[0] if flags else None,
                    "recyclability_tier": "Special Hazardous / E-Waste Drop-Off" if flags else None,
                    "confidence": round(conf, 4),
                    "detected_text": txt or None,
                    "hazard_flags": flags,
                    "method": method,
                    "guidance": (
                        "Do not crush or compact. Route batteries, electronics, or pressurized canisters to hazardous waste collection."
                        if flags
                        else None
                    ),
                }

    return outputs
