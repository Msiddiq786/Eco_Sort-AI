"""
Multi-Object Waste Detection and Localization Engine for EcoSort AI V3.

Architecture:
1. CWD-Independent Instance Localization ("WHERE objects are"):
   - Dedicated fine-tuned YOLOv8 weights at backend/models/detection_v2/weights/best.pt (when trained).
   - Otherwise, Hybrid Multi-Stage Instance Localizer combining:
     (a) Waste-trained YOLOv8 instance segmentation (5 trash super-classes: Glass, Metal, Paper, Plastic, Waste)
     (b) Pretrained COCO YOLOv8 detector (imgsz=1024) for upright bottles, organic produce, cups, bowls, boxes
     (c) Residual CLAHE + LAB contrast proposals strictly on uncovered pixels (translucent containers)
     (d) Multi-scale morphological + valley-splitting fallback for non-photo synthetic test arrays
2. Non-Maximum & Multi-Object Containment Suppression:
   - Never merges two or more already-localized distinct instances into an oversized bounding box.
   - Separates overlapping organic produce (e.g., lettuce beside banana peel) while suppressing inner sub-features.
   - Prevents image-border shifted duplicate boxes and tiny shadow sliver false positives.
3. Aspect-Preserving Crop Classification ("WHAT MATERIAL each object belongs to"):
   - Tightly crops (2% margin, clipped to bounds) and pads elongated crops (>1.35 aspect ratio)
     to square with neutral border median fill so ViT/CLIP never squash tall bottles or cans.
   - Independently runs ViT + CLIP V2 per crop and records Detection, ViT, CLIP, and Ensemble scores.
   - Uses balanced crop ensemble fusion (48% ViT + 52% CLIP when both active) plus strict organic bypass
     and honest disagreement/OOD handling.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.ml.clip_model import predict_clip
from app.ml.clip_model_v2 import predict_clip_v2, predict_clip_v2_batch
from app.ml.ensemble import fuse
from app.ml.resin_ocr import analyze_resin_and_hazards_batch
from app.ml.vit_model import predict_vit, predict_vit_batch

_ORIG_PREDICT_VIT = predict_vit
_ORIG_PREDICT_VIT_BATCH = predict_vit_batch
_ORIG_PREDICT_CLIP_V2 = predict_clip_v2
_ORIG_PREDICT_CLIP_V2_BATCH = predict_clip_v2_batch
from app.schemas.detect import (
    BoundingBox,
    DetectedObject,
    DetectResponse,
    ModelCoverage,
    RecyclingStatusCounts,
    RecyclingSummary,
)
from app.schemas.predict import RecyclingDecisionSchema, ResinInfoSchema
from app.services.recycling_engine import DisposalStatus, get_recycling_decision

logger = logging.getLogger(__name__)

# CWD-independent directory roots
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _BACKEND_DIR.parent

# Configured paths for V2/V3 detection models
DETECTOR_CONFIG_PATH = _BACKEND_DIR / "models" / "detection_v2" / "config.yaml"
DETECTOR_WEIGHTS_PATH = _BACKEND_DIR / "models" / "detection_v2" / "weights" / "best.pt"
TRASH_SEG_LOCAL_PATH = _BACKEND_DIR / "models" / "detection_v2" / "weights" / "yolov8m_trash_seg.pt"

# Module-level YOLO checkpoint cache to avoid reloading across requests or test calls
_YOLO_MODEL_CACHE: Dict[str, Any] = {}

COCO_ALLOWED_CLASSES = {
    # Liquid & Food Containers
    "bottle", "cup", "wine glass", "bowl", "vase",
    # Organic Food Waste
    "banana", "apple", "orange", "broccoli", "carrot", "sandwich", "pizza", "hot dog", "donut", "cake",
    # Packaging / Boxes / Bags
    "book", "suitcase", "handbag", "backpack",
    # Utensils & Objects
    "fork", "knife", "spoon", "scissors",
}

COCO_ORGANIC_CLASSES = {
    "banana", "apple", "orange", "broccoli", "carrot", "sandwich", "pizza", "hot dog", "donut", "cake",
}

# Category colors for bounding box visualization
CATEGORY_HEX: Dict[str, str] = {
    "cardboard": "#D4A54A",
    "glass": "#0EA5E9",
    "metal": "#64748B",
    "paper": "#10B981",
    "plastic": "#F97316",
    "organic": "#84CC16",
    "trash": "#6B7280",
    "unknown": "#94A3B8",
}

CATEGORY_RGB: Dict[str, Tuple[int, int, int]] = {
    "cardboard": (212, 165, 74),
    "glass": (14, 165, 233),
    "metal": (100, 116, 139),
    "paper": (16, 185, 129),
    "plastic": (249, 115, 22),
    "organic": (132, 204, 22),
    "trash": (107, 114, 128),
    "unknown": (148, 163, 184),
}


def _load_yolo_cached(path: Path) -> Optional[Any]:
    """Load and cache an Ultralytics YOLO checkpoint safely."""
    key = str(path.resolve())
    if key in _YOLO_MODEL_CACHE:
        return _YOLO_MODEL_CACHE[key]
    if not path.exists() or not path.is_file() or path.stat().st_size < 100_000:
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(str(path))
        _YOLO_MODEL_CACHE[key] = model
        return model
    except Exception as exc:
        logger.warning("Failed loading YOLO checkpoint %s: %s", path, exc)
        return None


def _find_cached_trash_seg_path() -> Optional[Path]:
    """Locate the waste-trained YOLOv8 segmentation weights cleanly regardless of CWD."""
    if TRASH_SEG_LOCAL_PATH.exists() and TRASH_SEG_LOCAL_PATH.stat().st_size > 1_000_000:
        return TRASH_SEG_LOCAL_PATH

    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--turhancan97--yolov8-segment-trash-detection"
    if hub_dir.exists():
        for pt in hub_dir.rglob("*.pt"):
            if ".no_exist" in str(pt) or pt.stat().st_size < 1_000_000:
                continue
            model = _load_yolo_cached(pt)
            if model is not None and len(getattr(model, "names", {})) == 5:
                return pt

    is_offline = (
        os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
        or os.getenv("TRANSFORMERS_OFFLINE", "").lower() in {"1", "true", "yes"}
    )
    if not is_offline:
        try:
            from app.core.config import settings
            if getattr(settings, "AUTO_DOWNLOAD_MODELS", True) and getattr(settings, "TRASH_SEG_HF_REPO_ID", None):
                from huggingface_hub import snapshot_download
                dl_dir = snapshot_download(
                    repo_id=settings.TRASH_SEG_HF_REPO_ID,
                    allow_patterns=["*.pt"],
                    token=getattr(settings, "HF_TOKEN", None),
                )
                for pt in Path(dl_dir).rglob("*.pt"):
                    if pt.stat().st_size > 1_000_000:
                        model = _load_yolo_cached(pt)
                        if model is not None:
                            return pt
        except Exception as exc:
            logger.debug("Optional trash-seg HF hub download skipped/failed: %s", exc)
    return None


def _find_coco_yolo_path() -> Optional[Path]:
    """Locate yolov8s.pt or yolov8n.pt regardless of CWD, with optional online download."""
    from app.core.config import settings

    candidates = [
        _REPO_ROOT / "yolov8s.pt",
        _BACKEND_DIR / "yolov8s.pt",
        settings.MODELS_DIR / "yolov8s.pt",
        Path("yolov8s.pt"),
        _REPO_ROOT / "yolov8n.pt",
        _BACKEND_DIR / "yolov8n.pt",
        settings.MODELS_DIR / "yolov8n.pt",
        Path("yolov8n.pt"),
    ]
    for p in candidates:
        if p.exists() and p.is_file() and p.stat().st_size > 1_000_000:
            return p

    is_offline = (
        os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
        or os.getenv("TRANSFORMERS_OFFLINE", "").lower() in {"1", "true", "yes"}
    )
    if not is_offline and getattr(settings, "AUTO_DOWNLOAD_MODELS", True):
        try:
            from ultralytics import YOLO

            target = settings.MODELS_DIR / "yolov8s.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            model = YOLO(str(target))
            if target.exists() and target.stat().st_size > 1_000_000:
                key = str(target.resolve())
                _YOLO_MODEL_CACHE[key] = model
                return target
        except Exception as exc:
            logger.debug("Optional COCO YOLO download skipped/failed: %s", exc)
    return None


def calculate_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union between two boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0.0:
        return 0.0
    return interArea / denom


def calculate_iomin(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Minimum (containment ratio) between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    min_area = min(boxAArea, boxBArea)
    if min_area <= 0.0:
        return 0.0
    return interArea / min_area


def _is_deep_label(label: str) -> bool:
    return label.startswith("seg_") or label.startswith("coco_") or label.startswith("yolo_") or label.startswith("det_")


def merge_can_fragments(
    proposals: List[Tuple[Tuple[float, float, float, float], float, str]],
    max_v_gap: float = 40.0,
) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
    """
    Merge vertically split fragments belonging to classical morphological proposals.
    Deep detector instances (seg_/coco_/yolo_/det_) are NEVER merged here so stacked
    objects (e.g., bottle above another bottle or can) remain distinct.
    """
    merged: List[Tuple[Tuple[float, float, float, float], float, str]] = []
    used = set()
    n = len(proposals)
    for i in range(n):
        if i in used:
            continue
        boxA, scoreA, labelA = proposals[i]
        if _is_deep_label(labelA) or labelA.startswith("residual_"):
            merged.append((boxA, scoreA, labelA))
            used.add(i)
            continue

        merged_box = list(boxA)
        merged_score = scoreA
        merged_label = labelA
        wA = merged_box[2] - merged_box[0]
        hA = merged_box[3] - merged_box[1]

        if wA <= 220 and hA <= 240:
            for j in range(i + 1, n):
                if j in used:
                    continue
                boxB, scoreB, labelB = proposals[j]
                if _is_deep_label(labelB) or labelB.startswith("residual_"):
                    continue
                wB = boxB[2] - boxB[0]
                hB = boxB[3] - boxB[1]
                if wB <= 220 and hB <= 240:
                    xA = max(merged_box[0], boxB[0])
                    xB = min(merged_box[2], boxB[2])
                    inter_x = max(0.0, xB - xA)
                    min_w = min(wA, wB)
                    if min_w > 0 and (inter_x / min_w) > 0.65:
                        if merged_box[3] <= boxB[1] and (boxB[1] - merged_box[3]) <= max_v_gap:
                            merged_box = [min(merged_box[0], boxB[0]), merged_box[1], max(merged_box[2], boxB[2]), boxB[3]]
                            merged_score = max(merged_score, scoreB)
                            used.add(j)
                        elif boxB[3] <= merged_box[1] and (merged_box[1] - boxB[3]) <= max_v_gap:
                            merged_box = [min(merged_box[0], boxB[0]), boxB[1], max(merged_box[2], boxB[2]), merged_box[3]]
                            merged_score = max(merged_score, scoreB)
                            used.add(j)
        merged.append(((float(merged_box[0]), float(merged_box[1]), float(merged_box[2]), float(merged_box[3])), merged_score, merged_label))
        used.add(i)
    return merged


def apply_containment_and_nms(
    proposals: List[Tuple[Tuple[float, float, float, float], float, str]],
    containment_threshold: float = 0.68,
    iou_threshold: float = 0.42,
    max_area_ratio: float = 0.65,
    total_area: float = 1.0,
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
    """
    Instance-safe multi-object containment suppression and Non-Maximum Suppression.

    Safeguards:
    1. Rejects oversized boxes exceeding max_area_ratio.
    2. NEVER merges or deletes >= 2 distinct kept boxes when a larger enclosing proposal arrives.
    3. Protects distinct organic classes (e.g. lettuce beside banana) from over-eager nested suppression.
    4. Never lets a lower-priority classical proposal supersede a deep detector box.
    5. Suppresses shifted edge-ghost duplicate bottles touching image borders.
    """
    if not proposals:
        return []

    def sort_key(p: Tuple[Tuple[float, float, float, float], float, str]) -> float:
        box, score, label = p
        if label.startswith("det_"):
            prio = 2.2
        elif label.startswith("seg_"):
            prio = 2.0
        elif label.startswith("coco_") or label.startswith("yolo_"):
            prio = 1.85
        else:
            prio = 0.75
        return prio * score

    sorted_props = sorted(proposals, key=sort_key, reverse=True)
    kept: List[Tuple[Tuple[float, float, float, float], float, str]] = []

    for boxA, scoreA, labelA in sorted_props:
        areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        if areaA <= 0.0:
            continue
        if total_area > 0 and areaA > total_area * max_area_ratio:
            continue

        is_deepA = _is_deep_label(labelA)
        should_add = True
        contained_kept: List[Tuple[int, str, float]] = []

        for idx, (boxK, scoreK, labelK) in enumerate(kept):
            areaK = max(0.0, (boxK[2] - boxK[0]) * (boxK[3] - boxK[1]))
            if areaK <= 0.0:
                continue

            xA = max(boxA[0], boxK[0])
            yA = max(boxA[1], boxK[1])
            xB = min(boxA[2], boxK[2])
            yB = min(boxA[3], boxK[3])
            inter = max(0.0, xB - xA) * max(0.0, yB - yA)
            if inter <= 0.0:
                continue

            union = areaA + areaK - inter
            iou = (inter / union) if union > 0 else 0.0

            # Preserve distinct overlapping produce items (e.g. banana & broccoli/lettuce)
            clsA = labelA.split("_", 1)[-1]
            clsK = labelK.split("_", 1)[-1]
            distinct_organic = (
                clsA in COCO_ORGANIC_CLASSES
                and clsK in COCO_ORGANIC_CLASSES
                and clsA != clsK
            )

            eff_iou_thr = 0.55 if distinct_organic else iou_threshold
            if iou >= eff_iou_thr:
                should_add = False
                break

            # Detect edge-ghost duplicate bottles/cups on image perimeter
            if (
                image_width is not None
                and labelA == labelK
                and labelA in ("coco_bottle", "coco_cup", "yolo_bottle")
            ):
                touches_edge = boxA[0] <= 4.0 or boxA[2] >= image_width - 4.0
                if touches_edge and (inter / areaA) >= 0.45 and scoreA < scoreK * 0.85:
                    should_add = False
                    break

            if areaA <= areaK:
                cont = inter / areaA
                eff_cont_thr = 0.85 if distinct_organic else containment_threshold
                if cont >= eff_cont_thr:
                    should_add = False
                    break
            else:
                cont_k_in_a = inter / areaK
                if cont_k_in_a >= 0.50:
                    contained_kept.append((idx, labelK, cont_k_in_a))

        if not should_add:
            continue

        # CRITICAL: Reject enclosing box if it merges >= 2 separate kept objects
        if len(contained_kept) >= 2:
            continue

        if contained_kept:
            idxK, labelK, cov = contained_kept[0]
            # Classical residual/morph proposals must never overwrite deep detector objects
            if not is_deepA and _is_deep_label(labelK):
                continue
            areaK = (kept[idxK][0][2] - kept[idxK][0][0]) * (kept[idxK][0][3] - kept[idxK][0][1])
            # Protect full deep objects (>40% of candidate area) from being eaten by slightly larger boxes
            if _is_deep_label(labelK) and areaK > 0.40 * areaA and cov < 0.85:
                continue
            if cov >= containment_threshold:
                kept.pop(idxK)

        kept.append((boxA, scoreA, labelA))

    return kept


def apply_non_maximum_suppression(
    proposals: List[Tuple[Tuple[float, float, float, float], float, str]],
    iou_threshold: float = 0.38,
) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
    """Sort proposals by detection confidence and filter out overlapping boxes (backward compatible)."""
    if not proposals:
        return []

    sorted_props = sorted(proposals, key=lambda x: x[1], reverse=True)
    kept: List[Tuple[Tuple[float, float, float, float], float, str]] = []

    for box, score, label in sorted_props:
        overlap = False
        for kept_box, _, _ in kept:
            if calculate_iou(box, kept_box) > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append((box, score, label))

    return kept


def _prepare_crop_for_classification(crop_img: Image.Image) -> Image.Image:
    """
    Preserve aspect ratio on elongated waste objects (e.g. tall bottles or cans)
    by padding with median border background color before ViT / CLIP resizing.
    Near-square crops (ratio <= 1.35) are left untouched to avoid adding artificial margins.
    """
    cw, ch = crop_img.size
    if cw <= 0 or ch <= 0:
        return crop_img
    aspect_ratio = max(cw / float(ch), ch / float(cw))
    if aspect_ratio <= 1.35:
        return crop_img

    side = max(cw, ch)
    rgb_arr = np.array(crop_img.convert("RGB"))
    edge_pixels = np.vstack([
        rgb_arr[0, :],
        rgb_arr[-1, :],
        rgb_arr[:, 0],
        rgb_arr[:, -1],
    ])
    bg_color = tuple(int(v) for v in np.median(edge_pixels, axis=0))
    return ImageOps.pad(crop_img, (side, side), color=bg_color, centering=(0.5, 0.5))


class WasteObjectDetector:
    """
    Dedicated Multi-Object Waste Detector.

    Priority chain:
    1. Dedicated 7-class fine-tuned weights (backend/models/detection_v2/weights/best.pt) when present.
    2. Hybrid Waste-Seg (YOLOv8m trash segmentation) + COCO YOLOv8s (bottles, organic produce, cups/bowls)
       + Residual LAB/CLAHE translucent container proposals on uncovered regions.
    3. Multi-scale morphological + contrast fallback when no deep weights exist or scene has synthetic shapes.
    """

    def __init__(self, weights_path: Optional[Path] = None):
        self.weights_path = weights_path or DETECTOR_WEIGHTS_PATH
        self.yolo_model = None
        self.trash_seg_model = None
        self.coco_model = None
        self.model_status = "awaiting_training"
        self._initialize_detector()

    def _initialize_detector(self) -> None:
        """Load dedicated weights if trained, else hybrid waste-seg + COCO instance localizers."""
        if self.weights_path.exists() and self.weights_path.is_file():
            loaded = _load_yolo_cached(self.weights_path)
            if loaded is not None:
                self.yolo_model = loaded
                self.model_status = "trained_weights_ready"
                logger.info("Loaded dedicated trained waste detector from %s", self.weights_path)
                return

        seg_path = _find_cached_trash_seg_path()
        coco_path = _find_coco_yolo_path()

        if seg_path is not None:
            self.trash_seg_model = _load_yolo_cached(seg_path)
        if coco_path is not None:
            self.coco_model = _load_yolo_cached(coco_path)
            self.yolo_model = self.coco_model

        if self.trash_seg_model is not None and self.coco_model is not None:
            self.model_status = "hybrid_waste_seg_and_coco"
            logger.info("Initialized hybrid localizer (trash_seg=%s, coco=%s)", seg_path.name, coco_path.name)
        elif self.trash_seg_model is not None:
            self.model_status = "pretrained_waste_seg"
            logger.info("Initialized waste-seg localizer (%s)", seg_path.name)
        elif self.coco_model is not None:
            self.model_status = f"pretrained_hybrid_{coco_path.stem}"
            logger.info("Initialized COCO hybrid localizer (%s)", coco_path.name)
        else:
            self.model_status = "fallback_multiscale_proposal"
            logger.info("No YOLO weights available. Using multi-scale proposal engine.")

    def _residual_translucent_proposals(
        self,
        pil_image: Image.Image,
        mag: np.ndarray,
        max_m: float,
        kept_deep: List[Tuple[Tuple[float, float, float, float], float, str]],
        total_area: float,
    ) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
        """
        Detect low-contrast translucent plastic trays/bowls only in image regions not covered
        by deep detector instances, strictly rejecting shadow strips along box borders.
        """
        w, h = pil_image.size
        rgb = np.array(pil_image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(lab[:, :, 0].astype(np.uint8))

        gx = cv2.Sobel(l_eq, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l_eq, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)
        grad_norm = grad / (float(np.percentile(grad, 99.5)) + 1e-5)

        borders = np.vstack([
            lab[:15, :].reshape(-1, 3),
            lab[-15:, :].reshape(-1, 3),
            lab[:, :15].reshape(-1, 3),
            lab[:, -15:].reshape(-1, 3),
        ])
        bg = np.median(borders, axis=0)
        lab_diff = np.linalg.norm(lab - bg, axis=2)
        diff_norm = lab_diff / (float(np.percentile(lab_diff, 99.0)) + 1e-5)
        saliency = np.clip(0.55 * diff_norm + 0.45 * grad_norm, 0.0, 1.0)

        occ = np.zeros((h, w), dtype=np.uint8)
        for box, _, _ in kept_deep:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            occ[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1

        min_res_area = total_area * 0.012
        max_res_area = total_area * 0.25
        proposals: List[Tuple[Tuple[float, float, float, float], float, str]] = []

        for thr in (0.17, 0.25):
            bin_mask = ((saliency > thr) & (occ == 0)).astype(np.uint8) * 255
            k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k_morph)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_morph)
            cleaned = ndi.binary_fill_holes(cleaned > 0).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                x, y, bw, bh = cv2.boundingRect(c)
                area = float(bw * bh)
                if area < min_res_area or area > max_res_area or min(bw, bh) < 42:
                    continue
                if max(bw / float(bh), bh / float(bw)) > 3.5:
                    continue
                contour_area = float(cv2.contourArea(c))
                if contour_area / area < 0.40:
                    continue
                if (x <= 4 or x + bw >= w - 4) and bw < 45 and bh > h * 0.5:
                    continue
                mean_g = float(np.mean(grad_norm[y:y + bh, x:x + bw]))
                if mean_g < 0.05:
                    continue
                score = float(np.clip(0.35 + 0.50 * float(np.mean(saliency[y:y + bh, x:x + bw])), 0.20, 0.72))
                proposals.append(((float(x), float(y), float(x + bw), float(y + bh)), score, "residual_sal"))

        return proposals

    def _classical_fallback_proposals(
        self,
        pil_image: Image.Image,
        mag: np.ndarray,
        max_m: float,
        total_area: float,
    ) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
        """Multi-scale Sobel morphology + contrast proposals for synthetic/unseen images when deep detectors fire 0."""
        w, h = pil_image.size
        min_box_area = total_area * 0.004
        max_box_area = total_area * 0.38
        proposals: List[Tuple[Tuple[float, float, float, float], float, str]] = []

        for p_ratio in (0.15, 0.28, 0.42):
            edge_bin = mag > (max_m * p_ratio)
            filled = ndi.binary_fill_holes(edge_bin)
            clean = ndi.binary_opening(filled, structure=np.ones((3, 3)))
            clean = ndi.binary_closing(clean, structure=np.ones((3, 3)))
            labeled, _ = ndi.label(clean)
            slices = ndi.find_objects(labeled)

            for s in slices:
                if s is None:
                    continue
                y1, y2 = s[0].start, s[0].stop
                x1, x2 = s[1].start, s[1].stop
                bw = x2 - x1
                bh = y2 - y1
                area = bw * bh
                if area < min_box_area or min(bw, bh) < 14:
                    continue
                if (x1 <= 4 and x2 <= 40 and bh > h * 0.7) or (x1 >= w - 40 and x2 >= w - 4 and bh > h * 0.7):
                    continue

                if area <= max_box_area:
                    score = float(np.mean(mag[y1:y2, x1:x2])) / (max_m + 1e-5)
                    area_ratio = area / total_area
                    boosted = 0.50 + 0.35 * score + 0.15 * min(1.0, area_ratio / 0.10)
                    proposals.append(((float(x1), float(y1), float(x2), float(y2)), boosted, "morph_obj"))
                else:
                    sub = edge_bin[y1:y2, x1:x2]
                    v_p = sub.sum(axis=0)
                    h_p = sub.sum(axis=1)
                    if bw >= bh and bw > 50:
                        ms, me = int(bw * 0.2), int(bw * 0.8)
                        mid = v_p[ms:me]
                        if len(mid) > 0 and float(mid.min()) < 0.60 * float(v_p.max()):
                            sx = ms + int(np.argmin(mid))
                            for b in [(float(x1), float(y1), float(x1 + sx), float(y2)), (float(x1 + sx), float(y1), float(x2), float(y2))]:
                                ca = (b[2] - b[0]) * (b[3] - b[1])
                                if min_box_area <= ca <= max_box_area:
                                    proposals.append((b, 0.85, "morph_split"))
                    elif bh > bw and bh > 50:
                        ms, me = int(bh * 0.2), int(bh * 0.8)
                        mid = h_p[ms:me]
                        if len(mid) > 0 and float(mid.min()) < 0.60 * float(h_p.max()):
                            sy = ms + int(np.argmin(mid))
                            for b in [(float(x1), float(y1), float(x2), float(y1 + sy)), (float(x1), float(y1 + sy), float(x2), float(y2))]:
                                ca = (b[2] - b[0]) * (b[3] - b[1])
                                if min_box_area <= ca <= max_box_area:
                                    proposals.append((b, 0.85, "morph_split"))

        cv_img = np.array(pil_image.convert("RGB"))
        borders = np.vstack([
            cv_img[:15, :].reshape(-1, 3),
            cv_img[-15:, :].reshape(-1, 3),
            cv_img[:, :15].reshape(-1, 3),
            cv_img[:, -15:].reshape(-1, 3),
        ])
        bg_color = np.median(borders, axis=0)
        diff = np.max(np.abs(cv_img.astype(float) - bg_color), axis=2)
        max_diff = diff.max() + 1e-5
        saliency = 0.6 * (diff / max_diff) + 0.4 * (mag / (max_m + 1e-5))

        for thresh in (0.28, 0.42):
            bin_mask = (saliency > thresh).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, bw, bh = cv2.boundingRect(c)
                area = float(bw * bh)
                if min_box_area <= area <= max_box_area:
                    score = float(np.mean(saliency[y:y + bh, x:x + bw]))
                    proposals.append(((float(x), float(y), float(x + bw), float(y + bh)), score, "contrast_obj"))

        return merge_can_fragments(proposals, max_v_gap=40.0)

    def detect_objects(
        self,
        pil_image: Image.Image,
        conf_threshold: float = 0.25,
        max_detections: int = 30,
    ) -> List[Tuple[BoundingBox, float, str]]:
        """
        Detect candidate waste objects and generate tight instance bounding boxes.
        Never fabricates detections; blank/uniform images return [].
        """
        w, h = pil_image.size
        total_area = float(w * h)

        gray = pil_image.convert("L")
        arr_gray = np.array(gray, dtype=np.float32)
        gx = ndi.sobel(arr_gray, axis=1)
        gy = ndi.sobel(arr_gray, axis=0)
        mag = np.hypot(gx, gy)
        max_m = float(mag.max())

        if max_m < 5.0 or float(np.std(arr_gray)) < 5.0:
            return []

        min_box_area = total_area * 0.003
        deep_max_area = total_area * 0.65
        deep_proposals: List[Tuple[Tuple[float, float, float, float], float, str]] = []

        # 1. Dedicated fine-tuned 7-class detector (when trained)
        if self.model_status == "trained_weights_ready" and self.yolo_model is not None:
            try:
                results = self.yolo_model(pil_image, imgsz=960, conf=conf_threshold, iou=0.45, verbose=False)
                for r in results:
                    for b in r.boxes:
                        cid = int(b.cls[0].cpu().numpy())
                        name = r.names.get(cid, f"obj_{cid}")
                        coords = [float(c) for c in b.xyxy[0].cpu().numpy()]
                        conf = float(b.conf[0].cpu().numpy())
                        area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                        if min_box_area <= area <= deep_max_area:
                            deep_proposals.append(((coords[0], coords[1], coords[2], coords[3]), min(0.99, conf), f"det_{name}"))
            except Exception as exc:
                logger.error("Dedicated detector inference failed: %s", exc)
        else:
            # 2a. Waste-trained YOLOv8 instance segmentation
            if self.trash_seg_model is not None:
                try:
                    results = self.trash_seg_model(
                        pil_image, imgsz=960, conf=0.20, iou=0.45, agnostic_nms=True, verbose=False
                    )
                    for r in results:
                        for b in r.boxes:
                            cid = int(b.cls[0].cpu().numpy())
                            name = r.names.get(cid, f"obj_{cid}")
                            coords = [float(c) for c in b.xyxy[0].cpu().numpy()]
                            conf = float(b.conf[0].cpu().numpy())
                            area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                            if min_box_area <= area <= deep_max_area:
                                deep_proposals.append(((coords[0], coords[1], coords[2], coords[3]), min(0.99, conf), f"seg_{name}"))
                except Exception as exc:
                    logger.error("Waste segmentation proposal inference failed: %s", exc)

            # 2b. Pretrained COCO YOLOv8 for bottles, produce, cans/cups, bowls
            if self.coco_model is not None:
                try:
                    results = self.coco_model(
                        pil_image, imgsz=1024, conf=0.20, iou=0.45, agnostic_nms=True, verbose=False
                    )
                    for r in results:
                        for b in r.boxes:
                            cid = int(b.cls[0].cpu().numpy())
                            name = r.names.get(cid, f"obj_{cid}")
                            if name not in COCO_ALLOWED_CLASSES:
                                continue
                            coords = [float(c) for c in b.xyxy[0].cpu().numpy()]
                            conf = float(b.conf[0].cpu().numpy())
                            area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                            if min_box_area <= area <= deep_max_area:
                                boost = 1.05 if name in ({"bottle", "wine glass"} | COCO_ORGANIC_CLASSES) else 0.90
                                deep_proposals.append(((coords[0], coords[1], coords[2], coords[3]), min(0.99, conf * boost), f"coco_{name}"))
                except Exception as exc:
                    logger.error("COCO proposal inference failed: %s", exc)

        if deep_proposals:
            kept_deep = apply_containment_and_nms(
                deep_proposals,
                containment_threshold=0.68,
                iou_threshold=0.42,
                max_area_ratio=0.65,
                total_area=total_area,
                image_width=float(w),
                image_height=float(h),
            )
            residual_props = self._residual_translucent_proposals(pil_image, mag, max_m, kept_deep, total_area)
            suppressed_proposals = apply_containment_and_nms(
                kept_deep + residual_props,
                containment_threshold=0.68,
                iou_threshold=0.42,
                max_area_ratio=0.65,
                total_area=total_area,
                image_width=float(w),
                image_height=float(h),
            )
        else:
            # Classical fallback for synthetic shapes or weightless fallback
            classical = self._classical_fallback_proposals(pil_image, mag, max_m, total_area)
            suppressed_proposals = apply_containment_and_nms(
                classical,
                containment_threshold=0.68,
                iou_threshold=0.40,
                max_area_ratio=0.38,
                total_area=total_area,
                image_width=float(w),
                image_height=float(h),
            )

        validated_boxes: List[Tuple[BoundingBox, float, str]] = []
        for box, score, label in suppressed_proposals[:max_detections]:
            x1 = max(0.0, min(float(w), round(box[0], 1)))
            y1 = max(0.0, min(float(h), round(box[1], 1)))
            x2 = max(x1 + 10.0, min(float(w), round(box[2], 1)))
            y2 = max(y1 + 10.0, min(float(h), round(box[3], 1)))
            x2 = min(float(w), x2)
            y2 = min(float(h), y2)
            if x2 <= x1 or y2 <= y1:
                continue

            box_area = (x2 - x1) * (y2 - y1)
            effective_max_area = total_area * 0.65 if _is_deep_label(label) else total_area * 0.38
            if box_area > effective_max_area or box_area < min_box_area:
                continue

            validated_boxes.append((
                BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    nx1=round(x1 / w, 6),
                    ny1=round(y1 / h, 6),
                    nx2=round(x2 / w, 6),
                    ny2=round(y2 / h, 6),
                ),
                round(min(0.99, max(0.10, score)), 3),
                label,
            ))

        return validated_boxes


_detector_instance: Optional[WasteObjectDetector] = None


def get_detector() -> WasteObjectDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = WasteObjectDetector()
    return _detector_instance


def extract_candidate_regions(pil_image: Image.Image) -> List[Tuple[BoundingBox, float, str]]:
    """Convenience wrapper for candidate region extraction."""
    return get_detector().detect_objects(pil_image)


def render_annotated_image(
    pil_image: Image.Image,
    detected_objects: List[DetectedObject],
) -> str:
    """
    Render category-colored bounding boxes, object ID, and confidence pills on the image.
    Returns a Base64-encoded Data URL string.
    """
    annotated = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for obj in detected_objects:
        cat = obj.material_category.lower()
        rgb = CATEGORY_RGB.get(cat, (107, 114, 128))
        box = obj.bbox

        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=rgb, width=3)

        label_text = f"#{obj.object_id} {obj.material_category.capitalize()} ({obj.calibrated_confidence*100:.0f}%)"
        text_bbox = draw.textbbox((box.x1, box.y1), label_text, font=font) if font else (box.x1, box.y1, box.x1 + 100, box.y1 + 18)
        text_w = text_bbox[2] - text_bbox[0] + 10
        text_h = text_bbox[3] - text_bbox[1] + 6

        pill_y1 = box.y1 - text_h - 2 if box.y1 >= text_h + 2 else box.y1 + 2
        pill_y2 = pill_y1 + text_h
        pill_x1 = min(box.x1, max(0.0, float(annotated.width) - text_w))
        pill_x2 = min(float(annotated.width), pill_x1 + text_w)

        draw.rectangle([pill_x1, pill_y1, pill_x2, pill_y2], fill=rgb)
        draw.text((pill_x1 + 5, pill_y1 + 3), label_text, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=88)
    base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"


def detect_and_classify_objects(
    pil_image: Image.Image,
    vit_model: Any,
    vit_processor: Any,
    clip_model: Optional[Any] = None,
    clip_processor: Optional[Any] = None,
) -> DetectResponse:
    """
    Full Multi-Object Waste Detection & Independent Crop-Level Classification pipeline.
    Detector localizes WHERE objects are; ViT + CLIP determine WHAT MATERIAL each object is.
    """
    t0 = time.perf_counter()
    w, h = pil_image.size

    detector = get_detector()
    candidates = detector.detect_objects(pil_image)

    detected_objects: List[DetectedObject] = []
    category_counts: Dict[str, int] = {}
    status_counts = RecyclingStatusCounts()

    vit_num_classes = 6
    if hasattr(vit_model, "config") and hasattr(vit_model.config, "num_labels"):
        val = getattr(vit_model.config, "num_labels", 6)
        if isinstance(val, int):
            vit_num_classes = val

    has_clip = clip_model is not None and clip_processor is not None

    crop_records = []
    for idx, (bbox, det_conf, label) in enumerate(candidates, start=1):
        # 1. Tight Individual Object Cropping (2% pad, clipped to image bounds)
        pad_x = (bbox.x2 - bbox.x1) * 0.02
        pad_y = (bbox.y2 - bbox.y1) * 0.02
        cx1 = max(0, int(round(bbox.x1 - pad_x)))
        cy1 = max(0, int(round(bbox.y1 - pad_y)))
        cx2 = min(w, int(round(bbox.x2 + pad_x)))
        cy2 = min(h, int(round(bbox.y2 + pad_y)))

        crop_img = pil_image.crop((cx1, cy1, cx2, cy2))

        crop_buf = io.BytesIO()
        crop_img.save(crop_buf, format="JPEG", quality=85)
        crop_b64 = f"data:image/jpeg;base64,{base64.b64encode(crop_buf.getvalue()).decode('utf-8')}"

        # Preserve aspect ratio before ViT/CLIP resizing so tall bottles/cans do not squash
        prep_crop = _prepare_crop_for_classification(crop_img)
        crop_records.append((idx, bbox, det_conf, label, cx1, cy1, cx2, cy2, crop_img, crop_b64, prep_crop))

    # 2. Batched ViT Inference across prepared crops (respecting test monkey-patches)
    vit_probs_list: List[Dict[str, float]] = []
    if crop_records:
        prep_crops = [rec[10] for rec in crop_records]
        vit_is_patched = (
            predict_vit is not _ORIG_PREDICT_VIT
            or predict_vit_batch is not _ORIG_PREDICT_VIT_BATCH
        )
        if not vit_is_patched:
            try:
                vit_probs_list = predict_vit_batch(vit_model, vit_processor, prep_crops)
            except Exception as exc:
                logger.warning("Batched ViT inference failed (%s). Falling back to per-crop inference.", exc)
                vit_probs_list = []

        if len(vit_probs_list) != len(crop_records):
            vit_probs_list = []
            for idx_item, *_, prep_crop in crop_records:
                try:
                    vit_probs_list.append(predict_vit(vit_model, vit_processor, prep_crop))
                except Exception as exc:
                    logger.warning("ViT crop classification failed on object %d: %s", idx_item, exc)
                    vit_probs_list.append(
                        {c: 1.0 / 6.0 for c in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]}
                    )

    # 3. Batched CLIP V2 Inference across prepared crops (respecting test monkey-patches)
    clip_results_list: List[Any] = []
    if crop_records and has_clip:
        prep_crops = [rec[10] for rec in crop_records]
        clip_is_patched = (
            predict_clip_v2 is not _ORIG_PREDICT_CLIP_V2
            or predict_clip_v2_batch is not _ORIG_PREDICT_CLIP_V2_BATCH
        )
        if not clip_is_patched:
            try:
                clip_results_list = predict_clip_v2_batch(clip_model, clip_processor, prep_crops)
            except Exception as exc:
                logger.warning("Batched CLIP V2 inference failed (%s). Falling back to per-crop inference.", exc)
                clip_results_list = []

        if len(clip_results_list) != len(crop_records):
            clip_results_list = []
            for idx_item, *_, prep_crop in crop_records:
                vit_probs_cur = vit_probs_list[idx_item - 1]
                vit_top_cur = max(vit_probs_cur, key=vit_probs_cur.get)  # type: ignore[arg-type]
                try:
                    clip_results_list.append(predict_clip_v2(clip_model, clip_processor, prep_crop))
                except Exception as exc:
                    logger.warning("CLIP V2 inference failed: %s. Using 6-class fallback.", exc)
                    try:
                        clip_results_list.append(predict_clip(clip_model, clip_processor, prep_crop))
                    except Exception:
                        clip_results_list.append(vit_probs_cur if vit_probs_cur else {vit_top_cur: 1.0})

    for record_idx, (idx, bbox, det_conf, label, cx1, cy1, cx2, cy2, crop_img, crop_b64, prep_crop) in enumerate(crop_records):
        vit_probs = vit_probs_list[record_idx]
        vit_top_class = max(vit_probs, key=vit_probs.get)  # type: ignore[arg-type]
        vit_top_conf = round(float(vit_probs[vit_top_class]), 4)

        clip_probs: Dict[str, float] = {}
        clip_pred: str = ""
        if has_clip:
            clip_v2_res = clip_results_list[record_idx]
            if hasattr(clip_v2_res, "probabilities"):
                clip_probs = clip_v2_res.probabilities
                clip_pred = clip_v2_res.predicted_category
            elif isinstance(clip_v2_res, dict):
                clip_probs = clip_v2_res
                clip_pred = max(clip_probs, key=clip_probs.get)  # type: ignore[arg-type]
            else:
                clip_probs = vit_probs
                clip_pred = vit_top_class
        else:
            clip_probs = vit_probs
            clip_pred = vit_top_class

        clip_top_conf = round(float(clip_probs.get(clip_pred, 0.0)), 4)

        # 4. Score Alignment & Balanced Crop Ensemble Fusion (separate from detector confidence)
        clip_organic_prob = clip_probs.get("organic", 0.0)
        second_clip_prob = max([v for k, v in clip_probs.items() if k != "organic"], default=0.0)
        clip_organic_margin = clip_organic_prob - second_clip_prob

        is_organic_v1_bypass = (
            vit_num_classes == 6
            and clip_pred == "organic"
            and clip_organic_prob >= 0.40
            and clip_organic_margin >= 0.10
        )

        if is_organic_v1_bypass:
            final_cat = "organic"
            raw_ensemble_conf = float(clip_organic_prob)
            final_conf = clip_organic_prob
            class_probs = {
                "cardboard": round(vit_probs.get("cardboard", 0.0) * 0.48 + clip_probs.get("cardboard", 0.0) * 0.52, 4),
                "glass": round(vit_probs.get("glass", 0.0) * 0.48 + clip_probs.get("glass", 0.0) * 0.52, 4),
                "metal": round(vit_probs.get("metal", 0.0) * 0.48 + clip_probs.get("metal", 0.0) * 0.52, 4),
                "paper": round(vit_probs.get("paper", 0.0) * 0.48 + clip_probs.get("paper", 0.0) * 0.52, 4),
                "plastic": round(vit_probs.get("plastic", 0.0) * 0.48 + clip_probs.get("plastic", 0.0) * 0.52, 4),
                "organic": round(clip_organic_prob, 4),
                "trash": round(vit_probs.get("trash", 0.0) * 0.25 + clip_probs.get("trash", 0.0) * 0.52, 4),
            }
            tot = sum(class_probs.values())
            class_probs = {k: round(v / tot, 4) for k, v in class_probs.items()}
            is_ood = False
            conf_status = "normal" if final_conf >= 0.70 else "ambiguous"
            models_agree = False
            vit_prediction_label = "Not supported (V1 6-class)"
            compat_decision = "organic_v1_clip_bypass"
        else:
            # Balanced 48/52 ViT+CLIP fusion on cropped instances prevents domain-shifted ViT
            # logits from automatically overriding confident CLIP predictions (e.g. paper vs cardboard)
            w_vit_crop, w_clip_crop = (0.48, 0.52) if has_clip else (1.0, 0.0)
            fused = fuse(
                vit_probs,
                {k: clip_probs.get(k, 0.0) for k in vit_probs},
                vit_weight=w_vit_crop,
                clip_weight=w_clip_crop,
            )
            final_cat = fused.predicted_category
            raw_ensemble_conf = float(fused.confidence)
            final_conf = fused.confidence
            class_probs = {k: round(v, 4) for k, v in fused.class_probabilities.items()}
            is_ood = fused.is_out_of_distribution
            models_agree = fused.vit_prediction.lower() == fused.clip_prediction.lower()
            vit_prediction_label = fused.vit_prediction
            compat_decision = "consensus_or_disagreement" if vit_num_classes >= 7 else "v1_shared_6class"

            if is_ood:
                conf_status = "out_of_distribution"
            elif not models_agree:
                conf_status = "ambiguous"
                final_conf = min(final_conf, 0.68)
            elif final_conf >= 0.70:
                conf_status = "normal"
            elif final_conf >= 0.40:
                conf_status = "ambiguous"
            else:
                conf_status = "low_confidence"

        final_conf = min(0.985, round(final_conf, 4))

        # 5. Smart Bin / Intelligent Recycling Decision
        rec_decision = get_recycling_decision(
            category=final_cat,
            confidence=final_conf,
            is_ood=is_ood,
            confidence_status=conf_status,
            item_context=label,
            model_agreement=models_agree,
            is_organic_v1_bypass=is_organic_v1_bypass,
        )

        category_counts[final_cat] = category_counts.get(final_cat, 0) + 1
        disp_status = rec_decision.disposal_status
        if disp_status == DisposalStatus.RECYCLABLE:
            status_counts.recyclable += 1
        elif disp_status == DisposalStatus.CONDITIONALLY_RECYCLABLE:
            status_counts.conditionally_recyclable += 1
        elif disp_status == DisposalStatus.COMPOSTABLE:
            status_counts.compostable += 1
        elif disp_status == DisposalStatus.HAZARDOUS_SPECIAL_DISPOSAL:
            status_counts.hazardous += 1
        elif disp_status == DisposalStatus.NOT_STANDARD_RECYCLABLE:
            status_counts.not_standard += 1
        else:
            status_counts.unknown += 1

        crop_w = int(cx2 - cx1)
        crop_h = int(cy2 - cy1)
        model_cov = ModelCoverage(
            vit_supported=(vit_num_classes >= 7 or final_cat != "organic"),
            clip_supported=True,
            ensemble_supported=(vit_num_classes >= 7 or final_cat != "organic"),
            unsupported_reason=(
                "ViT V1 is fine-tuned on 6 classes and lacks the organic class. Evaluated via CLIP V2 foundation model (validated organic confidence & margin)."
                if is_organic_v1_bypass
                else None
            ),
            compatibility_decision=compat_decision,
        )

        detected_objects.append(
            DetectedObject(
                object_id=idx,
                bbox=bbox,
                detection_confidence=round(det_conf, 3),
                detection_label=f"{final_cat.capitalize()} item",
                material_category=final_cat,
                vit_prediction=vit_prediction_label,
                clip_prediction=clip_pred,
                vit_confidence=0.0 if is_organic_v1_bypass else vit_top_conf,
                clip_confidence=clip_top_conf,
                raw_ensemble_confidence=round(raw_ensemble_conf, 4),
                final_category=final_cat,
                class_probabilities=class_probs,
                vit_probabilities={k: round(float(v), 4) for k, v in vit_probs.items()},
                clip_probabilities={k: round(float(v), 4) for k, v in clip_probs.items()},
                calibrated_confidence=round(final_conf, 4),
                confidence_status=conf_status,
                is_out_of_distribution=is_ood,
                model_agreement=models_agree,
                model_coverage=model_cov,
                detector_source=label,
                crop_width=crop_w,
                crop_height=crop_h,
                crop_image_base64=crop_b64,
                recycling_status=rec_decision.disposal_status.value,
                disposal_recommendation=rec_decision.action,
                preparation_steps=rec_decision.preparation_steps,
                recycling_decision=RecyclingDecisionSchema(**rec_decision.to_dict()),
            )
        )

    sum_stream = (
        status_counts.recyclable
        + status_counts.conditionally_recyclable
        + status_counts.compostable
        + status_counts.hazardous
        + status_counts.not_standard
        + status_counts.unknown
    )
    if sum_stream != len(detected_objects):
        logger.error("Disposal stream count mismatch: %d != %d", sum_stream, len(detected_objects))

    clean_buf = io.BytesIO()
    pil_image.convert("RGB").save(clean_buf, format="JPEG", quality=90)
    clean_base64 = f"data:image/jpeg;base64,{base64.b64encode(clean_buf.getvalue()).decode('utf-8')}"

    # Attach optional batched Resin Code / Hazard analysis on eligible crops
    if detected_objects and crop_records:
        try:
            resin_results = analyze_resin_and_hazards_batch(
                [(rec[8], obj.material_category) for rec, obj in zip(crop_records, detected_objects)],
                clip_model=clip_model if has_clip else None,
                clip_processor=clip_processor if has_clip else None,
            )
            for obj, r_dict in zip(detected_objects, resin_results):
                if r_dict:
                    obj.resin_info = ResinInfoSchema(**r_dict)
        except Exception as resin_exc:
            logger.warning("Batched resin/hazard crop analysis failed (non-fatal): %s", resin_exc)

    annotated_base64 = render_annotated_image(pil_image, detected_objects)
    inference_time = (time.perf_counter() - t0) * 1000.0

    vit_version_str = (
        "google/vit-base-patch16-224-in21k-v2"
        if vit_num_classes >= 7
        else "google/vit-base-patch16-224-in21k-v1"
    )

    return DetectResponse(
        image_width=w,
        image_height=h,
        total_objects=len(detected_objects),
        objects=detected_objects,
        category_counts=category_counts,
        recycling_summary=RecyclingSummary(
            total_objects=len(detected_objects),
            category_counts=category_counts,
            status_counts=status_counts,
        ),
        model_versions={
            "vit_classifier": vit_version_str,
            "clip_foundation": "openai/clip-vit-base-patch32",
            "detector_engine": f"WasteObjectDetector-{detector.model_status}",
            "decision_engine": "v2.0-swm-rules",
        },
        inference_time_ms=round(inference_time, 2),
        annotated_image_base64=annotated_base64,
        clean_image_base64=clean_base64,
        detection_model_status=detector.model_status,
    )
