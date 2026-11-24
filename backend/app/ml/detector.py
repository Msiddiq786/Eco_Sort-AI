"""
Multi-Object Waste Detection and Localization Engine for EcoSort AI V2.

Architecture:
1. Model Detection Engine:
   - Dedicated WasteObjectDetector.
   - Checks for trained weights in backend/models/detection_v2/weights/best.pt.
   - When awaiting trained weights, applies robust multi-scale edge gradient decomposition,
     binary morphological hole-filling, and valley splitting to reliably isolate
     distinct, non-overlapping individual waste items in complex scenes without fabricating detections.
2. Coordinate Bounds & Clipping:
   - Enforces strict image boundaries (0 <= x1 < x2 <= W, 0 <= y1 < y2 <= H).
   - Rejects scene-swallowing bounding boxes (caps maximum single object area to 38% of frame).
3. Non-Maximum Suppression (NMS):
   - Spatial overlap suppression (IoU <= 0.38) to eliminate duplicate proposals while
     safeguarding distinct adjacent items.
4. Independent Crop Extraction:
   - Pads each bounding box by 5%, clips to boundary, and crops original image.
   - Evaluates ViT and CLIP independently on each crop.
   - Generates crop base64 thumbnail for rich frontend inspection cards.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.ml.clip_model import predict_clip
from app.ml.clip_model_v2 import predict_clip_v2
from app.ml.ensemble import fuse
from app.ml.vit_model import predict_vit
from app.schemas.detect import (
    BoundingBox,
    DetectedObject,
    DetectResponse,
    ModelCoverage,
    RecyclingStatusCounts,
    RecyclingSummary,
)
from app.schemas.predict import RecyclingDecisionSchema
from app.services.recycling_engine import DisposalStatus, get_recycling_decision

logger = logging.getLogger(__name__)

# Configured paths for V2 detection models
DETECTOR_CONFIG_PATH = Path("backend/models/detection_v2/config.yaml")
DETECTOR_WEIGHTS_PATH = Path("backend/models/detection_v2/weights/best.pt")

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


def merge_can_fragments(
    proposals: List[Tuple[Tuple[float, float, float, float], float, str]],
    max_v_gap: float = 60.0,
) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
    """Merge vertically split fragments of the same cylindrical object (e.g. can top rim and body, or bottle segments)."""
    merged: List[Tuple[Tuple[float, float, float, float], float, str]] = []
    used = set()
    n = len(proposals)
    for i in range(n):
        if i in used:
            continue
        boxA, scoreA, labelA = proposals[i]
        merged_box = list(boxA)
        merged_score = scoreA
        merged_label = labelA
        wA = merged_box[2] - merged_box[0]
        hA = merged_box[3] - merged_box[1]

        if wA <= 250 and hA <= 300:
            for j in range(i + 1, n):
                if j in used:
                    continue
                boxB, scoreB, labelB = proposals[j]
                wB = boxB[2] - boxB[0]
                hB = boxB[3] - boxB[1]
                if wB <= 250 and hB <= 300:
                    xA = max(merged_box[0], boxB[0])
                    xB = min(merged_box[2], boxB[2])
                    inter_x = max(0.0, xB - xA)
                    min_w = min(wA, wB)
                    if min_w > 0 and (inter_x / min_w) > 0.60:
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
    containment_threshold: float = 0.65,
    iou_threshold: float = 0.35,
    max_area_ratio: float = 0.85,
    total_area: float = 1.0,
) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
    """
    Robust object-level containment suppression and Non-Maximum Suppression.
    Suppresses internal sub-features (labels, caps, logos, rims, reflection artifacts)
    while strictly safeguarding distinct adjacent physical items.
    """
    if not proposals:
        return []

    # Priority sort: YOLO proposals given 1.5x score priority, then score + area stability
    def sort_key(p):
        box, score, label = p
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        area = bw * bh
        yolo_boost = 1.5 if label.startswith("yolo_") else 1.0
        return yolo_boost * score * (1.0 + 0.25 * min(1.0, area / 40000.0))

    sorted_props = sorted(proposals, key=sort_key, reverse=True)
    kept: List[Tuple[Tuple[float, float, float, float], float, str]] = []

    for boxA, scoreA, labelA in sorted_props:
        areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        if total_area > 0 and areaA > total_area * max_area_ratio:
            continue

        should_add = True
        to_remove = []

        for idx, (boxK, scoreK, labelK) in enumerate(kept):
            areaK = max(0.0, (boxK[2] - boxK[0]) * (boxK[3] - boxK[1]))

            xA = max(boxA[0], boxK[0])
            yA = max(boxA[1], boxK[1])
            xB = min(boxA[2], boxK[2])
            yB = min(boxA[3], boxK[3])
            inter = max(0.0, xB - xA) * max(0.0, yB - yA)

            union = areaA + areaK - inter
            iou = (inter / union) if union > 0 else 0.0

            # Containment evaluation of smaller box inside larger box
            if areaA <= areaK:
                # boxA is smaller or equal; check if it is inside boxK
                cont = (inter / areaA) if areaA > 0 else 0.0
                if cont >= containment_threshold or iou >= iou_threshold:
                    # boxA is an inner sub-feature of boxK -> suppress boxA
                    should_add = False
                    break
            else:
                # boxA is larger; check if boxK is inside boxA
                cont = (inter / areaK) if areaK > 0 else 0.0
                if cont >= containment_threshold or iou >= iou_threshold:
                    # boxK was an inner sub-feature of boxA -> supersede boxK with boxA
                    to_remove.append(idx)

        if should_add:
            for idx in sorted(to_remove, reverse=True):
                kept.pop(idx)
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


class WasteObjectDetector:
    """
    Dedicated Multi-Object Waste Detector.

    Combines:
    - Pretrained YOLOv8 for physical object containers (glass/plastic bottles, cans, cups, bowls).
    - Multi-scale edge-gradient and morphological region analysis for cardboard, paper, and wrappers.
    - Border-contrast saliency for transparent plastic and subtle reflective objects.
    - Vertically aligned can fragment merging and IoMin containment suppression to eliminate duplicates.
    """

    def __init__(self, weights_path: Optional[Path] = None):
        self.weights_path = weights_path or DETECTOR_WEIGHTS_PATH
        self.yolo_model = None
        self.model_status = "awaiting_training"
        self._initialize_detector()

    def _initialize_detector(self):
        """Attempt to load trained YOLO weights or local pretrained YOLO weights."""
        if self.weights_path.exists() and self.weights_path.is_file():
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(str(self.weights_path))
                self.model_status = "trained_weights_ready"
                logger.info("Successfully loaded dedicated YOLO detector from %s", self.weights_path)
                return
            except Exception as exc:
                logger.warning("Could not load trained YOLO weights: %s.", exc)

        # Check for local yolov8s.pt or yolov8n.pt in workspace root
        for candidate_pt in [Path("yolov8s.pt"), Path("yolov8n.pt")]:
            if candidate_pt.exists() and candidate_pt.is_file():
                try:
                    from ultralytics import YOLO
                    self.yolo_model = YOLO(str(candidate_pt))
                    self.model_status = f"pretrained_hybrid_{candidate_pt.stem}"
                    logger.info("Loaded pretrained YOLO hybrid model from %s", candidate_pt)
                    return
                except Exception as exc:
                    logger.warning("Could not load local YOLO %s: %s", candidate_pt, exc)

        self.model_status = "fallback_multiscale_proposal"
        logger.info("No YOLO weights available. Using multi-scale proposal engine.")

    def detect_objects(
        self,
        pil_image: Image.Image,
        conf_threshold: float = 0.35,
        max_detections: int = 15,
    ) -> List[Tuple[BoundingBox, float, str]]:
        """
        Detect candidate waste objects and generate genuine individual bounding boxes.
        Never fabricates detections; blank images return [].
        """
        w, h = pil_image.size
        total_area = float(w * h)

        gray = pil_image.convert("L")
        arr_gray = np.array(gray, dtype=np.float32)

        gx = ndi.sobel(arr_gray, axis=1)
        gy = ndi.sobel(arr_gray, axis=0)
        mag = np.hypot(gx, gy)
        max_m = float(mag.max())

        # Zero variation / blank image check
        if max_m < 5.0 or float(np.std(arr_gray)) < 5.0:
            return []

        min_box_area = total_area * 0.003
        max_box_area = total_area * 0.38
        yolo_max_box_area = total_area * 0.85

        proposals: List[Tuple[Tuple[float, float, float, float], float, str]] = []

        # 1. Deep proposals from YOLO (physical objects & containers)
        if self.yolo_model is not None:
            try:
                ALLOWED_CLASSES = {
                    # Liquid & Food Containers
                    "bottle", "cup", "wine glass", "bowl", "vase",
                    # Organic Food Waste (banana peels, apple cores, fruit/vegetable scraps)
                    "banana", "apple", "orange", "broccoli", "carrot", "sandwich", "pizza", "hot dog", "donut", "cake",
                    # Packaging / Boxes / Bags / Wrappers
                    "book", "suitcase", "handbag", "backpack",
                    # Utensils & Objects
                    "fork", "knife", "spoon", "scissors"
                }
                results = self.yolo_model(pil_image, conf=0.15, verbose=False)
                for r in results:
                    for b in r.boxes:
                        cls_id = int(b.cls[0].cpu().numpy())
                        name = r.names.get(cls_id, f"obj_{cls_id}")
                        if name not in ALLOWED_CLASSES:
                            continue
                        coords = [float(c) for c in b.xyxy[0].cpu().numpy()]
                        conf = float(b.conf[0].cpu().numpy())
                        area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                        if min_box_area <= area <= yolo_max_box_area:
                            proposals.append(((coords[0], coords[1], coords[2], coords[3]), min(0.98, conf + 0.15), f"yolo_{name}"))
            except Exception as exc:
                logger.error("Deep proposal inference failed: %s", exc)

        # 2. Multi-scale edge-gradient and morphological region proposals
        for p_ratio in [0.15, 0.28, 0.42]:
            edge_bin = mag > (max_m * p_ratio)
            filled = ndi.binary_fill_holes(edge_bin)
            clean = ndi.binary_opening(filled, structure=np.ones((3, 3)))
            clean = ndi.binary_closing(clean, structure=np.ones((3, 3)))
            labeled, num_f = ndi.label(clean)
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

                # Filter border artifacts
                if (x1 <= 4 and x2 <= 40 and bh > h * 0.7) or (x1 >= w - 40 and x2 >= w - 4 and bh > h * 0.7):
                    continue

                if min_box_area <= area <= max_box_area:
                    score = float(np.mean(mag[y1:y2, x1:x2])) / (max_m + 1e-5)
                    area_ratio = area / total_area
                    boosted_score = 0.50 + 0.35 * score + 0.15 * min(1.0, area_ratio / 0.10)
                    proposals.append(((float(x1), float(y1), float(x2), float(y2)), boosted_score, "morph_obj"))
                elif area > max_box_area:
                    # Projection valley splitting to sever touching objects
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

        # 3. Background color-contrast saliency proposals (transparent plastic, soft-edge containers)
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
        norm_mag = mag / (max_m + 1e-5)
        saliency = 0.6 * (diff / max_diff) + 0.4 * norm_mag

        for thresh in [0.28, 0.42]:
            bin_mask = (saliency > thresh).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, bw, bh = cv2.boundingRect(c)
                area = float(bw * bh)
                if min_box_area <= area <= max_box_area:
                    score = float(np.mean(saliency[y:y+bh, x:x+bw]))
                    proposals.append(((float(x), float(y), float(x+bw), float(y+bh)), score, "contrast_obj"))

        # 4. Merge vertically aligned can fragments
        proposals = merge_can_fragments(proposals, max_v_gap=60.0)

        # 5. Suppress nested containment and overlapping boxes
        suppressed_proposals = apply_containment_and_nms(
            proposals,
            containment_threshold=0.65,
            iou_threshold=0.35,
            max_area_ratio=0.85,
            total_area=total_area,
        )

        validated_boxes: List[Tuple[BoundingBox, float, str]] = []
        for box, score, label in suppressed_proposals[:max_detections]:
            x1 = max(0.0, min(float(w), round(box[0], 1)))
            y1 = max(0.0, min(float(h), round(box[1], 1)))
            x2 = max(x1 + 10.0, min(float(w), round(box[2], 1)))
            y2 = max(y1 + 10.0, min(float(h), round(box[3], 1)))

            box_area = (x2 - x1) * (y2 - y1)
            # Deep YOLO objects or isolated foreground items can occupy up to 85% of frame
            effective_max_area = (
                total_area * 0.85
                if (label.startswith("yolo_") or len(suppressed_proposals) <= 2)
                else total_area * 0.38
            )
            if box_area > effective_max_area or box_area < min_box_area:
                continue

            validated_boxes.append((
                BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
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

        # 1. Draw bounding box rectangle
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=rgb, width=3)

        # 2. Pill label text
        label_text = f"#{obj.object_id} {obj.material_category.capitalize()} ({obj.calibrated_confidence*100:.0f}%)"

        text_bbox = draw.textbbox((box.x1, box.y1), label_text, font=font) if font else (box.x1, box.y1, box.x1 + 100, box.y1 + 18)
        text_w = text_bbox[2] - text_bbox[0] + 10
        text_h = text_bbox[3] - text_bbox[1] + 6

        pill_y1 = max(0.0, box.y1 - text_h - 2)
        pill_y2 = pill_y1 + text_h
        pill_x1 = box.x1
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
    Full Multi-Object Waste Detection & Crop-Level Classification pipeline.
    Reuses existing loaded model instances without per-crop reloading.
    """
    t0 = time.perf_counter()
    w, h = pil_image.size

    detector = get_detector()
    candidates = detector.detect_objects(pil_image)

    detected_objects: List[DetectedObject] = []
    category_counts: Dict[str, int] = {}
    status_counts = RecyclingStatusCounts()

    # Determine ViT class capability from model config if available
    vit_num_classes = 6
    if hasattr(vit_model, "config") and hasattr(vit_model.config, "num_labels"):
        val = getattr(vit_model.config, "num_labels", 6)
        if isinstance(val, int):
            vit_num_classes = val

    for idx, (bbox, det_conf, label) in enumerate(candidates, start=1):
        # 1. Individual Object Cropping with tight 3% padding & boundary clipping
        pad_x = (bbox.x2 - bbox.x1) * 0.03
        pad_y = (bbox.y2 - bbox.y1) * 0.03
        cx1 = max(0, int(bbox.x1 - pad_x))
        cy1 = max(0, int(bbox.y1 - pad_y))
        cx2 = min(w, int(bbox.x2 + pad_x))
        cy2 = min(h, int(bbox.y2 + pad_y))

        crop_img = pil_image.crop((cx1, cy1, cx2, cy2))

        # Generate base64 thumbnail of crop for UI card inspection
        crop_buf = io.BytesIO()
        crop_img.save(crop_buf, format="JPEG", quality=85)
        crop_b64 = f"data:image/jpeg;base64,{base64.b64encode(crop_buf.getvalue()).decode('utf-8')}"

        # 2. ViT Inference on crop
        try:
            vit_probs = predict_vit(vit_model, vit_processor, crop_img)
        except Exception as exc:
            logger.warning("ViT crop classification failed on object %d: %s", idx, exc)
            vit_probs = {c: 1.0 / 6.0 for c in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]}

        # 3. CLIP Inference on crop (7-class aware)
        clip_probs: Dict[str, float] = {}
        clip_pred: str = ""
        if clip_model is not None and clip_processor is not None:
            try:
                # Use CLIP V2 with 7 classes (including organic)
                clip_v2_res = predict_clip_v2(clip_model, clip_processor, crop_img)
                if hasattr(clip_v2_res, "probabilities"):
                    clip_probs = clip_v2_res.probabilities
                    clip_pred = clip_v2_res.predicted_category
                elif isinstance(clip_v2_res, dict):
                    clip_probs = clip_v2_res
                    clip_pred = max(clip_probs, key=clip_probs.get)
                else:
                    clip_probs = vit_probs
                    clip_pred = max(vit_probs, key=vit_probs.get)
            except Exception as exc:
                logger.warning("CLIP V2 inference failed: %s. Using 6-class fallback.", exc)
                try:
                    c_res = predict_clip(clip_model, clip_processor, crop_img)
                    clip_probs = c_res.probabilities if hasattr(c_res, "probabilities") else c_res
                    clip_pred = c_res.predicted_category if hasattr(c_res, "predicted_category") else "trash"
                except Exception:
                    clip_probs = vit_probs
                    clip_pred = max(vit_probs, key=vit_probs.get)
        else:
            clip_probs = vit_probs
            clip_pred = max(vit_probs, key=vit_probs.get)

        # 4. Dynamic Class Compatibility & Ensemble Fusion
        clip_organic_prob = clip_probs.get("organic", 0.0)
        second_clip_prob = max([v for k, v in clip_probs.items() if k != "organic"], default=0.0)
        clip_organic_margin = clip_organic_prob - second_clip_prob

        # Strict Organic V1 Bypass Policy:
        # 1. ViT checkpoint verified to have six classes (vit_num_classes == 6)
        # 2. ViT reports organic as unsupported
        # 3. CLIP V2 supports organic ("organic" in clip_probs)
        # 4. CLIP organic confidence >= 0.40 and margin >= 0.10
        # 5. Compatibility decision recorded in response
        is_organic_v1_bypass = (
            vit_num_classes == 6
            and clip_pred == "organic"
            and clip_organic_prob >= 0.40
            and clip_organic_margin >= 0.10
        )

        if is_organic_v1_bypass:
            final_cat = "organic"
            final_conf = clip_organic_prob
            class_probs = {
                "cardboard": round(vit_probs.get("cardboard", 0.0) * 0.7 + clip_probs.get("cardboard", 0.0) * 0.3, 4),
                "glass": round(vit_probs.get("glass", 0.0) * 0.7 + clip_probs.get("glass", 0.0) * 0.3, 4),
                "metal": round(vit_probs.get("metal", 0.0) * 0.7 + clip_probs.get("metal", 0.0) * 0.3, 4),
                "paper": round(vit_probs.get("paper", 0.0) * 0.7 + clip_probs.get("paper", 0.0) * 0.3, 4),
                "plastic": round(vit_probs.get("plastic", 0.0) * 0.7 + clip_probs.get("plastic", 0.0) * 0.3, 4),
                "organic": round(clip_organic_prob, 4),
                "trash": round(vit_probs.get("trash", 0.0) * 0.35 + clip_probs.get("trash", 0.0) * 0.3, 4),
            }
            # Strictly normalize to 1.0
            tot = sum(class_probs.values())
            class_probs = {k: round(v / tot, 4) for k, v in class_probs.items()}
            is_ood = False
            conf_status = "normal" if final_conf >= 0.70 else "ambiguous"
            models_agree = False
            vit_prediction_label = "Not supported (V1 6-class)"
            compat_decision = "organic_v1_clip_bypass"
        else:
            # Shared classes or 7-class ViT V2
            fused = fuse(vit_probs, {k: clip_probs.get(k, 0.0) for k in vit_probs})
            final_cat = fused.predicted_category
            final_conf = fused.confidence
            class_probs = {k: round(v, 4) for k, v in fused.class_probabilities.items()}
            is_ood = fused.is_out_of_distribution
            models_agree = fused.vit_prediction.lower() == fused.clip_prediction.lower()
            vit_prediction_label = fused.vit_prediction
            compat_decision = "consensus_or_disagreement" if vit_num_classes >= 7 else "v1_shared_6class"

            # Disagreement & confidence tier policy
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

        # 5. Intelligent Recycling Decision
        rec_decision = get_recycling_decision(
            category=final_cat,
            confidence=final_conf,
            is_ood=is_ood,
            confidence_status=conf_status,
            item_context=label,
            model_agreement=models_agree,
            is_organic_v1_bypass=is_organic_v1_bypass,
        )

        # Accumulate summary counts
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
                final_category=final_cat,
                class_probabilities=class_probs,
                calibrated_confidence=round(final_conf, 4),
                confidence_status=conf_status,
                is_out_of_distribution=is_ood,
                model_agreement=models_agree,
                model_coverage=model_cov,
                crop_width=crop_w,
                crop_height=crop_h,
                crop_image_base64=crop_b64,
                recycling_status=rec_decision.disposal_status.value,
                disposal_recommendation=rec_decision.action,
                preparation_steps=rec_decision.preparation_steps,
                recycling_decision=RecyclingDecisionSchema(**rec_decision.to_dict()),
            )
        )

    # Verify strict summary conservation invariant: sum of streams == total objects
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
