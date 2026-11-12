"""
Multi-Object Waste Detection and Material Classification Engine for EcoSort AI V2.

Pipeline:
1. Object Localization: Extracts spatial bounding boxes for waste items using Pillow + NumPy.
2. Image Region Crop: Slices individual bounding-box crops from the full image.
3. Material Classification: Passes each crop to the ViT + CLIP ensemble.
4. Intelligent Decision: Runs the Recycling Decision Engine per detected item.
5. Visualization: Renders annotated bounding boxes with category color-coding.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from app.ml.clip_model import predict_clip
from app.ml.ensemble import fuse
from app.ml.vit_model import predict_vit
from app.schemas.detect import (
    BoundingBox,
    DetectedObject,
    DetectResponse,
    RecyclingStatusCounts,
    RecyclingSummary,
)
from app.schemas.predict import RecyclingDecisionSchema
from app.services.recycling_engine import DisposalStatus, get_recycling_decision

logger = logging.getLogger(__name__)

# Category colors for visualization (Hex and RGB)
CATEGORY_HEX = {
    "cardboard": "#D4A54A",
    "glass": "#0EA5E9",
    "metal": "#64748B",
    "paper": "#10B981",
    "plastic": "#F97316",
    "organic": "#84CC16",
    "trash": "#6B7280",
    "unknown": "#94A3B8",
}

CATEGORY_RGB = {
    "cardboard": (212, 165, 74),
    "glass": (14, 165, 233),
    "metal": (100, 116, 139),
    "paper": (16, 185, 129),
    "plastic": (249, 115, 22),
    "organic": (132, 204, 22),
    "trash": (107, 114, 128),
    "unknown": (148, 163, 184),
}


def _calculate_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union of two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom


def extract_candidate_regions(pil_image: Image.Image) -> List[Tuple[BoundingBox, float, str]]:
    """
    Extract candidate waste object bounding boxes from image using Pillow edge detection,
    intensity thresholding, and morphological region clustering.

    Zero dependency on external opencv/cv2: uses PIL + NumPy.
    Returns a list of (BoundingBox, detection_confidence, detection_label).
    """
    rgb_img = pil_image.convert("RGB")
    w, h = rgb_img.size
    total_area = float(w * h)

    # 1. Edge & gradient detection via PIL
    gray_img = rgb_img.convert("L")
    edges_img = gray_img.filter(ImageFilter.FIND_EDGES)
    edges_arr = np.array(edges_img, dtype=np.float32)

    # 2. Local variance / contrast map
    blurred = gray_img.filter(ImageFilter.BoxBlur(5))
    diff_arr = np.abs(np.array(gray_img, dtype=np.float32) - np.array(blurred, dtype=np.float32))

    saliency_map = edges_arr + (diff_arr * 1.5)
    thresh = np.percentile(saliency_map, 72)
    binary_mask = (saliency_map > thresh).astype(np.uint8)

    # 3. Grid-based connected block aggregation
    # Divide image into a responsive cellular grid to cluster active pixels into boxes
    grid_rows, grid_cols = 16, 16
    cell_h = h / grid_rows
    cell_w = w / grid_cols

    grid_activity = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for r in range(grid_rows):
        y_start, y_end = int(r * cell_h), int((r + 1) * cell_h)
        for c in range(grid_cols):
            x_start, x_end = int(c * cell_w), int((c + 1) * cell_w)
            grid_activity[r, c] = np.mean(binary_mask[y_start:y_end, x_start:x_end])

    # Find connected components in the 16x16 grid
    visited = np.zeros_like(grid_activity, dtype=bool)
    active_thresh = 0.08
    components: List[List[Tuple[int, int]]] = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            if grid_activity[r, c] >= active_thresh and not visited[r, c]:
                comp: List[Tuple[int, int]] = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
                            if not visited[nr, nc] and grid_activity[nr, nc] >= active_thresh:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                components.append(comp)

    candidates: List[Tuple[Tuple[float, float, float, float], float, str]] = []
    min_box_area = total_area * 0.015
    max_box_area = total_area * 0.92

    for comp in components:
        min_r = min(pt[0] for pt in comp)
        max_r = max(pt[0] for pt in comp)
        min_c = min(pt[1] for pt in comp)
        max_c = max(pt[1] for pt in comp)

        x1 = max(0.0, float(min_c * cell_w) - cell_w * 0.1)
        y1 = max(0.0, float(min_r * cell_h) - cell_h * 0.1)
        x2 = min(float(w), float((max_c + 1) * cell_w) + cell_w * 0.1)
        y2 = min(float(h), float((max_r + 1) * cell_h) + cell_h * 0.1)

        box_area = (x2 - x1) * (y2 - y1)
        if box_area < min_box_area or box_area > max_box_area:
            continue

        # Confidence based on cell density
        mean_density = float(np.mean([grid_activity[pt[0], pt[1]] for pt in comp]))
        det_score = min(0.96, max(0.68, 0.72 + (mean_density * 0.8)))

        candidates.append(((x1, y1, x2, y2), det_score, "waste_item"))

    # 4. Non-Maximum Suppression (NMS)
    candidates.sort(key=lambda item: item[1], reverse=True)
    kept_candidates: List[Tuple[BoundingBox, float, str]] = []

    for box, score, label in candidates:
        overlap = False
        for kept_box, _, _ in kept_candidates:
            kept_tup = (kept_box.x1, kept_box.y1, kept_box.x2, kept_box.y2)
            if _calculate_iou(box, kept_tup) > 0.40:
                overlap = True
                break
        if not overlap:
            kept_candidates.append((
                BoundingBox(x1=round(box[0], 1), y1=round(box[1], 1), x2=round(box[2], 1), y2=round(box[3], 1)),
                round(score, 3),
                label,
            ))

    # If no distinctive objects found, fallback to single item bounding box
    if not kept_candidates:
        kept_candidates.append((
            BoundingBox(x1=float(w * 0.05), y1=float(h * 0.05), x2=float(w * 0.95), y2=float(h * 0.95)),
            0.88,
            "waste_object",
        ))

    return kept_candidates[:12]  # Cap at top 12 objects


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

        # Label background pill
        text_bbox = draw.textbbox((box.x1, box.y1), label_text, font=font) if font else (box.x1, box.y1, box.x1 + 100, box.y1 + 18)
        text_w = text_bbox[2] - text_bbox[0] + 10
        text_h = text_bbox[3] - text_bbox[1] + 6

        pill_y1 = max(0.0, box.y1 - text_h - 2)
        pill_y2 = pill_y1 + text_h
        pill_x1 = box.x1
        pill_x2 = min(float(annotated.width), pill_x1 + text_w)

        draw.rectangle([pill_x1, pill_y1, pill_x2, pill_y2], fill=rgb)
        draw.text((pill_x1 + 5, pill_y1 + 3), label_text, fill=(255, 255, 255), font=font)

    # Encode to JPEG Base64
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
    Full Multi-Object Waste Detection pipeline.

    1. Extracts candidate object bounding boxes.
    2. Crops each candidate region.
    3. Runs ViT + CLIP material classification on each crop.
    4. Computes per-object intelligent recycling recommendations.
    5. Summarizes multi-object statistics.
    6. Returns structured response with base64 annotated image.
    """
    t0 = time.perf_counter()
    w, h = pil_image.size

    # Step 1: Detect candidate regions
    candidates = extract_candidate_regions(pil_image)

    detected_objects: List[DetectedObject] = []
    category_counts: Dict[str, int] = {}
    status_counts = RecyclingStatusCounts()

    # Step 2 & 3: Crop and Classify
    for idx, (bbox, det_conf, label) in enumerate(candidates, start=1):
        # Crop region from original image
        crop_box = (int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2))
        crop_img = pil_image.crop(crop_box)

        # ViT inference on crop
        try:
            vit_probs = predict_vit(vit_model, vit_processor, crop_img)
        except Exception as exc:
            logger.warning("ViT crop classification failed on object %d: %s", idx, exc)
            vit_probs = {c: 1.0 / 6.0 for c in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]}

        # CLIP inference on crop
        if clip_model is not None and clip_processor is not None:
            try:
                clip_res = predict_clip(clip_model, clip_processor, crop_img)
                clip_probs = (
                    clip_res["probabilities"]
                    if isinstance(clip_res, dict) and "probabilities" in clip_res
                    else clip_res
                )
            except Exception as exc:
                logger.warning("CLIP crop classification failed on object %d: %s", idx, exc)
                clip_probs = vit_probs
        else:
            clip_probs = vit_probs

        # Fusion
        fused = fuse(vit_probs, clip_probs)

        # Recycling Decision Engine
        rec_decision = get_recycling_decision(
            category=fused.predicted_category,
            confidence=fused.confidence,
            is_ood=fused.is_out_of_distribution,
            confidence_status=fused.confidence_status,
            item_context=label,
        )

        # Accumulate counts
        category_counts[fused.predicted_category] = category_counts.get(fused.predicted_category, 0) + 1

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

        detected_objects.append(
            DetectedObject(
                object_id=idx,
                bbox=bbox,
                detection_confidence=round(det_conf, 3),
                detection_label=f"{fused.predicted_category.capitalize()} item",
                material_category=fused.predicted_category,
                vit_prediction=fused.vit_prediction,
                clip_prediction=fused.clip_prediction,
                final_category=fused.predicted_category,
                class_probabilities={k: round(v, 4) for k, v in fused.class_probabilities.items()},
                calibrated_confidence=round(fused.calibrated_confidence or fused.confidence, 4),
                confidence_status=fused.confidence_status,
                is_out_of_distribution=fused.is_out_of_distribution,
                recycling_status=rec_decision.disposal_status.value,
                disposal_recommendation=rec_decision.action,
                preparation_steps=rec_decision.preparation_steps,
                recycling_decision=RecyclingDecisionSchema(**rec_decision.to_dict()),
            )
        )

    # Step 4: Render annotated visual
    annotated_base64 = render_annotated_image(pil_image, detected_objects)
    inference_time = (time.perf_counter() - t0) * 1000.0

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
            "vit_classifier": "google/vit-base-patch16-224-in21k-v1",
            "clip_foundation": "openai/clip-vit-base-patch32",
            "detector_engine": "EcoSort-SpatialRegionProposal-v1",
            "decision_engine": "v2.0-swm-rules",
        },
        inference_time_ms=round(inference_time, 2),
        annotated_image_base64=annotated_base64,
        detection_model_status="ready",
    )
