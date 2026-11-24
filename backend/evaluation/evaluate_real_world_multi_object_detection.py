"""
Real-World Multi-Object Detection and Localization Evaluation Workflow.

Evaluates the EcoSort AI V2 multi-object waste detector on genuine real photographic
waste images across 10 realistic conditions:
1. Isolated objects
2. Two separate objects
3. Three or more objects
4. Touching objects
5. Overlapping objects
6. Transparent bottles
7. Reflective metal cans
8. Plastic wrappers
9. Organic food waste
10. Complex household garbage piles
11. Blank background (zero-object invariant)

Produces:
- Quantitative metrics: Precision, Recall, Average IoU, mAP@0.30, mAP@0.50,
  Missed objects, Duplicate detections, Merged objects, Split objects, Crop accuracy.
- Side-by-side visual annotations (Ground Truth vs Model Predictions).
- Inference latency, CPU time, and memory profile.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Set backend path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.ml.detector import (
    CATEGORY_HEX,
    CATEGORY_RGB,
    WasteObjectDetector,
    calculate_iou,
    detect_and_classify_objects,
    get_detector,
)
from app.ml.loader import model_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = _BACKEND_DIR / "evaluation" / "real_world_detection_eval"
IMG_DIR = EVAL_DIR / "images"
ANN_PATH = EVAL_DIR / "annotations" / "annotations.json"
VIS_DIR = EVAL_DIR / "visualizations"
REPORT_PATH = EVAL_DIR / "evaluation_report.json"

VIS_DIR.mkdir(parents=True, exist_ok=True)


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_process_memory_mb() -> float:
    """Get current process working set size in MB on Windows."""
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
        GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
        GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return round(counters.WorkingSetSize / (1024.0 * 1024.0), 2)
    except Exception:
        pass
    return 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute 11-point interpolated Average Precision (Pascal VOC standard)."""
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p_candidates = precisions[recalls >= t]
        p = np.max(p_candidates) if len(p_candidates) > 0 else 0.0
        ap += p / 11.0
    return float(ap)


def render_comparison_visualization(
    original_img: Image.Image,
    gt_boxes: List[Dict[str, Any]],
    pred_objects: List[Any],
    save_path: Path,
    title_text: str = "",
):
    """
    Render a clean side-by-side comparison:
    Left: Ground Truth Annotations (Green boxes)
    Right: Model Predictions (Category-colored boxes with ID, class, confidence)
    """
    w, h = original_img.size
    side_by_side = Image.new("RGB", (w * 2 + 20, h + 45), color=(248, 250, 252))
    draw = ImageDraw.Draw(side_by_side)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Header Titles
    draw.text((15, 12), f"GROUND TRUTH: {title_text}", fill=(15, 23, 42), font=font)
    draw.text((w + 35, 12), "PREDICTED DETECTIONS & CLASSIFICATION", fill=(15, 23, 42), font=font)

    # 1. Left: Ground Truth
    gt_panel = original_img.copy()
    draw_gt = ImageDraw.Draw(gt_panel)
    for gt in gt_boxes:
        box = [gt["x1"], gt["y1"], gt["x2"], gt["y2"]]
        draw_gt.rectangle(box, outline=(22, 163, 74), width=3)
        lbl = f"GT: {gt['category'].upper()}"
        tb = draw_gt.textbbox((box[0], box[1]), lbl, font=font) if font else (box[0], box[1], box[0]+70, box[1]+15)
        tw = tb[2] - tb[0] + 6
        th = tb[3] - tb[1] + 4
        draw_gt.rectangle([box[0], max(0, box[1] - th), box[0] + tw, box[1]], fill=(22, 163, 74))
        draw_gt.text((box[0] + 3, max(0, box[1] - th) + 2), lbl, fill=(255, 255, 255), font=font)

    # 2. Right: Predicted Detections
    pred_panel = original_img.copy()
    draw_pred = ImageDraw.Draw(pred_panel)
    for obj in pred_objects:
        cat = obj.material_category.lower()
        rgb = CATEGORY_RGB.get(cat, (100, 116, 139))
        box = [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2]
        draw_pred.rectangle(box, outline=rgb, width=3)

        lbl = f"#{obj.object_id} {cat.capitalize()} ({obj.calibrated_confidence*100:.0f}%)"
        tb = draw_pred.textbbox((box[0], box[1]), lbl, font=font) if font else (box[0], box[1], box[0]+80, box[1]+15)
        tw = tb[2] - tb[0] + 8
        th = tb[3] - tb[1] + 4
        pill_y1 = max(0.0, box[1] - th - 2)
        draw_pred.rectangle([box[0], pill_y1, box[0] + tw, pill_y1 + th], fill=rgb)
        draw_pred.text((box[0] + 4, pill_y1 + 2), lbl, fill=(255, 255, 255), font=font)

    # Paste into side-by-side
    side_by_side.paste(gt_panel, (10, 35))
    side_by_side.paste(pred_panel, (w + 20, 35))
    side_by_side.save(save_path, format="JPEG", quality=90)


def evaluate_single_image(
    img_name: str,
    meta: Dict[str, Any],
    detector: WasteObjectDetector,
    iou_thresholds: Tuple[float, float] = (0.30, 0.50),
) -> Dict[str, Any]:
    """Evaluate detection on one image and compare with ground-truth."""
    img_path = IMG_DIR / img_name
    img = Image.open(img_path).convert("RGB")
    gt_boxes = meta.get("ground_truth_boxes", [])

    # Measure performance
    t_start = time.perf_counter()
    cpu_start = time.process_time()
    mem_before = get_process_memory_mb()

    # Proposal timing
    t0_prop = time.perf_counter()
    proposals = detector.detect_objects(img)
    t_prop = (time.perf_counter() - t0_prop) * 1000.0

    # Full pipeline with classification
    response = detect_and_classify_objects(
        pil_image=img,
        vit_model=model_registry.vit_model,
        vit_processor=model_registry.vit_processor,
        clip_model=model_registry.clip_model if model_registry.clip_loaded else None,
        clip_processor=model_registry.clip_processor if model_registry.clip_loaded else None,
    )
    t_total = (time.perf_counter() - t_start) * 1000.0
    cpu_time = (time.process_time() - cpu_start) * 1000.0
    mem_after = get_process_memory_mb()

    preds = response.objects
    num_gt = len(gt_boxes)
    num_preds = len(preds)

    # Save visual comparison
    vis_filename = f"eval_{Path(img_name).stem}.jpg"
    vis_path = VIS_DIR / vis_filename
    render_comparison_visualization(
        original_img=img,
        gt_boxes=gt_boxes,
        pred_objects=preds,
        save_path=vis_path,
        title_text=meta.get("condition", ""),
    )

    # IoU Matrix calculation (N_preds x M_gt)
    iou_matrix = np.zeros((num_preds, num_gt), dtype=np.float32)
    for i, p in enumerate(preds):
        p_box = (p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2)
        for j, g in enumerate(gt_boxes):
            g_box = (g["x1"], g["y1"], g["x2"], g["y2"])
            iou_matrix[i, j] = calculate_iou(p_box, g_box)

    results_by_thresh = {}

    for thresh in iou_thresholds:
        matched_gt = set()
        matched_preds = set()
        tp_pairs = []
        cls_correct = 0

        # Sort predictions by confidence
        sorted_pred_indices = sorted(range(num_preds), key=lambda i: preds[i].calibrated_confidence, reverse=True)

        for p_idx in sorted_pred_indices:
            best_gt_idx = -1
            best_iou = -1.0
            for g_idx in range(num_gt):
                if g_idx in matched_gt:
                    continue
                iou = iou_matrix[p_idx, g_idx]
                if iou >= thresh and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                matched_preds.add(p_idx)
                tp_pairs.append((p_idx, best_gt_idx, float(best_iou)))
                # Check category classification accuracy
                p_cat = preds[p_idx].material_category.lower()
                g_cat = gt_boxes[best_gt_idx]["category"].lower()
                if p_cat == g_cat:
                    cls_correct += 1

        tp = len(tp_pairs)
        fp = num_preds - tp
        fn = num_gt - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if num_gt == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if num_gt == 0 else 0.0)
        avg_iou = float(np.mean([p[2] for p in tp_pairs])) if tp_pairs else 0.0
        cls_acc = (cls_correct / tp) if tp > 0 else (1.0 if num_gt == 0 else 0.0)

        # Detect duplicate detections: predictions overlapping an already-matched GT box
        duplicates = 0
        for p_idx in range(num_preds):
            if p_idx in matched_preds:
                continue
            for g_idx in matched_gt:
                if iou_matrix[p_idx, g_idx] >= 0.25:
                    duplicates += 1
                    break

        # Detect incorrectly merged objects: 1 prediction overlapping multiple GT boxes
        merged_count = 0
        for p_idx in range(num_preds):
            overlapping_gts = [g_idx for g_idx in range(num_gt) if iou_matrix[p_idx, g_idx] >= 0.20]
            if len(overlapping_gts) >= 2:
                merged_count += len(overlapping_gts) - 1

        # Detect incorrectly split objects: multiple predictions matching subsets of 1 GT box
        split_count = 0
        for g_idx in range(num_gt):
            overlapping_preds = [p_idx for p_idx in range(num_preds) if iou_matrix[p_idx, g_idx] >= 0.20]
            if len(overlapping_preds) >= 2:
                split_count += len(overlapping_preds) - 1

        results_by_thresh[f"iou_{thresh:.2f}"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "avg_iou": round(avg_iou, 4),
            "classification_accuracy": round(cls_acc, 4),
            "missed_objects": fn,
            "duplicate_detections": duplicates,
            "incorrectly_merged_objects": merged_count,
            "incorrectly_split_objects": split_count,
        }

    return {
        "filename": img_name,
        "condition": meta.get("condition"),
        "description": meta.get("description"),
        "image_width": img.width,
        "image_height": img.height,
        "ground_truth_count": num_gt,
        "detected_count": num_preds,
        "visualization_path": str(vis_path.relative_to(_BACKEND_DIR)),
        "performance": {
            "proposal_time_ms": round(t_prop, 2),
            "total_inference_time_ms": round(t_total, 2),
            "cpu_time_ms": round(cpu_time, 2),
            "memory_working_set_mb": mem_after,
        },
        "metrics": results_by_thresh,
    }


def run_full_real_world_evaluation():
    """Run full validation suite on the 12 evaluation images."""
    logger.info("Initializing ML Model Registry...")
    model_registry.load()

    logger.info("Loading ground-truth annotations from %s...", ANN_PATH)
    with open(ANN_PATH, "r", encoding="utf-8") as f:
        annotations: Dict[str, Any] = json.load(f)

    detector = get_detector()

    image_results = []
    total_prop_time = 0.0
    total_inf_time = 0.0
    total_cpu_time = 0.0

    logger.info("Beginning real-world evaluation across %d images...", len(annotations))

    for img_name, meta in sorted(annotations.items()):
        logger.info("Evaluating [%s]: %s...", meta.get("condition"), img_name)
        res = evaluate_single_image(img_name, meta, detector)
        image_results.append(res)
        perf = res["performance"]
        total_prop_time += perf["proposal_time_ms"]
        total_inf_time += perf["total_inference_time_ms"]
        total_cpu_time += perf["cpu_time_ms"]

    n_images = len(image_results)
    avg_prop_time = round(total_prop_time / n_images, 2)
    avg_inf_time = round(total_inf_time / n_images, 2)
    avg_cpu_time = round(total_cpu_time / n_images, 2)

    # Compute overall aggregate metrics at IoU 0.30 and 0.50
    aggregate_metrics = {}
    for t_key in ["iou_0.30", "iou_0.50"]:
        tot_tp = sum(r["metrics"][t_key]["tp"] for r in image_results)
        tot_fp = sum(r["metrics"][t_key]["fp"] for r in image_results)
        tot_fn = sum(r["metrics"][t_key]["fn"] for r in image_results)
        tot_gt = sum(r["ground_truth_count"] for r in image_results)
        tot_pred = sum(r["detected_count"] for r in image_results)

        prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0

        all_ious = [
            r["metrics"][t_key]["avg_iou"]
            for r in image_results
            if r["metrics"][t_key]["tp"] > 0
        ]
        mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

        all_cls_acc = [
            r["metrics"][t_key]["classification_accuracy"]
            for r in image_results
            if r["metrics"][t_key]["tp"] > 0
        ]
        mean_cls_acc = float(np.mean(all_cls_acc)) if all_cls_acc else 0.0

        # Calculate mAP
        recalls_list = [r["metrics"][t_key]["recall"] for r in image_results]
        precisions_list = [r["metrics"][t_key]["precision"] for r in image_results]
        ap = compute_ap(np.array(recalls_list), np.array(precisions_list))

        aggregate_metrics[t_key] = {
            "total_ground_truth_objects": tot_gt,
            "total_predicted_objects": tot_pred,
            "true_positives": tot_tp,
            "false_positives": tot_fp,
            "false_negatives_missed": tot_fn,
            "detection_precision": round(prec, 4),
            "detection_recall": round(rec, 4),
            "mean_iou": round(mean_iou, 4),
            "mAP": round(ap, 4),
            "crop_classification_accuracy": round(mean_cls_acc, 4),
            "total_duplicate_detections": sum(r["metrics"][t_key]["duplicate_detections"] for r in image_results),
            "total_incorrectly_merged": sum(r["metrics"][t_key]["incorrectly_merged_objects"] for r in image_results),
            "total_incorrectly_split": sum(r["metrics"][t_key]["incorrectly_split_objects"] for r in image_results),
        }

    # Verify blank image invariant
    blank_result = next((r for r in image_results if r["condition"] == "blank_background"), None)
    blank_zero_verified = (blank_result is not None and blank_result["detected_count"] == 0)

    # Verify touching objects separation
    touch_result = next((r for r in image_results if r["condition"] == "touching_objects"), None)
    touching_separated = (touch_result is not None and touch_result["detected_count"] >= 2)

    final_report = {
        "evaluation_name": "EcoSort AI V2 Real-World Multi-Object Detection Validation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluation_images": n_images,
        "blank_image_zero_verified": blank_zero_verified,
        "touching_objects_reliably_separated": touching_separated,
        "performance_benchmarks": {
            "average_proposal_time_ms": avg_prop_time,
            "average_total_inference_time_ms": avg_inf_time,
            "average_cpu_time_ms": avg_cpu_time,
            "memory_working_set_mb": get_process_memory_mb(),
        },
        "aggregate_metrics": aggregate_metrics,
        "per_image_results": image_results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info("Evaluation complete! Report written to %s", REPORT_PATH)
    logger.info("Visualizations saved in %s", VIS_DIR)

    # Print summary table
    print("\n" + "=" * 80)
    print("REAL-WORLD MULTI-OBJECT DETECTION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Test Images: {n_images}")
    print(f"Blank Image Returns 0: {'YES (VERIFIED)' if blank_zero_verified else 'NO (FAILED)'}")
    print(f"Touching Objects Separated: {'YES (VERIFIED)' if touching_separated else 'NO (FAILED)'}")
    print(f"Average Inference Latency: {avg_inf_time} ms (Proposal: {avg_prop_time} ms)")
    print(f"Peak Working Set Memory: {get_process_memory_mb()} MB")
    print("-" * 80)
    for t_key in ["iou_0.30", "iou_0.50"]:
        agg = aggregate_metrics[t_key]
        print(f"[{t_key.upper()} MATCHING PROTOCOL]")
        print(f"  Detection Precision : {agg['detection_precision']*100:.2f}%")
        print(f"  Detection Recall    : {agg['detection_recall']*100:.2f}%")
        print(f"  Mean IoU            : {agg['mean_iou']*100:.2f}%")
        print(f"  mAP (Pascal VOC)    : {agg['mAP']*100:.2f}%")
        print(f"  Crop Classification : {agg['crop_classification_accuracy']*100:.2f}%")
        print(f"  Missed Objects (FN) : {agg['false_negatives_missed']}")
        print(f"  Duplicate Detections: {agg['total_duplicate_detections']}")
        print(f"  Merged / Split Boxes: {agg['total_incorrectly_merged']} merged, {agg['total_incorrectly_split']} split")
        print("-" * 80)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_full_real_world_evaluation()
