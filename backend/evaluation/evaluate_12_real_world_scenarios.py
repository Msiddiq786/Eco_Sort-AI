"""
EcoSort AI V2 — 12-Scenario Real-World Multi-Object Detection Evaluation.

Calculates:
- Precision, Recall, F1
- Mean IoU on matched objects
- False positive count, False negative count
- Duplicate detection rate
- Per-class crop classification accuracy
- Complete 7x7 confusion matrix
- Performance benchmarks (inference latency)
- Generates side-by-side ground truth & detection visualizations.
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'backend')

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.ml.vit_model import load_vit
from app.ml.clip_model import load_clip
from app.ml.detector import (
    WasteObjectDetector,
    detect_and_classify_objects,
    calculate_iou,
    CATEGORY_RGB,
)

CLASSES_7 = ["cardboard", "glass", "metal", "paper", "plastic", "organic", "trash"]

def evaluate_12_scenarios():
    print("=" * 75)
    print("EcoSort AI V2 — 12-Scenario Real-World Multi-Object Detection Benchmark")
    print("=" * 75)

    base_dir = Path("backend/evaluation/real_world_12_scenarios")
    ann_path = base_dir / "annotations.json"
    with open(ann_path, "r") as f:
        annotations = json.load(f)

    vis_dir = base_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ViT and CLIP models...")
    t_load = time.perf_counter()
    vit_m, vit_p = load_vit()
    clip_m, clip_p = load_clip()
    print(f"Models loaded in {time.perf_counter() - t_load:.2f}s")

    total_gt = 0
    total_det = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_duplicates = 0
    iou_scores = []
    correct_class_count = 0
    matched_count = 0

    confusion_matrix = {c_true: {c_pred: 0 for c_pred in CLASSES_7} for c_true in CLASSES_7}
    class_stats = {c: {"gt": 0, "detected_correct": 0, "detected_total": 0} for c in CLASSES_7}
    scenario_results = []
    latencies = []

    for filename, meta in annotations.items():
        img_path = base_dir / "images" / filename
        if not img_path.exists():
            print(f"Warning: Image {filename} not found.")
            continue

        pil_img = Image.open(img_path).convert("RGB")
        w, h = pil_img.size
        gt_boxes = meta.get("ground_truth_boxes", [])
        total_gt += len(gt_boxes)
        for g in gt_boxes:
            cat = g["category"].lower()
            if cat in class_stats:
                class_stats[cat]["gt"] += 1

        t0 = time.perf_counter()
        resp = detect_and_classify_objects(
            pil_image=pil_img,
            vit_model=vit_m,
            vit_processor=vit_p,
            clip_model=clip_m,
            clip_processor=clip_p,
        )
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        det_objects = resp.objects
        total_det += len(det_objects)

        # One-to-one matching
        matched_gt_indices = set()
        det_matches = {}

        # Sort detections by confidence descending
        sorted_dets = sorted(enumerate(det_objects), key=lambda x: x[1].calibrated_confidence, reverse=True)

        scenario_tp = 0
        scenario_dup = 0
        scenario_fp = 0

        for d_idx, d_obj in sorted_dets:
            d_box = (d_obj.bbox.x1, d_obj.bbox.y1, d_obj.bbox.x2, d_obj.bbox.y2)
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, g in enumerate(gt_boxes):
                g_box = tuple(g["box"])
                iou = calculate_iou(d_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= 0.30:
                if best_gt_idx not in matched_gt_indices:
                    # Valid True Positive
                    matched_gt_indices.add(best_gt_idx)
                    det_matches[d_idx] = (best_gt_idx, best_iou)
                    scenario_tp += 1
                    total_tp += 1
                    iou_scores.append(best_iou)

                    gt_cat = gt_boxes[best_gt_idx]["category"].lower()
                    pred_cat = d_obj.material_category.lower()

                    if gt_cat in confusion_matrix and pred_cat in confusion_matrix[gt_cat]:
                        confusion_matrix[gt_cat][pred_cat] += 1

                    if pred_cat in class_stats:
                        class_stats[pred_cat]["detected_total"] += 1

                    matched_count += 1
                    if gt_cat == pred_cat:
                        correct_class_count += 1
                        if gt_cat in class_stats:
                            class_stats[gt_cat]["detected_correct"] += 1
                else:
                    # Extra detection on already matched physical object = Duplicate
                    scenario_dup += 1
                    total_duplicates += 1
            else:
                scenario_fp += 1
                total_fp += 1

        scenario_fn = len(gt_boxes) - len(matched_gt_indices)
        total_fn += scenario_fn

        # Visualization
        vis_cv = np.array(pil_img)[:, :, ::-1].copy()
        # Draw GT in dashed/green
        for g in gt_boxes:
            gb = [int(v) for v in g["box"]]
            cv2.rectangle(vis_cv, (gb[0], gb[1]), (gb[2], gb[3]), (0, 255, 0), 2)
            cv2.putText(vis_cv, f"GT: {g['category'].upper()}", (gb[0], max(16, gb[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Draw Detections
        for d_idx, d_obj in enumerate(det_objects):
            b = [int(d_obj.bbox.x1), int(d_obj.bbox.y1), int(d_obj.bbox.x2), int(d_obj.bbox.y2)]
            cat = d_obj.material_category.lower()
            rgb = CATEGORY_RGB.get(cat, (107, 114, 128))
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.rectangle(vis_cv, (b[0], b[1]), (b[2], b[3]), bgr, 2)
            cv2.putText(vis_cv, f"#{d_obj.object_id} {cat.upper()} ({d_obj.calibrated_confidence*100:.0f}%)", (b[0], min(h - 6, b[3] + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, bgr, 1)

        vis_path = vis_dir / f"eval_{filename}"
        cv2.imwrite(str(vis_path), vis_cv)

        s_p = scenario_tp / max(1, scenario_tp + scenario_fp)
        s_r = scenario_tp / max(1, len(gt_boxes)) if len(gt_boxes) > 0 else (1.0 if scenario_fp == 0 else 0.0)

        scenario_results.append({
            "scenario": meta.get("category_group"),
            "image": filename,
            "gt_count": len(gt_boxes),
            "detected_count": len(det_objects),
            "tp": scenario_tp,
            "fp": scenario_fp,
            "fn": scenario_fn,
            "duplicates": scenario_dup,
            "precision": round(s_p, 3),
            "recall": round(s_r, 3),
            "latency_ms": round(lat, 1),
            "visualization": str(vis_path),
        })

        print(f"[{meta.get('category_group'):<30}] GT={len(gt_boxes):<2} Det={len(det_objects):<2} TP={scenario_tp} FP={scenario_fp} Dup={scenario_dup} ({lat:.1f}ms)")

    # Overall Summary
    overall_p = total_tp / max(1, total_tp + total_fp)
    overall_r = total_tp / max(1, total_gt)
    overall_f1 = (2 * overall_p * overall_r / (overall_p + overall_r)) if (overall_p + overall_r) > 0 else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    crop_acc = correct_class_count / max(1, matched_count)
    dup_rate = total_duplicates / max(1, total_det)
    avg_lat = float(np.mean(latencies)) if latencies else 0.0

    print("\n" + "=" * 75)
    print("GLOBAL 12-SCENARIO BENCHMARK METRICS:")
    print("=" * 75)
    print(f"Total Ground Truth Objects: {total_gt}")
    print(f"Total Generated Detections: {total_det}")
    print(f"True Positives (IoU >= 0.30): {total_tp}")
    print(f"False Positives:            {total_fp}")
    print(f"False Negatives:            {total_fn}")
    print(f"Duplicate Detections:       {total_duplicates} (Duplicate Rate: {dup_rate*100:.1f}%)")
    print(f"Precision:                  {overall_p*100:.1f}%")
    print(f"Recall:                     {overall_r*100:.1f}%")
    print(f"Detection F1:               {overall_f1*100:.1f}%")
    print(f"Mean IoU (Matched Items):   {mean_iou:.3f}")
    print(f"Matched Crop Accuracy:      {crop_acc*100:.1f}%")
    print(f"Average Inference Latency:  {avg_lat:.1f} ms / image")

    print("\n" + "-" * 75)
    print("7-CLASS CROP CLASSIFICATION CONFUSION MATRIX (Rows: True, Cols: Pred):")
    print("-" * 75)
    header = f"{'True \\ Pred':<12}" + "".join(f"{c[:7]:>9}" for c in CLASSES_7)
    print(header)
    for c_true in CLASSES_7:
        row = f"{c_true:<12}" + "".join(f"{confusion_matrix[c_true][c_pred]:>9}" for c_pred in CLASSES_7)
        print(row)

    # Save complete JSON report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": {
            "total_ground_truth": total_gt,
            "total_detections": total_det,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "duplicate_count": total_duplicates,
            "duplicate_rate": round(dup_rate, 4),
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "detection_f1": round(overall_f1, 4),
            "mean_iou": round(mean_iou, 4),
            "crop_accuracy": round(crop_acc, 4),
            "average_latency_ms": round(avg_lat, 2),
        },
        "confusion_matrix": confusion_matrix,
        "per_class_stats": class_stats,
        "scenarios": scenario_results,
    }

    report_path = base_dir / "evaluation_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull evaluation report saved to: {report_path}")

if __name__ == "__main__":
    evaluate_12_scenarios()
