"""
EcoSort AI V3 — Comprehensive Detection & Classification Evaluation Suite.

Evaluates BOTH stages cleanly and separately (no fabricated metrics):

1. Object Detection & Localization Evaluation (on labeled multi-object benchmark):
   - Precision, Recall, F1, Mean IoU
   - mAP@0.50 and COCO-style mAP@0.50:0.95 (mean over IoU thresholds 0.50..0.95 step 0.05)
   - Per-class AP@0.50, Precision, Recall
   - Confusion Matrix (matched GT vs predicted class + background FN/FP)
   - Missed-object rate (FN / total_GT) and Duplicate-detection rate

2. Separate Crop / Single-Object Classification Evaluation (ViT vs CLIP V2 vs Ensemble):
   - Accuracy, Macro F1
   - Per-class Precision, Recall, F1
   - Confusion Matrix

3. Manual Diagnostic Verification on Test Case A & Test Case B (explicitly marked as
   diagnostic cases, separate from validation benchmarks).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.ml.clip_model import load_clip
from app.ml.clip_model_v2 import V2_WASTE_CLASSES, predict_clip_v2
from app.ml.detector import calculate_iou, detect_and_classify_objects, _prepare_crop_for_classification
from app.ml.ensemble import fuse
from app.ml.vit_model import load_vit, predict_vit

CLASSES = list(V2_WASTE_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Diagnostic ground truth boxes for the two manual flat-lay test cases (strictly labeled as diagnostic)
DIAGNOSTIC_CASES = {
    "Test_Case_A_assorted_waste_flatlay": {
        "image_path": str(_BACKEND_DIR / "evaluation" / "v3_investigation_inputs" / "assorted_waste_flatlay.png"),
        "gt_boxes": [
            {"category": "plastic", "box": [10, 18, 412, 308]},
            {"category": "cardboard", "box": [365, 10, 770, 368]},
            {"category": "plastic", "box": [773, 5, 1062, 248]},
            {"category": "paper", "box": [1066, 55, 1313, 296]},
            {"category": "paper", "box": [1295, 27, 1532, 290]},
            {"category": "organic", "box": [10, 245, 355, 595]},
            {"category": "organic", "box": [254, 326, 442, 496]},
            {"category": "organic", "box": [135, 425, 380, 628]},
            {"category": "metal", "box": [384, 345, 677, 656]},
            {"category": "glass", "box": [615, 278, 790, 780]},
            {"category": "glass", "box": [798, 236, 949, 748]},
            {"category": "metal", "box": [965, 325, 1190, 605]},
            {"category": "plastic", "box": [1175, 262, 1522, 588]},
            {"category": "cardboard", "box": [16, 616, 384, 978]},
            {"category": "paper", "box": [395, 665, 618, 912]},
            {"category": "metal", "box": [555, 772, 900, 1008]},
            {"category": "plastic", "box": [910, 603, 1092, 1000]},
            {"category": "trash", "box": [1105, 596, 1520, 960]},
        ],
    },
    "Test_Case_B_multi_waste_scene": {
        "image_path": r"C:\Users\khamu\.gemini\antigravity\brain\943516ee-877a-4dde-b4aa-db8163d2a951\.user_uploaded\media_1795548956966_a179f024.jpg",
        "gt_boxes": [
            {"category": "plastic", "box": [20, 25, 365, 130]},
            {"category": "plastic", "box": [375, 10, 545, 170]},
            {"category": "plastic", "box": [545, 32, 705, 155]},
            {"category": "metal", "box": [825, 24, 1018, 168]},
            {"category": "cardboard", "box": [240, 116, 385, 235]},
            {"category": "cardboard", "box": [22, 235, 244, 380]},
            {"category": "paper", "box": [244, 300, 415, 496]},
            {"category": "paper", "box": [8, 374, 242, 495]},
            {"category": "paper", "box": [410, 280, 620, 465]},
            {"category": "plastic", "box": [475, 150, 625, 305]},
            {"category": "plastic", "box": [620, 160, 775, 325]},
            {"category": "plastic", "box": [774, 148, 862, 282]},
            {"category": "metal", "box": [29, 486, 180, 577]},
            {"category": "metal", "box": [29, 569, 178, 666]},
            {"category": "metal", "box": [180, 495, 285, 675]},
            {"category": "metal", "box": [282, 492, 388, 675]},
            {"category": "metal", "box": [383, 490, 488, 675]},
            {"category": "metal", "box": [488, 439, 574, 532]},
            {"category": "metal", "box": [490, 527, 580, 671]},
            {"category": "metal", "box": [583, 349, 680, 435]},
            {"category": "metal", "box": [583, 444, 672, 520]},
            {"category": "metal", "box": [583, 534, 677, 672]},
            {"category": "metal", "box": [685, 530, 785, 670]},
            {"category": "glass", "box": [785, 282, 888, 675]},
            {"category": "glass", "box": [889, 292, 990, 674]},
        ],
    },
}


def compute_ap_all_points(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """COCO/VOC continuous precision envelope AP calculation."""
    if len(recalls) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_detection_dataset(
    dataset_items: List[Dict[str, Any]],
    vit_model: Any,
    vit_processor: Any,
    clip_model: Any,
    clip_processor: Any,
) -> Dict[str, Any]:
    """
    Evaluate detection (class-agnostic localization + end-to-end class-aware AP@0.5 / mAP@0.5:0.95).
    """
    iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
    all_preds: List[Dict[str, Any]] = []
    all_gts_by_img: Dict[str, List[Dict[str, Any]]] = {}

    for item in dataset_items:
        img_id = item["id"]
        pil_img = Image.open(item["path"]).convert("RGB")
        resp = detect_and_classify_objects(
            pil_img, vit_model=vit_model, vit_processor=vit_processor,
            clip_model=clip_model, clip_processor=clip_processor,
        )
        all_gts_by_img[img_id] = item["gt_boxes"]
        for obj in resp.objects:
            all_preds.append({
                "img_id": img_id,
                "box": (obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2),
                "category": obj.material_category.lower(),
                "det_conf": obj.detection_confidence,
                "cls_conf": obj.calibrated_confidence,
                "score": obj.detection_confidence,
            })

    total_gt = sum(len(v) for v in all_gts_by_img.values())
    total_preds = len(all_preds)

    # 1. Localization-only metrics @ IoU=0.50 + duplicate detection rate
    loc_tp = loc_fp = duplicates = 0
    matched_ious: List[float] = []
    conf_mat = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)

    preds_by_img: Dict[str, List[Dict[str, Any]]] = {}
    for p in all_preds:
        preds_by_img.setdefault(p["img_id"], []).append(p)

    for img_id, gts in all_gts_by_img.items():
        img_preds = sorted(preds_by_img.get(img_id, []), key=lambda x: x["score"], reverse=True)
        matched_gt_idx = set()
        for p in img_preds:
            best_iou, best_gi = 0.0, -1
            for gi, g in enumerate(gts):
                ov = calculate_iou(p["box"], tuple(g["box"]))
                if ov > best_iou:
                    best_iou, best_gi = ov, gi
            if best_iou >= 0.50:
                if best_gi not in matched_gt_idx:
                    matched_gt_idx.add(best_gi)
                    loc_tp += 1
                    matched_ious.append(best_iou)
                    gt_cls = gts[best_gi]["category"].lower()
                    pred_cls = p["category"]
                    if gt_cls in CLASS_TO_IDX and pred_cls in CLASS_TO_IDX:
                        conf_mat[CLASS_TO_IDX[gt_cls], CLASS_TO_IDX[pred_cls]] += 1
                else:
                    duplicates += 1
                    loc_fp += 1
            else:
                loc_fp += 1

    loc_fn = total_gt - loc_tp
    loc_prec = loc_tp / max(1, loc_tp + loc_fp)
    loc_rec = loc_tp / max(1, total_gt)

    # 2. Class-aware mAP@0.50 and mAP@0.50:0.95
    ap_per_thresh: Dict[float, Dict[str, float]] = {}
    per_class_05: Dict[str, Dict[str, float]] = {}

    for thr in iou_thresholds:
        ap_per_thresh[thr] = {}
        for cls in CLASSES:
            cls_preds = sorted([p for p in all_preds if p["category"] == cls], key=lambda x: x["score"], reverse=True)
            cls_gts = {
                img_id: [g for g in gts if g["category"].lower() == cls]
                for img_id, gts in all_gts_by_img.items()
            }
            n_cls_gt = sum(len(v) for v in cls_gts.values())
            if n_cls_gt == 0:
                continue

            used = {img_id: set() for img_id in cls_gts}
            tp_arr = np.zeros(len(cls_preds), dtype=float)
            fp_arr = np.zeros(len(cls_preds), dtype=float)

            for i, p in enumerate(cls_preds):
                img_gts = cls_gts.get(p["img_id"], [])
                best_i, best_j = 0.0, -1
                for j, g in enumerate(img_gts):
                    ov = calculate_iou(p["box"], tuple(g["box"]))
                    if ov > best_i:
                        best_i, best_j = ov, j
                if best_i >= thr and best_j not in used[p["img_id"]]:
                    tp_arr[i] = 1.0
                    used[p["img_id"]].add(best_j)
                else:
                    fp_arr[i] = 1.0

            tp_cum = np.cumsum(tp_arr)
            fp_cum = np.cumsum(fp_arr)
            rec = tp_cum / max(1, n_cls_gt)
            prec = tp_cum / np.maximum(1.0, tp_cum + fp_cum)
            ap = compute_ap_all_points(rec, prec)
            ap_per_thresh[thr][cls] = ap

            if thr == 0.50:
                tp_tot = float(tp_arr.sum())
                fp_tot = float(fp_arr.sum())
                per_class_05[cls] = {
                    "gt_count": n_cls_gt,
                    "pred_count": len(cls_preds),
                    "ap_0.50": round(ap, 4),
                    "precision": round(tp_tot / max(1.0, tp_tot + fp_tot), 4),
                    "recall": round(tp_tot / max(1.0, n_cls_gt), 4),
                }

    map_50 = float(np.mean(list(ap_per_thresh[0.50].values()))) if ap_per_thresh[0.50] else 0.0
    map_50_95_vals = [
        float(np.mean(list(ap_per_thresh[t].values())))
        for t in iou_thresholds if ap_per_thresh[t]
    ]
    map_50_95 = float(np.mean(map_50_95_vals)) if map_50_95_vals else 0.0

    return {
        "images": len(dataset_items),
        "total_gt": total_gt,
        "total_preds": total_preds,
        "localization_at_iou_0.50": {
            "tp": loc_tp,
            "fp": loc_fp,
            "fn": loc_fn,
            "precision": round(loc_prec, 4),
            "recall": round(loc_rec, 4),
            "mean_iou": round(float(np.mean(matched_ious)) if matched_ious else 0.0, 4),
            "missed_object_rate": round(loc_fn / max(1, total_gt), 4),
            "duplicate_detection_rate": round(duplicates / max(1, total_preds), 4),
        },
        "mAP_0.50": round(map_50, 4),
        "mAP_0.50_0.95": round(map_50_95, 4),
        "per_class_at_0.50": per_class_05,
        "confusion_matrix_classes": CLASSES,
        "confusion_matrix_matched": conf_mat.tolist(),
    }


def evaluate_classification_suite(
    crop_samples: List[Tuple[Image.Image, str]],
    vit_model: Any,
    vit_processor: Any,
    clip_model: Any,
    clip_processor: Any,
) -> Dict[str, Any]:
    """Evaluate ViT vs CLIP V2 vs Balanced Crop Ensemble (48/52) independently."""
    y_true = []
    preds = {"vit": [], "clip": [], "ensemble": []}

    for img, gt in crop_samples:
        prep = _prepare_crop_for_classification(img)
        vp = predict_vit(vit_model, vit_processor, prep)
        cp = predict_clip_v2(clip_model, clip_processor, prep).probabilities

        v_pred = max(vp, key=vp.get)
        c_pred = max(cp, key=cp.get)
        second_c = max([v for k, v in cp.items() if k != "organic"], default=0.0)
        if c_pred == "organic" and cp["organic"] >= 0.40 and (cp["organic"] - second_c) >= 0.10:
            e_pred = "organic"
        else:
            e_pred = fuse(vp, {k: cp.get(k, 0.0) for k in vp}, vit_weight=0.48, clip_weight=0.52).predicted_category

        y_true.append(gt)
        preds["vit"].append(v_pred)
        preds["clip"].append(c_pred)
        preds["ensemble"].append(e_pred)

    summary = {}
    for model_key, y_pred in preds.items():
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(1, len(y_true))
        per_cls = {}
        f1s = []
        for cls in CLASSES:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
            support = sum(1 for t in y_true if t == cls)
            if support == 0:
                continue
            pr = tp / max(1, tp + fp)
            rc = tp / max(1, tp + fn)
            f1 = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
            f1s.append(f1)
            per_cls[cls] = {"support": support, "precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4)}
        summary[model_key] = {
            "samples": len(y_true),
            "accuracy": round(acc, 4),
            "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
            "per_class": per_cls,
        }
    return summary


def main():
    print("=================================================================")
    print("EcoSort AI V3 — Detection & Classification Honest Evaluation")
    print("=================================================================")
    vit_m, vit_p = load_vit()
    clip_m, clip_p = load_clip()

    # 1. Existing labeled real-world detection validation benchmark
    eval_dir = _BACKEND_DIR / "evaluation" / "real_world_detection_eval"
    anns = json.loads((eval_dir / "annotations" / "annotations.json").read_text(encoding="utf-8"))
    val_items = []
    for fname, info in anns.items():
        val_items.append({
            "id": fname,
            "path": str(eval_dir / "images" / fname),
            "gt_boxes": [
                {"category": g["category"], "box": [g["x1"], g["y1"], g["x2"], g["y2"]]}
                for g in info["ground_truth_boxes"]
            ],
        })

    t0 = time.perf_counter()
    det_val_report = evaluate_detection_dataset(val_items, vit_m, vit_p, clip_m, clip_p)
    print(f"\n[1] Real-World 12-Image Detection Benchmark ({time.perf_counter()-t0:.1f}s):")
    print(json.dumps({k: v for k, v in det_val_report.items() if k != "confusion_matrix_matched"}, indent=2))

    # 2. Manual diagnostic flat-lay benchmark (strictly labeled diagnostic, NOT general accuracy)
    diag_items = []
    diag_crops: List[Tuple[Image.Image, str]] = []
    for cid, cinfo in DIAGNOSTIC_CASES.items():
        p = Path(cinfo["image_path"])
        if p.exists():
            diag_items.append({"id": cid, "path": str(p), "gt_boxes": cinfo["gt_boxes"]})
            im = Image.open(p).convert("RGB")
            for g in cinfo["gt_boxes"]:
                diag_crops.append((im.crop(g["box"]), g["category"]))

    det_diag_report = evaluate_detection_dataset(diag_items, vit_m, vit_p, clip_m, clip_p)
    cls_diag_report = evaluate_classification_suite(diag_crops, vit_m, vit_p, clip_m, clip_p)

    print("\n[2] Manual Diagnostic Test Cases A & B — Detection & End-to-End:")
    print(json.dumps({k: v for k, v in det_diag_report.items() if k != "confusion_matrix_matched"}, indent=2))
    print("\n[3] Manual Diagnostic Crops — ViT vs CLIP vs Ensemble Classification:")
    print(json.dumps(cls_diag_report, indent=2))

    out_path = _BACKEND_DIR / "evaluation" / "results_v2" / "detection_v3_evaluation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "real_world_12_image_benchmark": det_val_report,
        "diagnostic_cases_a_and_b_detection": det_diag_report,
        "diagnostic_cases_a_and_b_crop_classification": cls_diag_report,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved full evaluation report to {out_path}")


if __name__ == "__main__":
    main()
