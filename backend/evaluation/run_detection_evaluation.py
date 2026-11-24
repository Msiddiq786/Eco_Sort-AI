import sys
sys.path.insert(0, 'backend')

import os
import json
import time
import base64
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from app.ml.vit_model import load_vit
from app.ml.clip_model import load_clip
from app.ml.detector import WasteObjectDetector, detect_and_classify_objects, calculate_iou, calculate_iomin

# Ground Truth for test_image_waste_flatlay.jpg (13 physical objects)
GROUND_TRUTH_FLATLAY = [
    {"id": 1, "label": "cardboard", "box": [75, 125, 530, 435], "name": "Large Cardboard Packaging"},
    {"id": 2, "label": "paper", "box": [500, 345, 650, 490], "name": "Crumpled Paper (Middle)"},
    {"id": 3, "label": "metal", "box": [30, 450, 260, 765], "name": "Crushed Metal Can/Foil (Bottom Left)"},
    {"id": 4, "label": "metal", "box": [245, 520, 360, 765], "name": "Metal Can 1 (Orange)"},
    {"id": 5, "label": "metal", "box": [350, 520, 470, 765], "name": "Metal Can 2 (Purple)"},
    {"id": 6, "label": "metal", "box": [465, 500, 585, 765], "name": "Metal Can 3 (Gold)"},
    {"id": 7, "label": "metal", "box": [585, 490, 705, 765], "name": "Metal Can 4 (Green)"},
    {"id": 8, "label": "metal", "box": [705, 480, 860, 765], "name": "Metal Can 5 (Silver)"},
    {"id": 9, "label": "glass", "box": [890, 310, 1015, 775], "name": "Green Glass Bottle (Left)"},
    {"id": 10, "label": "glass", "box": [1010, 320, 1135, 775], "name": "Green Glass Bottle (Right)"},
    {"id": 11, "label": "plastic", "box": [880, 160, 990, 320], "name": "Clear Plastic Bottle (Top Right)"},
    {"id": 12, "label": "plastic", "box": [620, 30, 810, 190], "name": "Clear Plastic Tray (Top Middle)"},
    {"id": 13, "label": "plastic", "box": [930, 20, 1140, 190], "name": "Plastic Container / Tray (Top Far Right)"},
]

def main():
    print("=" * 70)
    print("EcoSort AI V2 — Multi-Object Detection Validation & Audit")
    print("=" * 70)

    # 1. Blank Image Verification
    print("\n--- TEST 1: Blank Image Rejection ---")
    detector = WasteObjectDetector()
    blank_white = Image.new("RGB", (400, 400), (255, 255, 255))
    blank_black = Image.new("RGB", (400, 400), (0, 0, 0))
    blank_gray = Image.new("RGB", (400, 400), (128, 128, 128))

    res_white = detector.detect_objects(blank_white)
    res_black = detector.detect_objects(blank_black)
    res_gray = detector.detect_objects(blank_gray)

    print(f"  White image: {len(res_white)} detections (Expected: 0)")
    print(f"  Black image: {len(res_black)} detections (Expected: 0)")
    print(f"  Gray image:  {len(res_gray)} detections (Expected: 0)")
    assert len(res_white) == 0 and len(res_black) == 0 and len(res_gray) == 0, "Blank image produced false detections!"
    print("  [PASS] All blank images produced exactly 0 detections.")

    # 2. Real Photograph Evaluation
    print("\n--- TEST 2: Evaluation on Original Test Photograph ---")
    img_path = "backend/evaluation/test_image_waste_flatlay.jpg"
    pil_img = Image.open(img_path).convert("RGB")
    w, h = pil_img.size
    print(f"  Loaded test image: {img_path} ({w}x{h})")

    # Load ViT and CLIP
    print("  Loading ViT and CLIP models...")
    t0 = time.time()
    vit_model, vit_proc = load_vit()
    clip_model, clip_proc = load_clip()
    print(f"  Models loaded in {time.time() - t0:.2f}s")

    # Run detection and classification
    print("  Running detect_and_classify_objects()...")
    t0 = time.time()
    detect_resp = detect_and_classify_objects(
        pil_img,
        vit_model=vit_model,
        vit_processor=vit_proc,
        clip_model=clip_model,
        clip_processor=clip_proc,
    )
    latency_ms = (time.time() - t0) * 1000.0
    print(f"  Pipeline completed in {latency_ms:.1f}ms. Total detected objects: {detect_resp.total_objects}")

    # Inspect each detected object
    print(f"\n{'ID':<4} {'Category':<12} {'Confidence':<12} {'Agreement':<10} {'Status':<16} {'Disposal':<26} {'Box [x1, y1, x2, y2]'}")
    print("-" * 115)
    
    det_boxes = []
    category_counts = {}
    review_count = 0
    agreement_count = 0

    for obj in detect_resp.objects:
        b = obj.bbox
        box_coords = [b.x1, b.y1, b.x2, b.y2]
        det_boxes.append((box_coords, obj))
        category_counts[obj.material_category] = category_counts.get(obj.material_category, 0) + 1
        if obj.confidence_status == "ambiguous":
            review_count += 1
        if obj.model_agreement:
            agreement_count += 1
        print(f"#{obj.object_id:<3} {obj.material_category:<12} {obj.calibrated_confidence*100:>5.1f}%      {str(obj.model_agreement):<10} {obj.confidence_status:<16} {obj.recycling_status:<26} [{b.x1:.0f}, {b.y1:.0f}, {b.x2:.0f}, {b.y2:.0f}]")

    print("-" * 115)
    print(f"Category summary: {category_counts}")
    print(f"Model agreement: {agreement_count}/{detect_resp.total_objects} ({agreement_count/max(1, detect_resp.total_objects)*100:.1f}%)")
    print(f"Requiring review: {review_count}/{detect_resp.total_objects} ({review_count/max(1, detect_resp.total_objects)*100:.1f}%)")

    # 3. Ground Truth Matching & Detection Metrics
    print("\n--- TEST 3: Ground Truth Evaluation Metrics ---")
    matched_gt = set()
    matched_dets = set()
    iou_scores = []
    correct_class_count = 0

    for g_idx, gt in enumerate(GROUND_TRUTH_FLATLAY):
        gt_box = gt["box"]
        best_det_idx = -1
        best_iou = 0.0

        for d_idx, (d_box, d_obj) in enumerate(det_boxes):
            iou = calculate_iou(tuple(d_box), tuple(gt_box))
            if iou > best_iou:
                best_iou = iou
                best_det_idx = d_idx

        if best_det_idx >= 0 and best_iou >= 0.30:
            matched_gt.add(g_idx)
            matched_dets.add(best_det_idx)
            iou_scores.append(best_iou)
            d_obj = det_boxes[best_det_idx][1]
            is_correct_cat = (d_obj.material_category.lower() == gt["label"].lower())
            if is_correct_cat:
                correct_class_count += 1
            print(f"  GT #{gt['id']:2d} ({gt['label'].upper():<9} - {gt['name']:<35}): MATCHED by Det #{d_obj.object_id} (IoU={best_iou:.2f}, Pred={d_obj.material_category.upper()}) {'[CORRECT]' if is_correct_cat else '[CAT MISMATCH]'}")
        else:
            print(f"  GT #{gt['id']:2d} ({gt['label'].upper():<9} - {gt['name']:<35}): MISSED (Best IoU={best_iou:.2f})")

    recall = len(matched_gt) / len(GROUND_TRUTH_FLATLAY)
    precision = len(matched_dets) / max(1, len(det_boxes))
    f1 = 2 * precision * recall / max(1e-5, precision + recall)
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    crop_acc = correct_class_count / max(1, len(matched_gt))

    print("\nQuantitative Metrics Summary:")
    print(f"  Total Ground Truth Objects: {len(GROUND_TRUTH_FLATLAY)}")
    print(f"  Total Detected Objects:     {len(det_boxes)}")
    print(f"  True Positives (IoU >= 0.30): {len(matched_gt)}")
    print(f"  Precision:                  {precision * 100:.1f}%")
    print(f"  Recall:                     {recall * 100:.1f}%")
    print(f"  F1 Score:                   {f1 * 100:.1f}%")
    print(f"  Mean IoU on Matched Items:  {mean_iou:.2f}")
    print(f"  Matched Crop Accuracy:      {crop_acc * 100:.1f}%")

    # 4. Save Before and After Visualizations
    out_dir = Path("backend/evaluation/detection_improvements")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save upgraded annotated image from API
    if detect_resp.annotated_image_base64:
        b64_str = detect_resp.annotated_image_base64
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        annotated_bytes = base64.b64decode(b64_str)
        annotated_path = out_dir / "after_upgraded_detection.jpg"
        with open(annotated_path, "wb") as f:
            f.write(annotated_bytes)
        print(f"\n  Saved updated detection visualization to: {annotated_path}")

    # Save Ground Truth Visualization
    gt_vis = np.array(pil_img)[:, :, ::-1].copy()
    for gt in GROUND_TRUTH_FLATLAY:
        box = gt["box"]
        cv2.rectangle(gt_vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(gt_vis, f"{gt['label'].upper()}: {gt['name']}", (box[0], max(18, box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.imwrite(str(out_dir / "ground_truth_reference.jpg"), gt_vis)
    print(f"  Saved ground truth reference to: {out_dir / 'ground_truth_reference.jpg'}")

    # Save JSON results
    results_json = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image": img_path,
        "image_size": [w, h],
        "total_ground_truth": len(GROUND_TRUTH_FLATLAY),
        "total_detected": len(det_boxes),
        "true_positives": len(matched_gt),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "crop_classification_accuracy": round(crop_acc, 4),
        "agreement_rate": round(agreement_count / max(1, len(det_boxes)), 4),
        "review_rate": round(review_count / max(1, len(det_boxes)), 4),
        "category_counts": category_counts,
        "objects": [
            {
                "id": o.object_id,
                "category": o.material_category,
                "confidence": o.calibrated_confidence,
                "models_agree": o.model_agreement,
                "status": o.confidence_status,
                "recycling_status": o.recycling_status,
                "box": [o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2],
            }
            for o in detect_resp.objects
        ]
    }
    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved evaluation metrics JSON to: {out_dir / 'evaluation_metrics.json'}")

if __name__ == "__main__":
    main()
