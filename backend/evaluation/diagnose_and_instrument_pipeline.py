"""
EcoSort AI V2 — Multi-Object Detection Pipeline Diagnostic & Instrumentation Harness.

Instruments every stage of the proposal, suppression, merging, and classification pipeline:
1. Original image.
2. Classical edge-gradient proposals.
3. Morphological segmentation proposals.
4. YOLO proposals.
5. Proposals before merging.
6. Proposals after merging.
7. Proposals removed by IoU NMS.
8. Proposals removed by IoMin containment suppression.
9. Final accepted boxes.
10. Individual crop images.
11. ViT probabilities.
12. CLIP probabilities.
13. Final ensemble probabilities.
14. Final material classification.
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
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

from app.ml.vit_model import load_vit, predict_vit
from app.ml.clip_model import load_clip
from app.ml.clip_model_v2 import predict_clip_v2
from app.ml.ensemble import fuse
from app.ml.detector import (
    WasteObjectDetector,
    calculate_iou,
    calculate_iomin,
    CATEGORY_RGB,
    CATEGORY_HEX,
)

def draw_boxes_on_image(
    image: Image.Image,
    boxes: List[Dict[str, Any]],
    color: Tuple[int, int, int] = (0, 255, 0),
    title: str = "",
) -> Image.Image:
    """Draw bounding boxes with labels and diagnostic IDs."""
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for b in boxes:
        coords = b["coords"]
        lbl = b.get("label", "")
        box_id = b.get("id", "")
        box_color = b.get("color", color)

        draw.rectangle([coords[0], coords[1], coords[2], coords[3]], outline=box_color, width=2)
        txt = f"#{box_id} {lbl}"
        if "score" in b:
            txt += f" ({b['score']:.2f})"
        
        # Pill
        tb = draw.textbbox((coords[0], coords[1]), txt, font=font) if font else (coords[0], coords[1], coords[0]+80, coords[1]+15)
        tw = tb[2] - tb[0] + 6
        th = tb[3] - tb[1] + 4
        py1 = max(0.0, coords[1] - th - 2)
        py2 = py1 + th
        draw.rectangle([coords[0], py1, coords[0] + tw, py2], fill=box_color)
        draw.text((coords[0] + 3, py1 + 2), txt, fill=(255, 255, 255), font=font)

    if title:
        draw.rectangle([0, 0, len(title) * 8 + 16, 24], fill=(0, 0, 0))
        draw.text((8, 5), title, fill=(255, 255, 255), font=font)

    return vis


def instrument_image_pipeline(
    image_path: str,
    output_dir: Path,
    detector: WasteObjectDetector,
    vit_model: Any,
    vit_processor: Any,
    clip_model: Any,
    clip_processor: Any,
) -> Dict[str, Any]:
    """Execute complete 14-stage diagnostic instrumentation on an image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pil_img = Image.open(image_path).convert("RGB")
    w, h = pil_img.size
    total_area = float(w * h)

    # 1. Save original image
    pil_img.save(output_dir / "01_original_image.jpg")

    gray = pil_img.convert("L")
    arr_gray = np.array(gray, dtype=np.float32)
    gx = ndi.sobel(arr_gray, axis=1)
    gy = ndi.sobel(arr_gray, axis=0)
    mag = np.hypot(gx, gy)
    max_m = float(mag.max())

    min_box_area = total_area * 0.003
    max_box_area = total_area * 0.85  # diagnose full scope

    proposal_counter = 1
    raw_proposals: List[Dict[str, Any]] = []

    # 2. YOLO proposals
    yolo_proposals: List[Dict[str, Any]] = []
    if detector.yolo_model is not None:
        try:
            results = detector.yolo_model(pil_img, conf=0.10, verbose=False)
            for r in results:
                for b in r.boxes:
                    cls_id = int(b.cls[0].cpu().numpy())
                    cname = r.names.get(cls_id, f"obj_{cls_id}")
                    coords = [float(c) for c in b.xyxy[0].cpu().numpy()]
                    conf = float(b.conf[0].cpu().numpy())
                    area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                    prop = {
                        "id": f"Y{proposal_counter}",
                        "coords": coords,
                        "score": round(conf, 3),
                        "label": f"yolo_{cname}",
                        "source": "yolo",
                        "coco_class": cname,
                        "area": round(area, 1),
                        "area_ratio": round(area / total_area, 4),
                        "color": (59, 130, 246),  # Blue
                    }
                    proposal_counter += 1
                    yolo_proposals.append(prop)
                    raw_proposals.append(prop)
        except Exception as exc:
            print(f"YOLO error: {exc}")

    vis_yolo = draw_boxes_on_image(pil_img, yolo_proposals, title="Stage 4: YOLO Proposals")
    vis_yolo.save(output_dir / "04_yolo_proposals.jpg")

    # 3. Morphological & edge gradient proposals
    morph_proposals: List[Dict[str, Any]] = []
    edge_proposals: List[Dict[str, Any]] = []
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
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh
            if area < min_box_area or min(bw, bh) < 14:
                continue
            if (x1 <= 4 and x2 <= 40 and bh > h * 0.7) or (x1 >= w - 40 and x2 >= w - 4 and bh > h * 0.7):
                continue

            score = float(np.mean(mag[y1:y2, x1:x2])) / (max_m + 1e-5)
            prop = {
                "id": f"M{proposal_counter}",
                "coords": [float(x1), float(y1), float(x2), float(y2)],
                "score": round(0.50 + 0.35 * score, 3),
                "label": f"morph_p{int(p_ratio*100)}",
                "source": "morphology",
                "area": round(float(area), 1),
                "area_ratio": round(area / total_area, 4),
                "color": (234, 88, 12),  # Orange
            }
            proposal_counter += 1
            morph_proposals.append(prop)
            raw_proposals.append(prop)

    vis_edge = draw_boxes_on_image(pil_img, morph_proposals[:30], title="Stage 2 & 3: Morphological Proposals")
    vis_edge.save(output_dir / "02_03_morph_edge_proposals.jpg")

    # 4. Color-contrast saliency proposals
    contrast_proposals: List[Dict[str, Any]] = []
    cv_img = np.array(pil_img)
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
                prop = {
                    "id": f"C{proposal_counter}",
                    "coords": [float(x), float(y), float(x+bw), float(y+bh)],
                    "score": round(score, 3),
                    "label": f"contrast_t{int(thresh*100)}",
                    "source": "contrast_saliency",
                    "area": round(area, 1),
                    "area_ratio": round(area / total_area, 4),
                    "color": (16, 185, 129),  # Green
                }
                proposal_counter += 1
                contrast_proposals.append(prop)
                raw_proposals.append(prop)

    # 5. Proposals before merging
    vis_pre_merge = draw_boxes_on_image(pil_img, raw_proposals[:40], title=f"Stage 5: All Proposals Before Merging ({len(raw_proposals)})")
    vis_pre_merge.save(output_dir / "05_proposals_before_merging.jpg")

    # 6. Fragment Merging
    proposals_fmt = [(tuple(p["coords"]), p["score"], p["label"]) for p in raw_proposals]
    # Perform vertical can merging
    merged_fmt = []
    used_indices = set()
    n = len(raw_proposals)
    merged_proposals: List[Dict[str, Any]] = []

    for i in range(n):
        if i in used_indices:
            continue
        pA = raw_proposals[i]
        bA = list(pA["coords"])
        sA = pA["score"]
        lA = pA["label"]
        wA = bA[2] - bA[0]
        hA = bA[3] - bA[1]

        merged_with = []
        if wA <= 250 and hA <= 300:  # updated merge boundary
            for j in range(i + 1, n):
                if j in used_indices:
                    continue
                pB = raw_proposals[j]
                bB = pB["coords"]
                wB = bB[2] - bB[0]
                hB = bB[3] - bB[1]
                if wB <= 250 and hB <= 300:
                    xA = max(bA[0], bB[0])
                    xB = min(bA[2], bB[2])
                    inter_x = max(0.0, xB - xA)
                    min_w = min(wA, wB)
                    if min_w > 0 and (inter_x / min_w) > 0.65:
                        if bA[3] <= bB[1] and (bB[1] - bA[3]) <= 60:
                            bA = [min(bA[0], bB[0]), bA[1], max(bA[2], bB[2]), bB[3]]
                            sA = max(sA, pB["score"])
                            used_indices.add(j)
                            merged_with.append(pB["id"])
                        elif bB[3] <= bA[1] and (bA[1] - bB[3]) <= 60:
                            bA = [min(bA[0], bB[0]), bB[1], max(bA[2], bB[2]), bA[3]]
                            sA = max(sA, pB["score"])
                            used_indices.add(j)
                            merged_with.append(pB["id"])

        prop_m = {
            "id": pA["id"],
            "coords": bA,
            "score": sA,
            "label": f"{lA}_merged" if merged_with else lA,
            "source": pA["source"],
            "merged_with": merged_with,
            "color": (217, 70, 239) if merged_with else pA["color"],
        }
        merged_proposals.append(prop_m)
        used_indices.add(i)

    vis_post_merge = draw_boxes_on_image(pil_img, merged_proposals[:40], title=f"Stage 6: Proposals After Merging ({len(merged_proposals)})")
    vis_post_merge.save(output_dir / "06_proposals_after_merging.jpg")

    # 7 & 8: Suppression (IoU NMS and IoMin Containment)
    # Sort with preference for full deep objects and larger area
    sorted_merged = sorted(
        merged_proposals,
        key=lambda p: (
            (1.5 if p["source"] == "yolo" else 1.0) *
            p["score"] *
            (1.0 + 0.35 * min(1.0, (p["coords"][2] - p["coords"][0]) * (p["coords"][3] - p["coords"][1]) / 35000.0))
        ),
        reverse=True,
    )

    accepted_proposals: List[Dict[str, Any]] = []
    removed_by_iomin: List[Dict[str, Any]] = []
    removed_by_iou: List[Dict[str, Any]] = []

    for p in sorted_merged:
        boxA = tuple(p["coords"])
        suppressed_iomin = False
        suppressed_iou = False
        suppressing_box = None

        for acc in accepted_proposals:
            boxB = tuple(acc["coords"])
            iomin = calculate_iomin(boxA, boxB)
            iou = calculate_iou(boxA, boxB)

            if iomin > 0.45:
                suppressed_iomin = True
                suppressing_box = acc["id"]
                break
            elif iou > 0.30:
                suppressed_iou = True
                suppressing_box = acc["id"]
                break

        if suppressed_iomin:
            p_rem = dict(p)
            p_rem["removed_by"] = suppressing_box
            p_rem["color"] = (239, 68, 68)  # Red
            removed_by_iomin.append(p_rem)
        elif suppressed_iou:
            p_rem = dict(p)
            p_rem["removed_by"] = suppressing_box
            p_rem["color"] = (234, 179, 8)  # Yellow
            removed_by_iou.append(p_rem)
        else:
            accepted_proposals.append(p)

    vis_rem_iomin = draw_boxes_on_image(pil_img, removed_by_iomin[:30], title=f"Stage 8: Removed by IoMin Containment ({len(removed_by_iomin)})")
    vis_rem_iomin.save(output_dir / "08_removed_by_iomin.jpg")

    vis_rem_iou = draw_boxes_on_image(pil_img, removed_by_iou[:30], title=f"Stage 7: Removed by IoU NMS ({len(removed_by_iou)})")
    vis_rem_iou.save(output_dir / "07_removed_by_iou.jpg")

    # 9. Final Accepted Boxes
    final_accepted = accepted_proposals[:15]
    vis_accepted = draw_boxes_on_image(pil_img, final_accepted, title=f"Stage 9: Final Accepted Boxes ({len(final_accepted)})")
    vis_accepted.save(output_dir / "09_final_accepted_boxes.jpg")

    # 10. Individual Crop Images & 11-14 Classification
    crops_dir = output_dir / "10_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_objects: List[Dict[str, Any]] = []

    for idx, p in enumerate(final_accepted, 1):
        coords = p["coords"]
        pad_x = (coords[2] - coords[0]) * 0.03
        pad_y = (coords[3] - coords[1]) * 0.03
        cx1 = max(0, int(coords[0] - pad_x))
        cy1 = max(0, int(coords[1] - pad_y))
        cx2 = min(w, int(coords[2] + pad_x))
        cy2 = min(h, int(coords[3] + pad_y))

        crop = pil_img.crop((cx1, cy1, cx2, cy2))
        crop_path = crops_dir / f"crop_{idx:02d}_{p['id']}.jpg"
        crop.save(crop_path)

        # 11. ViT inference
        vit_probs = predict_vit(vit_model, vit_processor, crop)
        vit_pred = max(vit_probs, key=vit_probs.get)

        # 12. CLIP inference
        clip_res = predict_clip_v2(clip_model, clip_processor, crop)
        clip_probs = clip_res.probabilities if hasattr(clip_res, "probabilities") else clip_res
        clip_pred = clip_res.predicted_category if hasattr(clip_res, "predicted_category") else max(clip_probs, key=clip_probs.get)

        # 13. Ensemble fusion
        # Check organic bypass
        clip_org = clip_probs.get("organic", 0.0)
        sec_clip = max([v for k, v in clip_probs.items() if k != "organic"], default=0.0)
        is_organic_bypass = (clip_pred == "organic" and clip_org >= 0.40 and (clip_org - sec_clip) >= 0.10)

        if is_organic_bypass:
            final_cat = "organic"
            final_conf = clip_org
            fused_probs = {
                "cardboard": round(vit_probs.get("cardboard", 0.0) * 0.7 + clip_probs.get("cardboard", 0.0) * 0.3, 4),
                "glass": round(vit_probs.get("glass", 0.0) * 0.7 + clip_probs.get("glass", 0.0) * 0.3, 4),
                "metal": round(vit_probs.get("metal", 0.0) * 0.7 + clip_probs.get("metal", 0.0) * 0.3, 4),
                "paper": round(vit_probs.get("paper", 0.0) * 0.7 + clip_probs.get("paper", 0.0) * 0.3, 4),
                "plastic": round(vit_probs.get("plastic", 0.0) * 0.7 + clip_probs.get("plastic", 0.0) * 0.3, 4),
                "organic": round(clip_org, 4),
                "trash": round(vit_probs.get("trash", 0.0) * 0.35 + clip_probs.get("trash", 0.0) * 0.3, 4),
            }
            tot = sum(fused_probs.values())
            fused_probs = {k: round(v / tot, 4) for k, v in fused_probs.items()}
            models_agree = False
        else:
            fused = fuse(vit_probs, {k: clip_probs.get(k, 0.0) for k in vit_probs})
            final_cat = fused.predicted_category
            final_conf = fused.confidence
            fused_probs = {k: round(v, 4) for k, v in fused.class_probabilities.items()}
            models_agree = (vit_pred.lower() == clip_pred.lower())

        diag_obj = {
            "object_id": idx,
            "proposal_id": p["id"],
            "proposal_source": p["source"],
            "proposal_label": p["label"],
            "box": [round(c, 1) for c in coords],
            "crop_path": str(crop_path),
            "vit_prediction": vit_pred,
            "vit_confidence": round(vit_probs[vit_pred], 4),
            "vit_probabilities": {k: round(v, 4) for k, v in vit_probs.items()},
            "clip_prediction": clip_pred,
            "clip_confidence": round(clip_probs[clip_pred], 4),
            "clip_probabilities": {k: round(v, 4) for k, v in clip_probs.items()},
            "final_category": final_cat,
            "final_confidence": round(final_conf, 4),
            "ensemble_probabilities": fused_probs,
            "models_agree": models_agree,
            "is_organic_bypass": is_organic_bypass,
        }
        diagnostic_objects.append(diag_obj)

    # 14. Save Final Annotated Classification Image
    final_classified_boxes = [
        {
            "id": obj["object_id"],
            "coords": obj["box"],
            "label": f"{obj['final_category'].upper()} {obj['final_confidence']*100:.0f}%",
            "color": CATEGORY_RGB.get(obj["final_category"], (107, 114, 128)),
        }
        for obj in diagnostic_objects
    ]
    vis_classified = draw_boxes_on_image(pil_img, final_classified_boxes, title=f"Stage 14: Final Material Classification ({len(diagnostic_objects)})")
    vis_classified.save(output_dir / "14_final_material_classification.jpg")

    # Export Diagnostic Summary JSON
    summary_data = {
        "image_path": str(image_path),
        "image_size": [w, h],
        "stages": {
            "01_original": str(output_dir / "01_original_image.jpg"),
            "04_yolo_proposals_count": len(yolo_proposals),
            "02_03_morph_proposals_count": len(morph_proposals),
            "contrast_proposals_count": len(contrast_proposals),
            "05_total_raw_proposals": len(raw_proposals),
            "06_proposals_post_merge": len(merged_proposals),
            "07_removed_by_iou": len(removed_by_iou),
            "08_removed_by_iomin": len(removed_by_iomin),
            "09_final_accepted_count": len(final_accepted),
        },
        "accepted_objects": diagnostic_objects,
        "removed_iomin": removed_by_iomin,
        "removed_iou": removed_by_iou,
    }

    with open(output_dir / "diagnostic_trace.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[{image_path}] Diagnostics complete: {len(raw_proposals)} raw -> {len(final_accepted)} accepted.")
    return summary_data


def main():
    print("=" * 70)
    print("EcoSort AI V2 — Multi-Object Pipeline Diagnostic Instrumentation")
    print("=" * 70)

    detector = WasteObjectDetector()
    print(f"Loaded detector: status={detector.model_status}")

    print("Loading ViT and CLIP models...")
    vit_m, vit_p = load_vit()
    clip_m, clip_p = load_clip()
    print("Models loaded successfully.")

    test_images = [
        ("scene1_assorted_waste", "backend/evaluation/v3_investigation_inputs/assorted_waste_flatlay.png"),
        ("scene2_cardboard", "backend/evaluation/v3_investigation_inputs/scene2_cardboard.jpg"),
        ("scene3_glass_tumbler", "backend/evaluation/v3_investigation_inputs/scene3_tumbler.jpg"),
        ("scene4_water_bottle", "backend/evaluation/v3_investigation_inputs/scene4_bottle.jpg"),
        ("scene5_cans_and_plastic", "backend/evaluation/v3_investigation_inputs/scene5_cans_white.jpg"),
        ("scene6_original_flatlay", "backend/evaluation/test_image_waste_flatlay.jpg"),
    ]

    base_out = Path("backend/evaluation/v3_diagnostic_stages")
    for key, path in test_images:
        if os.path.exists(path):
            print(f"\nInstrumenting {key} ({path})...")
            out_dir = base_out / key
            instrument_image_pipeline(
                image_path=path,
                output_dir=out_dir,
                detector=detector,
                vit_model=vit_m,
                vit_processor=vit_p,
                clip_model=clip_m,
                clip_processor=clip_p,
            )
        else:
            print(f"Skipping {key}: file not found {path}")

    print("\nAll pipeline diagnostics completed!")

if __name__ == "__main__":
    main()
