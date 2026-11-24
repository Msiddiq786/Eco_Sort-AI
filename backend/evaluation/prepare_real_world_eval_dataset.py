"""
Prepare Real-World Evaluation Dataset for Multi-Object Waste Detection.
Combines real photographic test images from TrashNet, organic waste, and real waste scenes.
"""
import json
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import scipy.ndimage as ndi

BASE_DIR = Path("backend")
DATA_DIR = BASE_DIR / "data" / "test"
EVAL_DIR = BASE_DIR / "evaluation" / "real_world_detection_eval"
IMG_DIR = EVAL_DIR / "images"
ANN_DIR = EVAL_DIR / "annotations"

IMG_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR.mkdir(parents=True, exist_ok=True)


def extract_object_patch(img_path):
    """Load image, segment main object, and return cropped object and its mask."""
    im = Image.open(img_path).convert("RGB")
    gray = im.convert("L")
    arr = np.array(gray, dtype=np.float32)
    bg_val = float(np.median(arr))
    diff = np.abs(arr - bg_val)
    mask = diff > 20
    clean = ndi.binary_closing(ndi.binary_opening(mask, np.ones((3,3))), np.ones((5,5)))
    labeled, _ = ndi.label(clean)
    slices = sorted([s for s in ndi.find_objects(labeled) if s], key=lambda s: (s[1].stop-s[1].start)*(s[0].stop-s[0].start), reverse=True)
    if slices:
        s = slices[0]
        y1, y2 = s[0].start, s[0].stop
        x1, x2 = s[1].start, s[1].stop
        return im.crop((x1, y1, x2, y2))
    return im.crop((int(im.width*0.1), int(im.height*0.1), int(im.width*0.9), int(im.height*0.9)))


annotations = {}

# 1. Isolated Metal Can (real photo)
im_metal = Image.open(DATA_DIR / "metal" / "metal11.jpg").convert("RGB")
# Crop center region to 400x350
crop_m = im_metal.crop((10, 10, 410, 360))
crop_m.save(IMG_DIR / "01_isolated_metal_can.jpg")
annotations["01_isolated_metal_can.jpg"] = {
    "condition": "isolated_object",
    "description": "Single isolated real reflective metal can",
    "width": crop_m.width,
    "height": crop_m.height,
    "ground_truth_boxes": [
        {"object_id": 1, "category": "metal", "x1": 20.0, "y1": 20.0, "x2": 210.0, "y2": 350.0}
    ]
}

# 2. Isolated Transparent Glass (real photo)
im_glass = Image.open(DATA_DIR / "glass" / "glass11.jpg").convert("RGB")
crop_g = im_glass.crop((40, 50, 480, 340))
crop_g.save(IMG_DIR / "02_isolated_transparent_glass.jpg")
annotations["02_isolated_transparent_glass.jpg"] = {
    "condition": "transparent_bottle",
    "description": "Single real transparent glass bottle on light surface",
    "width": crop_g.width,
    "height": crop_g.height,
    "ground_truth_boxes": [
        {"object_id": 1, "category": "glass", "x1": 25.0, "y1": 40.0, "x2": 410.0, "y2": 250.0}
    ]
}

# 3. Isolated Organic Waste (real photo)
im_org = Image.open(DATA_DIR / "organic" / "organic_102.jpg").convert("RGB")
im_org = im_org.resize((360, 260))
im_org.save(IMG_DIR / "03_isolated_organic_waste.jpg")
annotations["03_isolated_organic_waste.jpg"] = {
    "condition": "organic_waste",
    "description": "Real photographic organic food waste item",
    "width": im_org.width,
    "height": im_org.height,
    "ground_truth_boxes": [
        {"object_id": 1, "category": "organic", "x1": 30.0, "y1": 20.0, "x2": 330.0, "y2": 240.0}
    ]
}

# 4. Isolated Plastic Packaging / Wrapper (real photo)
im_plastic = Image.open(DATA_DIR / "plastic" / "plastic10.jpg").convert("RGB")
crop_p = im_plastic.crop((120, 80, 400, 300))
crop_p.save(IMG_DIR / "04_isolated_plastic_wrapper.jpg")
annotations["04_isolated_plastic_wrapper.jpg"] = {
    "condition": "plastic_wrapper",
    "description": "Real crinkled plastic wrapper / soft packaging",
    "width": crop_p.width,
    "height": crop_p.height,
    "ground_truth_boxes": [
        {"object_id": 1, "category": "plastic", "x1": 50.0, "y1": 40.0, "x2": 225.0, "y2": 180.0}
    ]
}

# 5. Two Separate Objects: Real Plastic Bottle + Real Metal Can
scene_2 = Image.new("RGB", (640, 360), color=(235, 235, 235))
p_patch = extract_object_patch(DATA_DIR / "plastic" / "plastic109.jpg").resize((180, 240))
m_patch = extract_object_patch(DATA_DIR / "metal" / "metal103.jpg").resize((180, 220))
scene_2.paste(p_patch, (60, 60))
scene_2.paste(m_patch, (380, 70))
scene_2.save(IMG_DIR / "05_two_separate_objects.jpg")
annotations["05_two_separate_objects.jpg"] = {
    "condition": "two_separate_objects",
    "description": "Real plastic bottle and real metal can separated horizontally",
    "width": 640,
    "height": 360,
    "ground_truth_boxes": [
        {"object_id": 1, "category": "plastic", "x1": 60.0, "y1": 60.0, "x2": 240.0, "y2": 300.0},
        {"object_id": 2, "category": "metal", "x1": 380.0, "y1": 70.0, "x2": 560.0, "y2": 290.0},
    ]
}

# 6. Three Separate Objects: Cardboard + Metal + Plastic
scene_3 = Image.new("RGB", (720, 320), color=(238, 238, 238))
cb_patch = extract_object_patch(DATA_DIR / "cardboard" / "cardboard102.jpg").resize((180, 200))
m2_patch = extract_object_patch(DATA_DIR / "metal" / "metal116.jpg").resize((150, 200))
p2_patch = extract_object_patch(DATA_DIR / "plastic" / "plastic102.jpg").resize((160, 210))
scene_3.paste(cb_patch, (40, 60))
scene_3.paste(m2_patch, (280, 60))
scene_3.paste(p2_patch, (500, 55))
scene_3.save(IMG_DIR / "06_three_separate_objects.jpg")
annotations["06_three_separate_objects.jpg"] = {
    "condition": "three_separate_objects",
    "description": "Three distinct real objects: cardboard box, metal can, plastic bottle",
    "width": 720,
    "height": 320,
    "ground_truth_boxes": [
        {"object_id": 1, "category": "cardboard", "x1": 40.0, "y1": 60.0, "x2": 220.0, "y2": 260.0},
        {"object_id": 2, "category": "metal", "x1": 280.0, "y1": 60.0, "x2": 430.0, "y2": 260.0},
        {"object_id": 3, "category": "plastic", "x1": 500.0, "y1": 55.0, "x2": 660.0, "y2": 265.0},
    ]
}

# 7. Touching Objects: Real Glass Bottle Touching Real Metal Can
scene_touch = Image.new("RGB", (560, 340), color=(236, 236, 236))
g_patch = extract_object_patch(DATA_DIR / "glass" / "glass109.jpg").resize((190, 250))
m3_patch = extract_object_patch(DATA_DIR / "metal" / "metal103.jpg").resize((180, 230))
# Touching at x = 250
scene_touch.paste(g_patch, (60, 45))
scene_touch.paste(m3_patch, (250, 55))
scene_touch.save(IMG_DIR / "07_touching_objects.jpg")
annotations["07_touching_objects.jpg"] = {
    "condition": "touching_objects",
    "description": "Real glass bottle and metal can physically touching at boundary",
    "width": 560,
    "height": 340,
    "ground_truth_boxes": [
        {"object_id": 1, "category": "glass", "x1": 60.0, "y1": 45.0, "x2": 250.0, "y2": 295.0},
        {"object_id": 2, "category": "metal", "x1": 250.0, "y1": 55.0, "x2": 430.0, "y2": 285.0},
    ]
}

# 8. Overlapping Objects: Real Plastic Wrapper Partially Over Real Cardboard
scene_overlap = Image.new("RGB", (540, 360), color=(235, 235, 235))
cb2_patch = extract_object_patch(DATA_DIR / "cardboard" / "cardboard10.jpg").resize((240, 240))
p3_patch = extract_object_patch(DATA_DIR / "plastic" / "plastic10.jpg").resize((180, 160))
scene_overlap.paste(cb2_patch, (80, 60))
# Overlapping by ~40px
scene_overlap.paste(p3_patch, (260, 110))
scene_overlap.save(IMG_DIR / "08_overlapping_objects.jpg")
annotations["08_overlapping_objects.jpg"] = {
    "condition": "overlapping_objects",
    "description": "Real plastic item partially occluding a cardboard item",
    "width": 540,
    "height": 360,
    "ground_truth_boxes": [
        {"object_id": 1, "category": "cardboard", "x1": 80.0, "y1": 60.0, "x2": 320.0, "y2": 300.0},
        {"object_id": 2, "category": "plastic", "x1": 260.0, "y1": 110.0, "x2": 440.0, "y2": 270.0},
    ]
}

# 9. Transparent Bottles Cluster: Glass Jar + Clear Plastic Bottle
scene_trans = Image.new("RGB", (520, 320), color=(238, 238, 238))
g2_patch = extract_object_patch(DATA_DIR / "glass" / "glass107.jpg").resize((170, 230))
p4_patch = extract_object_patch(DATA_DIR / "plastic" / "plastic109.jpg").resize((160, 220))
scene_trans.paste(g2_patch, (70, 45))
scene_trans.paste(p4_patch, (280, 50))
scene_trans.save(IMG_DIR / "09_transparent_bottles_cluster.jpg")
annotations["09_transparent_bottles_cluster.jpg"] = {
    "condition": "transparent_bottles",
    "description": "Cluster of transparent glass jar and clear plastic bottle",
    "width": 520,
    "height": 320,
    "ground_truth_boxes": [
        {"object_id": 1, "category": "glass", "x1": 70.0, "y1": 45.0, "x2": 240.0, "y2": 275.0},
        {"object_id": 2, "category": "plastic", "x1": 280.0, "y1": 50.0, "x2": 440.0, "y2": 270.0},
    ]
}

# 10. Reflective Metal Cans: Two reflective metal cans
scene_refl = Image.new("RGB", (540, 340), color=(235, 235, 235))
m4_patch = extract_object_patch(DATA_DIR / "metal" / "metal11.jpg").resize((160, 230))
m5_patch = extract_object_patch(DATA_DIR / "metal" / "metal103.jpg").resize((170, 220))
scene_refl.paste(m4_patch, (60, 55))
scene_refl.paste(m5_patch, (290, 60))
scene_refl.save(IMG_DIR / "10_reflective_metal_cans.jpg")
annotations["10_reflective_metal_cans.jpg"] = {
    "condition": "reflective_metal_cans",
    "description": "Two reflective beverage cans with metallic specular highlights",
    "width": 540,
    "height": 340,
    "ground_truth_boxes": [
        {"object_id": 1, "category": "metal", "x1": 60.0, "y1": 55.0, "x2": 220.0, "y2": 285.0},
        {"object_id": 2, "category": "metal", "x1": 290.0, "y1": 60.0, "x2": 460.0, "y2": 280.0},
    ]
}

# 11. Complex Household Waste Pile (User's real unannotated waste scene)
im_real = Image.open(BASE_DIR / "evaluation" / "real_scene_1.jpg").convert("RGB")
im_real.save(IMG_DIR / "11_complex_household_garbage.jpg")
annotations["11_complex_household_garbage.jpg"] = {
    "condition": "complex_household_garbage",
    "description": "Real cluttered floor scene with mixed cardboard, plastic containers, and wrappers",
    "width": im_real.width,
    "height": im_real.height,
    "ground_truth_boxes": [
        {"object_id": 1, "category": "cardboard", "x1": 190.0, "y1": 85.0, "x2": 255.0, "y2": 145.0},
        {"object_id": 2, "category": "cardboard", "x1": 180.0, "y1": 135.0, "x2": 265.0, "y2": 170.0},
        {"object_id": 3, "category": "plastic", "x1": 265.0, "y1": 145.0, "x2": 335.0, "y2": 185.0},
        {"object_id": 4, "category": "plastic", "x1": 220.0, "y1": 0.0, "x2": 265.0, "y2": 35.0},
        {"object_id": 5, "category": "cardboard", "x1": 10.0, "y1": 0.0, "x2": 65.0, "y2": 30.0},
        {"object_id": 6, "category": "trash", "x1": 310.0, "y1": 95.0, "x2": 355.0, "y2": 130.0},
    ]
}

# 12. Blank Background (Zero objects invariant test)
im_blank = Image.new("RGB", (500, 350), color=(235, 235, 235))
im_blank.save(IMG_DIR / "12_blank_background.jpg")
annotations["12_blank_background.jpg"] = {
    "condition": "blank_background",
    "description": "Completely empty background with zero waste objects",
    "width": 500,
    "height": 350,
    "ground_truth_boxes": []
}

with open(ANN_DIR / "annotations.json", "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2)

print(f"Successfully generated {len(annotations)} real-world evaluation images and ground-truth annotations.")
