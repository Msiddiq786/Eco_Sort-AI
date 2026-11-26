# EcoSort AI V3 — Multi-Object Waste Detection Dataset & Training Requirements

## 1. Why Generic Detectors and Classification Datasets Are Insufficient

An inspection of the existing checkpoints and datasets confirms the following structural gaps:

1. **Classification-only datasets (`backend/data/{train,val,test}`)** contain 2,927 single-object images across 7 classes (`cardboard`, `glass`, `metal`, `organic`, `paper`, `plastic`, `trash`), with **zero bounding-box or segmentation annotations**. They cannot train a multi-object detector directly.
2. **Pretrained COCO (`yolov8s.pt`, 80 classes)** lacks dedicated waste categories (`cardboard`, `paper`, `metal can`, `aluminum foil`, `styrofoam/foam`, `plastic wrapper`, `transparent tray`). It can localize classic `bottle`, `cup`, `bowl`, and produce (`banana`, `apple`, `broccoli`) but misses flat-lay cartons, crumpled paper, foil, and clear trays.
3. **Pretrained `turhancan97/yolov8-segment-trash-detection` (5 super-classes: `Glass`, `Metal`, `Paper`, `Plastic`, `Waste`)** generalizes across cluttered scenes for localization, but collapses `cardboard` + `paper` together, omits `organic` as a separate class, and does not cover expanded polystyrene (`foam`) packaging cleanly.
4. **Dedicated checkpoint `backend/models/detection_v2/weights/best.pt`** does not exist yet (`AWAITING_TRAINING`), because external COCO/TACO detection images and bounding-box labels (`backend/data/detection/`) have not been ingested into the workspace.

## 2. Required Multi-Object Detection Taxonomy & Target Coverage

To train a dedicated 7-class detector at `backend/models/detection_v2/weights/best.pt` while preserving V1/V2 ViT + CLIP crop classifiers cleanly separated from localization confidence:

| Class ID | EcoSort Material | Required Fine-Grained Visual Coverage (Mandatory Before Training) |
|---|---|---|
| `0` | `cardboard` | Corrugated shipping boxes, open/closed flaps, printed cartons, egg/meal cartons (visually separated from thin/kraft paper). |
| `1` | `glass` | Clear, green, and amber beverage bottles, wide-mouth glass jars, glassware under harsh studio/tabletop specular reflections. |
| `2` | `metal` | Upright & crushed aluminum beverage cans, ribbed tin food cans, oval pull-tab sardine/spam tins, crumpled aluminum foil, metal lids. |
| `3` | `paper` | Crumpled white office paper, brown kraft paper bags, crumpled brown packing paper, disposable paper cups, newspapers/flyers. |
| `4` | `plastic` | Clear/colored PET bottles (with caps/labels), crushed bottles, translucent trays/clamshells, colored plastic bowls, bags & wrappers. |
| `5` | `organic` | Banana peels, apple cores, leafy vegetable scraps (lettuce), citrus rinds, coffee grounds, kitchen food scraps (dense overlapping scenes). |
| `6` | `trash` | **Expanded polystyrene / styrofoam food trays and foam cups**, contaminated multi-layer snack pouches, composite non-recyclables. |

## 3. Dataset Sourcing & Acceptance Targets

1. **Primary Source:** TACO (Trash Annotations in Context, COCO format) mapped via `TACO_TO_ECOSORT_MAP` in `backend/training/prepare_detection_dataset.py`.
2. **Supplements Required for Missing Modes:**
   - **Styrofoam / Foam (`trash`):** Minimum 300 annotated instances of clean white polystyrene trays/cups so classifiers/detectors distinguish foam from white molded plastic (`#1/#2/#5`).
   - **Kraft & Crumpled Paper vs. Cardboard (`paper` vs `cardboard`):** Minimum 300 instances each of brown kraft paper bags, crumpled paper, and corrugated cartons annotated in multi-object flat-lay scenes.
   - **Organic Food Waste (`organic`):** Minimum 300 multi-object food scrap bounding boxes (TACO lacks dense fruit/vegetable scrap annotations).
   - **Transparent Plastic Containers (`plastic` vs `glass`):** Minimum 300 instances of clamshell trays, bowls, and clear PET bottles against white/wooden/dark surfaces.
3. **Split & Integrity Rules:**
   - Strict image-level **70% train / 15% val / 15% test** split with MD5 hash deduplication (zero leakage).
   - Minimum `300` annotated boxes per class across the detection dataset before evaluating mAP claims.
4. **Preparation & Training Commands:**
   ```bash
   python backend/training/prepare_detection_dataset.py --json <path/to/annotations.json> --img-dir <path/to/images> --export-yolo
   python backend/training/train_detector_v2.py --data backend/data/detection/data.yaml --epochs 50 --batch-size 16
   ```
