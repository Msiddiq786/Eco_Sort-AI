# EcoSort AI V2 — 7-Class Model Specification

## Model Status: AWAITING_TRAINING

This directory is designated for the EcoSort AI Version 2 model suite with 7 waste categories:
1. `cardboard` (class 0)
2. `glass` (class 1)
3. `metal` (class 2)
4. `paper` (class 3)
5. `plastic` (class 4)
6. `organic` (class 5)
7. `trash` (class 6)

### Protection Notice
- Version 1 model is actively being trained in `backend/models/vit/` on the 6-class TrashNet dataset.
- Per safety guidelines, Version 1 checkpoints (`backend/models/vit/best` and `latest`) will NOT be overwritten or touched.
- Version 2 training (`backend/training/train_vit_v2.py`) must be executed only AFTER Version 1 training has reached its 20 epochs or early stopping completion.

### Expected Directory Layout After V2 Training:
```
backend/models/vit_v2/
├── config_v2.yaml
├── v2_classes.json
├── best/
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   └── training_metadata.json
└── latest/
    ├── config.json
    ├── model.safetensors
    ├── preprocessor_config.json
    └── training_metadata.json
```
