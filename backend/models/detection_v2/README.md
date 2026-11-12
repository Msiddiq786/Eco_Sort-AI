# EcoSort AI V2 — Dedicated Waste Object Detector

## Directory Specification: `backend/models/detection_v2/`

This directory houses the dedicated object detection model architecture and configurations for EcoSort AI V2.

### Status: AWAITING_TRAINING
- **Active V1 Training Safeguard**: The 6-class Vision Transformer fine-tuning process is currently active on CPU in another terminal. In accordance with critical project protection rules, heavy detector training must NOT run concurrently.
- **Model Checkpoints**: Trained weights will be saved to `weights/best.pt` and `weights/last.pt`.
- **Runtime Integration**: At inference time, `backend/app/ml/detector.py` loads `weights/best.pt` if present. If weights are not yet generated, the system uses a high-resolution multi-scale spatial peak detection pipeline to produce individual localized bounding boxes.

### Supported Classes:
0. Cardboard
1. Glass
2. Metal
3. Paper
4. Plastic
5. Organic
6. Trash
