# EcoSort AI — API Reference

> **Base URL:** `http://localhost:8000/api`  
> **OpenAPI Docs:** `http://localhost:8000/docs`

---

## Endpoints

### `GET /api/health`

Returns server and model load status.

**Response 200**
```json
{
  "status": "ok",
  "vit_loaded": false,
  "clip_loaded": false,
  "models_ready": false
}
```

---

### `GET /api/model-info`

Returns model configuration metadata and active loading statuses.

**Response 200**
```json
{
  "vit_checkpoint": "google/vit-base-patch16-224-in21k",
  "clip_checkpoint": "openai/clip-vit-base-patch32",
  "vit_model_name": "google/vit-base-patch16-224-in21k",
  "clip_model_name": "openai/clip-vit-base-patch32",
  "clip_enabled": true,
  "ensemble_vit_weight": 0.7,
  "ensemble_clip_weight": 0.3,
  "waste_classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
  "model_version": "1.0.0",
  "vit_loaded": false,
  "clip_loaded": false
}
```

---

### `POST /api/predict/clip` ⚠️ Requires loaded CLIP foundation model

Direct zero-shot classification using the CLIP foundation model (`openai/clip-vit-base-patch32`) with natural language prompts. Returns class probabilities without saving to the database or applying ensemble fusion.

**Request** — `multipart/form-data`

| Field | Type | Constraint |
|-------|------|-----------|
| `file` | `UploadFile` | JPEG / PNG / WebP / BMP, max 10 MB |

**Response 200**
```json
{
  "predicted_category": "cardboard",
  "confidence": 0.88412,
  "probabilities": {
    "cardboard": 0.88412,
    "glass": 0.02105,
    "metal": 0.03140,
    "paper": 0.04122,
    "plastic": 0.01521,
    "trash": 0.00700
  }
}
```

**Response 503** (Model Unloaded)
```json
{
  "detail": "CLIP foundation model is not loaded or is unavailable for inference. Ensure CLIP weights are available and restart the server."
}
```

---

### `POST /api/predict` ⚠️ Requires trained ViT checkpoint

Upload a waste image and receive the ensemble classification.

**Request** — `multipart/form-data`

| Field | Type | Constraint |
|-------|------|-----------|
| `file` | `UploadFile` | JPEG / PNG / WebP, max 10 MB |

**Response 200**
```json
{
  "id": 1,
  "filename": "a3f9b1c2_photo.jpg",
  "original_filename": "photo.jpg",
  "image_path": "uploads/a3f9b1c2_photo.jpg",
  "predicted_category": "cardboard",
  "confidence": 0.92,
  "vit_prediction": "cardboard",
  "clip_prediction": "cardboard",
  "class_probabilities": {
    "cardboard": 0.92,
    "glass": 0.02,
    "metal": 0.02,
    "paper": 0.02,
    "plastic": 0.01,
    "trash": 0.01
  },
  "recycling_recommendation": "Flatten cardboard boxes and place them in the recycling bin.",
  "model_version": "1.0",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Error responses**

| Code | Condition |
|------|-----------|
| 400 | File type not allowed, file too large, bad magic bytes |
| 422 | Image cannot be decoded |
| 503 | ViT model checkpoint not installed |

---

### `GET /api/history`

Paginated prediction history, newest first.

**Query parameters**

| Param | Default | Description |
|-------|---------|-------------|
| `page` | `1` | 1-based page number |
| `page_size` | `20` | Records per page (max 100) |
| `category` | — | Filter by predicted category |
| `search` | — | Substring match on `original_filename` |

**Response 200**
```json
{
  "items": [
    {
      "id": 1,
      "filename": "a3f9b1c2_photo.jpg",
      "original_filename": "photo.jpg",
      "image_path": "uploads/a3f9b1c2_photo.jpg",
      "predicted_category": "cardboard",
      "confidence": 0.92,
      "vit_prediction": "cardboard",
      "clip_prediction": "cardboard",
      "class_probabilities": {"cardboard": 0.92, "glass": 0.02, "...": "..."},
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20
}
```

---

### `DELETE /api/history/{id}`

Delete a prediction record by ID.

**Path parameter:** `id` — integer prediction ID

**Response 204** — no body on success  
**Response 404**
```json
{ "detail": "Prediction 42 not found." }
```

---

### `GET /api/stats`

Category distribution and recycling totals derived from the database. Returns zero / empty values when no predictions exist. Never returns synthetic data.

**Response 200**
```json
{
  "total_predictions": 100,
  "category_distribution": [
    {
      "category": "cardboard",
      "count": 40,
      "percentage": 40.0,
      "is_recyclable": true
    },
    {
      "category": "trash",
      "count": 10,
      "percentage": 10.0,
      "is_recyclable": false
    }
  ],
  "average_confidence": 0.8732,
  "recyclable_count": 90
}
```

---

### `GET /api/metrics` ⚠️ Requires trained ViT checkpoint

Returns model evaluation metrics loaded from the training artefact file. Returns `null` values when no evaluation has been run.

**Response 200**
```json
{
  "accuracy": null,
  "precision": null,
  "recall": null,
  "f1_score": null
}
```

---

### `GET /api/train-log` ⚠️ Requires trained ViT checkpoint

Returns epoch-by-epoch training loss and validation accuracy from the last training run.

**Response 200**
```json
{
  "epochs": [],
  "train_loss": [],
  "val_accuracy": []
}
```

---

## Recyclable Categories

| Category | Recyclable |
|----------|-----------|
| Cardboard | ✅ |
| Glass | ✅ |
| Metal | ✅ |
| Paper | ✅ |
| Plastic | ✅ |
| Trash | ❌ |

---

## Notes

- Binary image data is **never stored** in the database.
- Uploaded files are stored under `backend/uploads/` with UUID-prefixed safe filenames.
- Original filenames are stored in `original_filename` for display; the filesystem path uses the UUID-safe `filename`.
- Phase 2 extended the `HistoryItem` and `PredictResponse` schemas with `original_filename` and `image_path` fields.
