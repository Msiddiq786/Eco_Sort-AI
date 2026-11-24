"""
Benchmark CLIP V2 Scoring Approaches on EcoSort Validation Dataset (backend/data/val).

Compares:
- Approach A: Current global prompt softmax and category summation.
- Approach B: Class-centroid scoring (normalize individual text embeddings, average per class,
              normalize class centroid, compute similarity, scale by logit_scale, 7-class softmax).
- Approach C1: Mean prompt cosine similarity per class, scale by logit_scale, 7-class softmax.
- Approach C2: LogSumExp prompt logits per class (- log(K)), 7-class softmax.
- Approach C3: Max prompt similarity per class, scale by logit_scale, 7-class softmax.

Evaluates on all 7 classes:
- Overall Accuracy
- Macro F1
- Weighted F1
- Per-class precision, recall, F1
- Confusion matrix
- Cardboard vs Paper confusion
- Plastic vs Glass confusion
- Organic performance
- Trash performance
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clip_benchmark")

V2_CLASSES: tuple[str, ...] = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "organic",
    "trash",
)
CLASS_TO_IDX = {c: i for i, c in enumerate(V2_CLASSES)}

# Original prompt templates from clip_model_v2.py
ORIGINAL_PROMPTS: Dict[str, List[str]] = {
    "cardboard": [
        "a photo of a cardboard box or cardboard waste",
        "a photo of brown corrugated cardboard packaging",
        "a photo of cardboard shipping carton or box",
        "a photo of flattened cardboard for recycling",
    ],
    "glass": [
        "a photo of a glass bottle or glass container",
        "a photo of clear or colored beverage glass bottle",
        "a photo of a transparent glass jar or bottle",
        "a photo of clean container glass for recycling",
    ],
    "metal": [
        "a photo of a metal can or metallic object",
        "a photo of an aluminum soda can or tin food can",
        "a photo of scrap metal or aluminum beverage can",
        "a photo of a clean tin can or aluminum foil",
    ],
    "paper": [
        "a photo of a paper cup or disposable coffee cup",
        "a photo of paper packaging, paper bag, or kraft paper",
        "a photo of paper waste, newspaper, or paper sheets",
        "a photo of printed office paper, magazine, or document",
        "a photo of clean sheets of paper or newsprint",
        "a photo of dry paper for paper recycling",
    ],
    "plastic": [
        "a photo of a plastic bottle or plastic bag",
        "a photo of clear PET plastic drink bottle or container",
        "a photo of plastic jug, bottle, or packaging wrap",
        "a photo of a single-use plastic container or wrapper",
    ],
    "organic": [
        "a photo of a yellow or brown banana peel or banana skins",
        "a photo of an apple core, fruit core, and fruit scraps",
        "a photo of watermelon rind, melon slices, or melon peel scraps",
        "a photo of orange peel, citrus rinds, or lemon peel scraps",
        "a photo of raw vegetable scraps, peelings, and kitchen scraps",
        "a photo of carrot peels, carrot trimmings, and vegetable cuttings",
        "a photo of fruit leftovers, fruit peels, and fruit waste",
        "a photo of coffee grounds and used tea leaves",
        "a photo of eggshells, fruit rinds, or decaying organic food scraps",
        "a photo of compostable food scraps, kitchen waste, and organic leftovers",
        "a photo of biodegradable wet waste for composting",
        "a photo of green vegetable leaves, lettuce salad trimmings, and celery ends",
    ],
    "trash": [
        "a photo of general household landfill trash or refuse",
        "a photo of non-recyclable solid domestic garbage or dirty rubbish",
        "a photo of non-compostable, non-recyclable composite trash, sweepings, or litter",
        "a photo of disposable sanitary waste, dirty packaging, or mixed dry rubbish",
    ],
}


def load_val_images(val_dir: Path) -> List[Tuple[Path, int, str]]:
    samples = []
    for cls in V2_CLASSES:
        folder = val_dir / cls
        if not folder.is_dir():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in folder.glob(ext):
                samples.append((p, CLASS_TO_IDX[cls], cls))
    return sorted(samples, key=lambda x: str(x[0]))


def precompute_text_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    prompt_dict: Dict[str, List[str]],
    device: torch.device,
) -> Dict[str, Any]:
    """Precompute and cache text embeddings and class centroids."""
    prompt_list = []
    prompt_class_idx = []
    prompt_class_name = []
    class_prompt_counts = {}

    for c in V2_CLASSES:
        templates = prompt_dict[c]
        class_prompt_counts[c] = len(templates)
        for t in templates:
            prompt_list.append(t)
            prompt_class_idx.append(CLASS_TO_IDX[c])
            prompt_class_name.append(c)

    text_inputs = processor(text=prompt_list, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Class centroids (Approach B)
    centroids = []
    for c in V2_CLASSES:
        c_idx = CLASS_TO_IDX[c]
        indices = [i for i, idx in enumerate(prompt_class_idx) if idx == c_idx]
        class_embeds = text_features[indices]  # [K, D]
        mean_embed = class_embeds.mean(dim=0)
        norm_centroid = mean_embed / mean_embed.norm()
        centroids.append(norm_centroid)
    centroids = torch.stack(centroids, dim=0)  # [7, D]

    return {
        "text_features": text_features,  # [num_prompts, D]
        "prompt_class_idx": torch.tensor(prompt_class_idx, device=device),
        "centroids": centroids,  # [7, D]
        "logit_scale": model.logit_scale.exp().clamp(max=100.0),
        "class_prompt_counts": class_prompt_counts,
    }


def evaluate_batch_scoring(
    image_features: torch.Tensor,  # [B, D] normalized
    text_data: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Compute predictions for a batch of normalized image features using all approaches.
    Returns dictionary of probability tensors: [B, 7] for each approach.
    """
    text_features = text_data["text_features"]  # [num_prompts, D]
    centroids = text_data["centroids"]  # [7, D]
    prompt_class_idx = text_data["prompt_class_idx"]  # [num_prompts]
    logit_scale = text_data["logit_scale"]

    # --- Approach A: Current global prompt softmax and category summation ---
    prompt_sim = image_features @ text_features.T  # [B, num_prompts]
    prompt_logits = prompt_sim * logit_scale
    global_prompt_probs = torch.softmax(prompt_logits, dim=-1)  # [B, num_prompts]
    
    probs_A = torch.zeros(image_features.size(0), 7, device=image_features.device)
    for c_idx in range(7):
        mask = (prompt_class_idx == c_idx)
        probs_A[:, c_idx] = global_prompt_probs[:, mask].sum(dim=-1)
    probs_A = probs_A / probs_A.sum(dim=-1, keepdim=True)

    # --- Approach B: Class-Centroid scoring ---
    centroid_sim = image_features @ centroids.T  # [B, 7]
    centroid_logits = centroid_sim * logit_scale
    probs_B = torch.softmax(centroid_logits, dim=-1)  # [B, 7]

    # --- Approach C1: Mean prompt cosine similarity per class ---
    probs_C1 = torch.zeros(image_features.size(0), 7, device=image_features.device)
    for c_idx in range(7):
        mask = (prompt_class_idx == c_idx)
        probs_C1[:, c_idx] = prompt_sim[:, mask].mean(dim=-1)
    probs_C1 = torch.softmax(probs_C1 * logit_scale, dim=-1)

    # --- Approach C2: LogSumExp prompt logits per class (- log(K)) ---
    logits_C2 = torch.zeros(image_features.size(0), 7, device=image_features.device)
    for c_idx in range(7):
        mask = (prompt_class_idx == c_idx)
        k = mask.sum().float()
        logits_C2[:, c_idx] = torch.logsumexp(prompt_logits[:, mask], dim=-1) - torch.log(k)
    probs_C2 = torch.softmax(logits_C2, dim=-1)

    # --- Approach C3: Max prompt similarity per class ---
    logits_C3 = torch.zeros(image_features.size(0), 7, device=image_features.device)
    for c_idx in range(7):
        mask = (prompt_class_idx == c_idx)
        logits_C3[:, c_idx] = prompt_sim[:, mask].max(dim=-1).values
    probs_C3 = torch.softmax(logits_C3 * logit_scale, dim=-1)

    return {
        "A_global_softmax": probs_A,
        "B_class_centroid": probs_B,
        "C1_mean_sim": probs_C1,
        "C2_logsumexp": probs_C2,
        "C3_max_sim": probs_C3,
    }


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(7)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))

    per_class = {}
    for i, c in enumerate(V2_CLASSES):
        per_class[c] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    # Specific confusions requested
    c_cardboard = CLASS_TO_IDX["cardboard"]
    c_paper = CLASS_TO_IDX["paper"]
    c_plastic = CLASS_TO_IDX["plastic"]
    c_glass = CLASS_TO_IDX["glass"]

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "cardboard_true_pred_paper": int(cm[c_cardboard, c_paper]),
        "paper_true_pred_cardboard": int(cm[c_paper, c_cardboard]),
        "plastic_true_pred_glass": int(cm[c_plastic, c_glass]),
        "glass_true_pred_plastic": int(cm[c_glass, c_plastic]),
        "organic_f1": float(f1[CLASS_TO_IDX["organic"]]),
        "trash_f1": float(f1[CLASS_TO_IDX["trash"]]),
    }


def run_benchmark():
    root = Path(__file__).resolve().parent.parent
    val_dir = root / "data" / "val"
    assert val_dir.is_dir(), f"Validation dir not found: {val_dir}"

    samples = load_val_images(val_dir)
    logger.info("Loaded %d validation images across 7 classes.", len(samples))

    device = torch.device("cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    text_data_orig = precompute_text_features(model, processor, ORIGINAL_PROMPTS, device)

    # Precompute all image features in batches
    logger.info("Extracting CLIP image features for validation set...")
    all_image_features = []
    y_true = []
    batch_size = 32

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]
        pil_images = [Image.open(p).convert("RGB") for p, _, _ in batch_samples]
        inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            img_feats = model.get_image_features(**inputs)
            if hasattr(img_feats, "pooler_output"):
                img_feats = img_feats.pooler_output
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            all_image_features.append(img_feats)
        for _, y, _ in batch_samples:
            y_true.append(y)

    all_image_features = torch.cat(all_image_features, dim=0)
    logger.info("Extracted features shape: %s", all_image_features.shape)

    # Evaluate all approaches with original prompts
    results_orig = evaluate_batch_scoring(all_image_features, text_data_orig)

    benchmark_summary = {}
    for name, probs in results_orig.items():
        preds = probs.argmax(dim=-1).cpu().tolist()
        metrics = compute_metrics(y_true, preds)
        benchmark_summary[name] = metrics
        logger.info(
            "Approach [%s]: Acc=%.4f, Macro_F1=%.4f, Cardboard->Paper=%d, Paper->Cardboard=%d",
            name,
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["cardboard_true_pred_paper"],
            metrics["paper_true_pred_cardboard"],
        )

    out_file = root / "evaluation" / "results" / "clip_scoring_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    logger.info("Saved benchmark results to %s", out_file)
    return benchmark_summary


if __name__ == "__main__":
    run_benchmark()
