"""
Experiment with Refined Prompts and Class-Centroid / Aggregation Scoring.
Evaluates on the exact same 439 validation images.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import torch
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from PIL import Image

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

# Candidate Set 1: Targeted Disambiguation Prompts
CANDIDATE_PROMPTS_V1: Dict[str, List[str]] = {
    "cardboard": [
        "a photo of corrugated cardboard box or cardboard packaging",
        "a photo of brown corrugated cardboard with visible fluting",
        "a photo of thick cardboard shipping box or cardboard carton",
        "a photo of flattened corrugated cardboard waste",
        "a photo of heavy cardboard box for cardboard recycling",
    ],
    "glass": [
        "a photo of a transparent glass bottle or glass beverage bottle",
        "a photo of a clear glass jar or colored glass container",
        "a photo of rigid glass beverage bottle or glass jar for recycling",
        "a photo of shiny transparent glass container or glassware",
    ],
    "metal": [
        "a photo of an aluminum soda can or beverage can",
        "a photo of a tin food can, steel can, or metal food container",
        "a photo of crumpled aluminum foil or metallic food packaging",
        "a photo of metallic scrap, tinplate, or aluminum can for recycling",
    ],
    "paper": [
        "a photo of printed office paper, newspaper, magazine, or book pages",
        "a photo of white office printer paper, notebook paper, or written documents",
        "a photo of thin paper sheets, envelopes, or loose leaf paper",
        "a photo of disposable paper cup, paper flyer, or paper receipts",
        "a photo of recyclable thin paper, pamphlets, or newsprint",
    ],
    "plastic": [
        "a photo of a plastic water bottle or plastic beverage bottle",
        "a photo of a plastic jug, plastic container, or plastic tub",
        "a photo of flexible plastic bag, plastic film, or wrapper",
        "a photo of molded plastic packaging or PET plastic bottle",
    ],
    "organic": [
        "a photo of fruit peels, banana skin, apple core, or fruit rinds",
        "a photo of raw vegetable scraps, carrot peels, and food cuttings",
        "a photo of food waste, kitchen scraps, and compostable organic leftovers",
        "a photo of coffee grounds, eggshells, or used tea leaves",
        "a photo of biodegradable wet kitchen food waste for composting",
        "a photo of decaying vegetable salad leaves and fruit leftovers",
    ],
    "trash": [
        "a photo of styrofoam cup, expanded polystyrene foam, or foam packaging",
        "a photo of dirty non-recyclable rubbish, broken items, or landfill garbage",
        "a photo of soiled multi-layer snack wrappers, foil chip bags, and candy wrappers",
        "a photo of non-recyclable composite trash, plastic straws, and disposable utensils",
        "a photo of discarded landfill rubbish and contaminated waste",
    ],
}

# Candidate Set 2: Balanced 5-prompt per class design
CANDIDATE_PROMPTS_V2: Dict[str, List[str]] = {
    "cardboard": [
        "a photo of a corrugated cardboard box",
        "a photo of brown corrugated cardboard packaging",
        "a photo of a cardboard shipping carton",
        "a photo of thick corrugated cardboard waste",
        "a photo of flattened brown cardboard box",
    ],
    "glass": [
        "a photo of a glass bottle or glass jar",
        "a photo of transparent glass container",
        "a photo of clear beverage glass bottle",
        "a photo of colored glass bottle or glass jar",
        "a photo of recyclable glass container",
    ],
    "metal": [
        "a photo of an aluminum soda can or drink can",
        "a photo of a tin food can or steel can",
        "a photo of metallic aluminum can or tin can",
        "a photo of scrap metal or aluminum foil",
        "a photo of metal food can for recycling",
    ],
    "paper": [
        "a photo of office paper, document, or printed sheet",
        "a photo of newspaper, magazine, or paper brochure",
        "a photo of thin sheets of white paper",
        "a photo of a paper cup or paper flyer",
        "a photo of recyclable thin paper waste",
    ],
    "plastic": [
        "a photo of a plastic bottle or plastic drink container",
        "a photo of a clear PET plastic water bottle",
        "a photo of a plastic jug or plastic container",
        "a photo of a disposable plastic wrapper or bag",
        "a photo of molded plastic packaging",
    ],
    "organic": [
        "a photo of banana peels, apple cores, and fruit scraps",
        "a photo of raw vegetable scraps and vegetable peelings",
        "a photo of compostable food scraps and kitchen waste",
        "a photo of organic fruit peels and food leftovers",
        "a photo of biodegradable wet food waste for compost",
    ],
    "trash": [
        "a photo of styrofoam cup or polystyrene foam packaging",
        "a photo of non-recyclable landfill garbage and rubbish",
        "a photo of dirty mixed trash and contaminated waste",
        "a photo of soiled snack packaging, chip bag, or sweet wrapper",
        "a photo of disposable sanitary waste or landfill rubbish",
    ],
}


def evaluate_prompts_and_methods(prompt_sets, image_features, y_true, model, processor, device):
    results = {}
    logit_scale = model.logit_scale.exp().clamp(max=100.0)

    for set_name, pdict in prompt_sets.items():
        # Precompute centroids and prompt embeddings
        prompt_list = []
        prompt_class_idx = []
        for c in V2_CLASSES:
            for t in pdict[c]:
                prompt_list.append(t)
                prompt_class_idx.append(CLASS_TO_IDX[c])
        
        inputs = processor(text=prompt_list, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_feats = model.get_text_features(**inputs)
            if hasattr(text_feats, "pooler_output"):
                text_feats = text_feats.pooler_output
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # Centroids
        centroids = []
        for c in V2_CLASSES:
            c_idx = CLASS_TO_IDX[c]
            indices = [i for i, idx in enumerate(prompt_class_idx) if idx == c_idx]
            mean_embed = text_feats[indices].mean(dim=0)
            centroids.append(mean_embed / mean_embed.norm())
        centroids = torch.stack(centroids, dim=0)

        # 1. Approach B (Centroid)
        sim_b = image_features @ centroids.T
        probs_b = torch.softmax(sim_b * logit_scale, dim=-1)
        preds_b = probs_b.argmax(dim=-1).cpu().tolist()

        # 2. Approach A (Global softmax + sum)
        sim_a = image_features @ text_feats.T
        global_p = torch.softmax(sim_a * logit_scale, dim=-1)
        probs_a = torch.zeros(image_features.size(0), 7, device=device)
        p_class_t = torch.tensor(prompt_class_idx, device=device)
        for c_idx in range(7):
            probs_a[:, c_idx] = global_p[:, p_class_t == c_idx].sum(dim=-1)
        probs_a = probs_a / probs_a.sum(dim=-1, keepdim=True)
        preds_a = probs_a.argmax(dim=-1).cpu().tolist()

        # 3. Approach C1 (Mean prompt sim per class -> 7-class softmax)
        sim_c1 = torch.zeros(image_features.size(0), 7, device=device)
        for c_idx in range(7):
            sim_c1[:, c_idx] = sim_a[:, p_class_t == c_idx].mean(dim=-1)
        probs_c1 = torch.softmax(sim_c1 * logit_scale, dim=-1)
        preds_c1 = probs_c1.argmax(dim=-1).cpu().tolist()

        # Compute metrics for each
        for m_name, preds, probs in [
            ("Approach_B_Centroid", preds_b, probs_b),
            ("Approach_A_GlobalSum", preds_a, probs_a),
            ("Approach_C1_MeanSim", preds_c1, probs_c1),
        ]:
            key = f"{set_name}__{m_name}"
            acc = accuracy_score(y_true, preds)
            macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
            prec, rec, f1, support = precision_recall_fscore_support(
                y_true, preds, labels=list(range(7)), zero_division=0
            )
            cm = confusion_matrix(y_true, preds, labels=list(range(7)))

            per_class = {
                V2_CLASSES[i]: {
                    "precision": float(prec[i]),
                    "recall": float(rec[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i in range(7)
            }

            c_card = CLASS_TO_IDX["cardboard"]
            c_pap = CLASS_TO_IDX["paper"]
            c_plas = CLASS_TO_IDX["plastic"]
            c_gla = CLASS_TO_IDX["glass"]

            results[key] = {
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "cardboard_true_pred_paper": int(cm[c_card, c_pap]),
                "paper_true_pred_cardboard": int(cm[c_pap, c_card]),
                "plastic_true_pred_glass": int(cm[c_plas, c_gla]),
                "glass_true_pred_plastic": int(cm[c_gla, c_plas]),
                "per_class": per_class,
                "confusion_matrix": cm.tolist(),
            }
            print(
                f"{key:<35} | Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f} | "
                f"Cardboard->Paper: {cm[c_card, c_pap]} | Paper->Cardboard: {cm[c_pap, c_card]} | "
                f"Glass->Plastic: {cm[c_gla, c_plas]} | Trash-F1: {f1[CLASS_TO_IDX['trash']]:.4f}"
            )

    return results


def main():
    root = Path(__file__).resolve().parent.parent
    val_dir = root / "data" / "val"
    device = torch.device("cpu")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # Load val images
    samples = []
    for cls in V2_CLASSES:
        folder = val_dir / cls
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in folder.glob(ext):
                samples.append((p, CLASS_TO_IDX[cls]))
    samples = sorted(samples, key=lambda x: str(x[0]))
    y_true = [s[1] for s in samples]

    # Precompute image features
    batch_size = 32
    all_feats = []
    for i in range(0, len(samples), batch_size):
        batch = [Image.open(s[0]).convert("RGB") for s in samples[i : i + batch_size]]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            f = model.get_image_features(**inputs)
            if hasattr(f, "pooler_output"):
                f = f.pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
            all_feats.append(f)
    image_features = torch.cat(all_feats, dim=0)

    prompt_sets = {
        "Candidate_V1": CANDIDATE_PROMPTS_V1,
        "Candidate_V2": CANDIDATE_PROMPTS_V2,
    }

    eval_results = evaluate_prompts_and_methods(prompt_sets, image_features, y_true, model, processor, device)

    out_file = root / "evaluation" / "results" / "clip_prompt_optimization_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    main()
