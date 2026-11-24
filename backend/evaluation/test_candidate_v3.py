"""
Test Candidate V3 for Glass vs Plastic discrimination.
"""
from __future__ import annotations

import json
from pathlib import Path
import torch
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from PIL import Image

V2_CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "organic", "trash")
CLASS_TO_IDX = {c: i for i, c in enumerate(V2_CLASSES)}

CANDIDATE_PROMPTS_V3 = {
    "cardboard": [
        "a photo of a corrugated cardboard box",
        "a photo of brown corrugated cardboard packaging",
        "a photo of a cardboard shipping carton",
        "a photo of thick corrugated cardboard waste",
        "a photo of flattened brown cardboard box",
    ],
    "glass": [
        "a photo of a glass bottle or glass jar",
        "a photo of transparent glass container or glassware",
        "a photo of clear or green beverage glass bottle",
        "a photo of a wine bottle, beer bottle, or sauce jar made of glass",
        "a photo of heavy rigid glass container for glass recycling",
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
        "a photo of a plastic water bottle or plastic beverage bottle",
        "a photo of a flexible plastic bag or plastic wrapper",
        "a photo of a plastic jug, milk jug, or plastic tub",
        "a photo of a clear PET plastic drink bottle with plastic cap",
        "a photo of lightweight molded plastic packaging or container",
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


def run():
    root = Path(__file__).resolve().parent.parent
    val_dir = root / "data" / "val"
    device = torch.device("cpu")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    samples = []
    for cls in V2_CLASSES:
        folder = val_dir / cls
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in folder.glob(ext):
                samples.append((p, CLASS_TO_IDX[cls]))
    samples = sorted(samples, key=lambda x: str(x[0]))
    y_true = [s[1] for s in samples]

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

    # Encode V3 prompts
    prompt_list = []
    prompt_class_idx = []
    for c in V2_CLASSES:
        for t in CANDIDATE_PROMPTS_V3[c]:
            prompt_list.append(t)
            prompt_class_idx.append(CLASS_TO_IDX[c])

    inputs = processor(text=prompt_list, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        tf = model.get_text_features(**inputs)
        if hasattr(tf, "pooler_output"):
            tf = tf.pooler_output
        tf = tf / tf.norm(dim=-1, keepdim=True)

    # Centroids
    centroids = []
    for c in V2_CLASSES:
        c_idx = CLASS_TO_IDX[c]
        indices = [i for i, idx in enumerate(prompt_class_idx) if idx == c_idx]
        mean_embed = tf[indices].mean(dim=0)
        centroids.append(mean_embed / mean_embed.norm())
    centroids = torch.stack(centroids, dim=0)

    logit_scale = model.logit_scale.exp().clamp(max=100.0)
    sim = image_features @ centroids.T
    probs = torch.softmax(sim * logit_scale, dim=-1)
    preds = probs.argmax(dim=-1).cpu().tolist()

    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=list(range(7)))

    c_card = CLASS_TO_IDX["cardboard"]
    c_pap = CLASS_TO_IDX["paper"]
    c_plas = CLASS_TO_IDX["plastic"]
    c_gla = CLASS_TO_IDX["glass"]

    print("Candidate V3 Results:")
    print(f"Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1:    {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"Weighted F1: {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
    print(f"Cardboard -> Paper: {cm[c_card, c_pap]}")
    print(f"Paper -> Cardboard: {cm[c_pap, c_card]}")
    print(f"Glass -> Plastic:   {cm[c_gla, c_plas]}")
    print(f"Plastic -> Glass:   {cm[c_plas, c_gla]}")
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, preds, labels=list(range(7)), zero_division=0)
    for i, c in enumerate(V2_CLASSES):
        print(f"  {c:<10}: Prec={prec[i]:.4f}, Rec={rec[i]:.4f}, F1={f1[i]:.4f} (sup={sup[i]})")


if __name__ == "__main__":
    run()
