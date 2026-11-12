"""
Evaluate models on the held-out test set (ViT, CLIP, and Ensemble).

Usage:
    cd smart-waste-segregation
    python backend/evaluation/evaluate_model.py

Outputs:
    backend/evaluation/results/metrics.json
    backend/evaluation/results/model_comparison.json
    backend/evaluation/results/confusion_matrix.json
    backend/evaluation/results/confusion_matrix.png
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure backend root is on sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from evaluation.evaluate_all import RESULTS_DIR, WASTE_CLASSES, run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("evaluate_model")


def generate_confusion_matrix_plot(results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        cm = np.array(results["ensemble"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=str(cm[i, j]), va="center", ha="center", size="medium")

        fig.colorbar(cax)
        ax.set_xticks(range(len(WASTE_CLASSES)))
        ax.set_yticks(range(len(WASTE_CLASSES)))
        ax.set_xticklabels(WASTE_CLASSES, rotation=45, ha="left")
        ax.set_yticklabels(WASTE_CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("70:30 Ensemble Confusion Matrix (Test Set)")
        plt.tight_layout()

        out_path = RESULTS_DIR / "confusion_matrix.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved confusion matrix visualization to %s", out_path)
    except Exception as exc:
        logger.warning("Could not generate confusion matrix plot: %s", exc)


def main():
    logger.info("Starting EcoSort AI model evaluation...")
    results = run_evaluation()
    generate_confusion_matrix_plot(results)

    print("\n" + "=" * 70)
    print("ECOSORT AI — HELD-OUT TEST DATASET EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Model':<18} | {'Accuracy':<10} | {'Macro Prec':<12} | {'Macro Rec':<12} | {'Macro F1':<10}")
    print("-" * 70)
    for name, key in [("ViT (Fine-tuned)", "vit"), ("CLIP (Zero-Shot)", "clip"), ("70:30 Ensemble", "ensemble")]:
        m = results[key]
        print(f"{name:<18} | {m['accuracy']*100:>8.2f}% | {m['macro_precision']*100:>10.2f}% | {m['macro_recall']*100:>10.2f}% | {m['macro_f1']*100:>8.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
