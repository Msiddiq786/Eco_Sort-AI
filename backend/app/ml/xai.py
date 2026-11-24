"""
Explainable AI (XAI) module for EcoSort AI.

Implements:
  1. ViT Attention Rollout — produces a saliency map from multi-head
     self-attention weights across all transformer layers.
  2. CLIP similarity explanation — returns per-class cosine similarity
     scores so users understand CLIP's ranking.

References:
  - Abnar & Zuidema (2020). "Quantifying Attention Flow in Transformers."
    arXiv:2005.00928
  - Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale." arXiv:2010.11929
"""
from __future__ import annotations

import base64
import io
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ViT Attention Rollout
# ---------------------------------------------------------------------------

def _extract_attention_maps(
    model: Any,
    processor: Any,
    pil_image: Any,
) -> Tuple[List[np.ndarray], int, int]:
    """
    Run one forward pass on *pil_image* with output_attentions=True and
    return a list of attention tensors (one per transformer layer).

    Returns:
        (attention_list, num_patches_h, num_patches_w)

    Each element of attention_list has shape [num_heads, seq_len, seq_len]
    after squeezing the batch dimension.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for XAI attention extraction.")

    from app.ml.vit_model import DEVICE

    inputs = processor(images=pil_image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )

    # outputs.attentions: tuple of Tensors, one per layer
    # Each tensor shape: (batch=1, num_heads, seq_len, seq_len)
    attentions: List[np.ndarray] = []
    for attn_layer in outputs.attentions:
        # Remove batch dim → (num_heads, seq_len, seq_len)
        a = attn_layer.squeeze(0).cpu().float().numpy()
        attentions.append(a)

    # For ViT-Base-Patch16, image_size=224, patch_size=16 → 14×14 patches + 1 CLS
    # seq_len = num_patches + 1 = 197
    # Derive grid size from seq_len
    seq_len = attentions[0].shape[-1]
    num_patches = seq_len - 1  # subtract CLS token
    grid = int(math.sqrt(num_patches))

    return attentions, grid, grid


def _attention_rollout(
    attentions: List[np.ndarray],
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Compute attention rollout (Abnar & Zuidema 2020).

    Args:
        attentions: list of attention maps per layer, each shape (heads, S, S).
        head_fusion: how to fuse heads — 'mean', 'max', or 'min'.
        discard_ratio: fraction of lowest-attention tokens to zero out per layer.

    Returns:
        1-D rollout array of length (S-1) = num_patches.
        Values reflect the cumulative attention flow from CLS to each patch.
    """
    result = None
    seq_len = attentions[0].shape[-1]

    for attn in attentions:
        # Head fusion
        if head_fusion == "mean":
            attn_fused = np.mean(attn, axis=0)       # (S, S)
        elif head_fusion == "max":
            attn_fused = np.max(attn, axis=0)
        else:
            attn_fused = np.min(attn, axis=0)

        # Discard lowest-attended tokens (safely clamped index)
        flat = attn_fused.flatten()
        if len(flat) > 0:
            threshold_idx = min(max(int(discard_ratio * len(flat)), 0), len(flat) - 1)
            threshold = np.partition(flat, threshold_idx)[threshold_idx]
            attn_fused = attn_fused.copy()
            attn_fused[attn_fused < threshold] = 0.0

        # Add residual connection (identity matrix) and normalize rows
        I = np.eye(seq_len, dtype=np.float32)
        a = attn_fused + I
        row_sums = a.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        a = a / row_sums

        if result is None:
            result = a
        else:
            result = np.matmul(result, a)

    # CLS token's attention to each patch (index 0 is CLS)
    cls_to_patches = result[0, 1:]  # shape: (num_patches,)

    # Normalize to [0, 1]
    mn, mx = cls_to_patches.min(), cls_to_patches.max()
    if mx - mn > 1e-8:
        cls_to_patches = (cls_to_patches - mn) / (mx - mn)

    return cls_to_patches.astype(np.float32)


def generate_attention_heatmap(
    model: Any,
    processor: Any,
    pil_image: Any,
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
    overlay_alpha: float = 0.55,
    colormap: str = "jet",
) -> Tuple[str, np.ndarray]:
    """
    Generate a ViT attention rollout heatmap and overlay it on the original image.

    Args:
        model: loaded ViT model.
        processor: ViT image processor.
        pil_image: PIL Image of the uploaded waste item.
        head_fusion: 'mean' | 'max' | 'min'.
        discard_ratio: fraction of unimportant attention to zero out.
        overlay_alpha: blend ratio for the heatmap overlay.
        colormap: OpenCV colormap name (default: 'jet').

    Returns:
        (base64_png_string, raw_rollout_array)
        base64_png_string: a PNG image (base64-encoded) of the overlaid heatmap.
        raw_rollout_array: 2D numpy array of attention values (H_patch × W_patch).
    """
    if not HAS_PIL:
        raise RuntimeError("Pillow is required for XAI heatmap generation.")

    # --- 1. Extract attention maps from ViT ---
    attentions, grid_h, grid_w = _extract_attention_maps(model, processor, pil_image)

    # --- 2. Compute attention rollout ---
    rollout_1d = _attention_rollout(attentions, head_fusion=head_fusion, discard_ratio=discard_ratio)
    rollout_2d = rollout_1d.reshape(grid_h, grid_w)

    # --- 3. Resize rollout to original image size ---
    img_rgb = np.array(pil_image.convert("RGB"), dtype=np.uint8)
    img_h, img_w = img_rgb.shape[:2]

    if HAS_CV2:
        # High-quality Lanczos resize
        mask = cv2.resize(rollout_2d, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)
        # Apply colormap
        mask_uint8 = np.uint8(255 * mask)
        colormap_id = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
        heatmap_bgr = cv2.applyColorMap(mask_uint8, colormap_id)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend heatmap with original image
        blended = cv2.addWeighted(
            img_rgb.astype(np.float32), 1.0 - overlay_alpha,
            heatmap_rgb.astype(np.float32), overlay_alpha,
            0,
        ).clip(0, 255).astype(np.uint8)
        result_pil = Image.fromarray(blended)
    else:
        # Fallback: use Pillow only (less smooth but dependency-free)
        from PIL import ImageFilter
        from PIL import ImageEnhance

        # Resize to image size using PIL
        mask_pil = Image.fromarray(np.uint8(255 * rollout_2d), mode="L").resize(
            (img_w, img_h), Image.LANCZOS
        )
        # Gaussian blur for smoothness
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=2))
        # Create a simple green-to-red colormap using Pillow
        mask_arr = np.array(mask_pil)
        r = mask_arr.astype(np.float32)
        g = (255 - mask_arr).astype(np.float32)
        b = np.zeros_like(r)
        heatmap_arr = np.stack([r, g, b], axis=2).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_arr)
        # Blend
        orig_pil = pil_image.convert("RGB")
        result_pil = Image.blend(orig_pil, heatmap_pil, alpha=overlay_alpha)

    # --- 4. Encode result to base64 PNG ---
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")

    return b64_str, rollout_2d


# ---------------------------------------------------------------------------
# CLIP Explanation
# ---------------------------------------------------------------------------

def explain_clip_scores(
    clip_probs: Dict[str, float],
    top_k: int = 7,
) -> List[Dict[str, Any]]:
    """
    Return a ranked list of waste categories and their CLIP similarity scores.

    Args:
        clip_probs: dict mapping class name → normalized CLIP probability/score.
        top_k: number of top classes to return.

    Returns:
        List of dicts: [{"category": str, "score": float, "rank": int}, ...]
    """
    ranked = sorted(clip_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "category": cat,
            "score": round(score, 4),
            "percentage": round(score * 100, 2),
            "rank": i + 1,
        }
        for i, (cat, score) in enumerate(ranked)
    ]


def build_xai_explanation(
    predicted_category: str,
    confidence: float,
    vit_probs: Dict[str, float],
    clip_probs: Dict[str, float],
    ensemble_probs: Dict[str, float],
    attention_heatmap_b64: Optional[str],
    vit_weight: float = 0.70,
    clip_weight: float = 0.30,
) -> Dict[str, Any]:
    """
    Construct a structured XAI explanation payload.

    Returns a dict suitable for direct serialisation to the API response.
    """
    vit_ranked = sorted(vit_probs.items(), key=lambda x: x[1], reverse=True)
    clip_ranked = sorted(clip_probs.items(), key=lambda x: x[1], reverse=True)
    ens_ranked = sorted(ensemble_probs.items(), key=lambda x: x[1], reverse=True)

    top3_ensemble = [
        {"category": c, "score": round(s, 4), "percentage": round(s * 100, 2)}
        for c, s in ens_ranked[:3]
    ]

    vit_top1 = vit_ranked[0][0] if vit_ranked else predicted_category
    clip_top1 = clip_ranked[0][0] if clip_ranked else predicted_category
    models_agree = vit_top1 == clip_top1

    # Human-readable reasoning text
    if models_agree:
        reasoning = (
            f"Both the Vision Transformer and CLIP agree that this item is "
            f"<strong>{predicted_category}</strong>. "
            f"The ensemble confidence is {confidence * 100:.1f}%."
        )
    else:
        reasoning = (
            f"The Vision Transformer predicted <strong>{vit_top1}</strong> "
            f"while CLIP predicted <strong>{clip_top1}</strong>. "
            f"The weighted ensemble ({vit_weight * 100:.0f}% ViT + {clip_weight * 100:.0f}% CLIP) "
            f"resolved the final category as <strong>{predicted_category}</strong>."
        )

    return {
        "predicted_category": predicted_category,
        "confidence": round(confidence, 4),
        "reasoning_text": reasoning,
        "models_agree": models_agree,
        "attention_heatmap_base64": attention_heatmap_b64,
        "attention_method": "attention_rollout",
        "attention_note": (
            "Heatmap generated via Attention Rollout (Abnar & Zuidema, 2020). "
            "Bright regions indicate image patches that most influenced the ViT prediction."
        ),
        "vit_explanation": {
            "predicted": vit_top1,
            "confidence": round(vit_ranked[0][1] if vit_ranked else 0.0, 4),
            "weight_in_ensemble": vit_weight,
            "top3": [
                {"category": c, "score": round(s, 4), "percentage": round(s * 100, 2)}
                for c, s in vit_ranked[:3]
            ],
        },
        "clip_explanation": {
            "predicted": clip_top1,
            "confidence": round(clip_ranked[0][1] if clip_ranked else 0.0, 4),
            "weight_in_ensemble": clip_weight,
            "note": (
                "CLIP computes cosine similarity between the image embedding and "
                "text embeddings of each waste category prompt."
            ),
            "ranked_categories": explain_clip_scores(clip_probs),
        },
        "ensemble_explanation": {
            "formula": f"P_final(c) = {vit_weight}×P_ViT(c) + {clip_weight}×P_CLIP(c)",
            "top3": top3_ensemble,
        },
    }
