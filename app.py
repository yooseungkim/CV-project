"""
Gradio Inference App for Attention-CBM Pipeline.

Provides an interactive interface for:
1. Uploading a dermoscopy image and running inference
2. Visualizing per-concept attention heatmaps
3. Human-in-the-loop: manually adjusting concept values and re-predicting

Usage:
    uv run python app.py --checkpoint checkpoints/resnet50/<checkpoint>.pth
"""

import argparse
import io
import json
import os
import math
from typing import Optional

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.models.cbm_factory import UniversalFlexibleCBM


# ---------------------------------------------------------------------------
# Globals (populated in main())
# ---------------------------------------------------------------------------
MODEL: Optional[UniversalFlexibleCBM] = None
DEVICE: torch.device = torch.device("cpu")
CONCEPT_NAMES: list[str] = []
CONCEPT_CONFIG: dict = {}
CONCEPT_GROUPS: list[dict] = []
TARGET_CLASSES: list[str] = []
NUM_CONCEPTS: int = 0
NUM_CLASSES: int = 1
USE_CONCEPT_GROUPS = True
DATASET_NAME = "derm7pt"

def is_group_exclusive(group_name: str) -> bool:
    """Helper to check if a specific concept group is mutually exclusive (Softmax).
    In hybrid mode, only specified groups use Softmax, while others use Sigmoid.
    """
    if isinstance(USE_CONCEPT_GROUPS, bool):
        return USE_CONCEPT_GROUPS
    if isinstance(USE_CONCEPT_GROUPS, set):
        return group_name in USE_CONCEPT_GROUPS
    return True

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ---------------------------------------------------------------------------
# Helper: Generate individual concept heatmaps
# ---------------------------------------------------------------------------
def _generate_single_heatmap(
    img_np: np.ndarray,
    heatmap: np.ndarray,
    concept_name: str,
    colormap: str = 'jet',
    alpha: float = 0.5
) -> Image.Image:
    """Generate a single concept heatmap overlay as a PIL Image."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # Min-Max normalize
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)
    else:
        heatmap = np.zeros_like(heatmap)

    ax.imshow(img_np)
    ax.imshow(heatmap, cmap=colormap, alpha=alpha)
    ax.set_title(concept_name, fontsize=11, fontweight='bold', pad=8)
    ax.axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    plt.close(fig)
    pil_img = Image.open(buf)
    pil_img.load()
    return pil_img


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# (generate_segmentation_overlay removed)


def _unnormalize_tensor(img_tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization and convert to numpy HWC [0,1]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def plot_concept_contributions(model, concept_logits_tensor, concept_names, target_class_idx):
    """
    Generate a matplotlib horizontal bar chart showing concept contributions for the predicted class.
    Returns:
        PIL.Image.Image of the plot or None
    """
    if model is None or len(concept_names) == 0:
        return None

    with torch.no_grad():
        if hasattr(model, "classifier_head") and hasattr(model.classifier_head, "conv1") and hasattr(model.classifier_head, "concept_gates"):
            # GatedSparseNAMHead case
            if getattr(model, 'use_multimodal', False):
                x = torch.cat([concept_logits_tensor, torch.zeros(concept_logits_tensor.size(0), 3, device=concept_logits_tensor.device)], dim=-1)
            else:
                x = concept_logits_tensor
            
            supervised_x = x[:, :model.classifier_head.num_concepts].unsqueeze(-1)
            h = F.relu(model.classifier_head.conv1(supervised_x))
            y = model.classifier_head.conv2(h)
            y = y.view(1, model.classifier_head.num_concepts, model.num_classes)
            gated_y = y * model.classifier_head.concept_gates.view(1, model.classifier_head.num_concepts, 1)
            contributions = gated_y[0, :, target_class_idx].cpu().numpy()
        elif hasattr(model, "classifier_head") and hasattr(model.classifier_head, "weight"):
            # Standard nn.Linear case
            weight = model.classifier_head.weight[target_class_idx].cpu().numpy()
            if getattr(model, 'use_multimodal', False):
                x = torch.cat([concept_logits_tensor, torch.zeros(concept_logits_tensor.size(0), 3, device=concept_logits_tensor.device)], dim=-1)
            else:
                x = concept_logits_tensor
            x = x[0, :weight.shape[0]].cpu().numpy()
            contributions = weight * x
        else:
            return None

    import numpy as np
    names = list(concept_names)
    if len(names) < len(contributions):
        extra_names = ["Age", "Sex (Male)", "Sex (Female)"]
        names += extra_names[:len(contributions) - len(names)]
        if len(names) < len(contributions):
            names += [f"Concept {i}" for i in range(len(names), len(contributions))]
    names = names[:len(contributions)]
        
    abs_contribs = np.abs(contributions)
    top_indices = np.argsort(abs_contribs)[-12:]
    
    plot_contribs = contributions[top_indices]
    plot_names = [names[idx] for idx in top_indices]
    
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    colors = ['#10b981' if c >= 0 else '#ef4444' for c in plot_contribs]
    
    y_pos = np.arange(len(plot_names))
    bars = ax.barh(y_pos, plot_contribs, color=colors, edgecolor='none', height=0.6)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_names, fontsize=8, fontweight='semibold')
    ax.axvline(0, color='#6b7280', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Contribution Score (Logit Scale)", fontsize=9, fontweight='semibold')
    
    class_name = TARGET_CLASSES[target_class_idx] if target_class_idx < len(TARGET_CLASSES) else f"Class {target_class_idx}"
    ax.set_title(f"Concept Contribution to: {class_name}", fontsize=10, fontweight='bold', pad=10)
    
    max_val = max(abs(plot_contribs)) if len(plot_contribs) > 0 else 0
    ax_offset = 0.02 * (max_val + 1e-5)
    for bar in bars:
        width = bar.get_width()
        if width >= 0:
            ax.text(width + ax_offset, bar.get_y() + bar.get_height()/2, f"+{width:.2f}", 
                    va='center', ha='left', fontsize=7, color='#1e293b', fontweight='semibold')
        else:
            ax.text(width - ax_offset, bar.get_y() + bar.get_height()/2, f"{width:.2f}", 
                    va='center', ha='right', fontsize=7, color='#1e293b', fontweight='semibold')
            
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------
def run_inference(image: Image.Image):
    """
    Run full model inference on an uploaded image.
    
    Returns:
        prediction_text: str - formatted prediction result
        concept_values: list[float] - predicted concept probabilities
        heatmap_gallery: list[tuple] - (PIL image, caption) pairs for gallery
        original_image: np.ndarray - unnormalized image for display
    """
    if MODEL is None or image is None:
        return "No model loaded.", [], [], None

    # Preprocess
    img_tensor = TRANSFORM(image.convert('RGB')).unsqueeze(0).to(DEVICE)

    # Print input details
    print(f"[Debug Inference] Input image: size={image.size}, mode={image.mode}")
    print(f"[Debug Inference] img_tensor: shape={img_tensor.shape}, min={img_tensor.min().item():.4f}, max={img_tensor.max().item():.4f}, mean={img_tensor.mean().item():.4f}, has_nan={torch.isnan(img_tensor).any().item()}")

    # Forward pass (extract intermediate representations)
    MODEL.eval()
    with torch.no_grad():
        class_logits, concept_logits, attn_weights = MODEL(img_tensor)
        concept_probs = MODEL.concept_activation(concept_logits)  # [1, num_concepts]

    # Print output details
    print(f"[Debug Inference] class_logits: shape={class_logits.shape}, min={class_logits.min().item():.4f}, max={class_logits.max().item():.4f}, mean={class_logits.mean().item():.4f}, has_nan={torch.isnan(class_logits).any().item()}")
    print(f"[Debug Inference] concept_logits: shape={concept_logits.shape}, min={concept_logits.min().item():.4f}, max={concept_logits.max().item():.4f}, mean={concept_logits.mean().item():.4f}, has_nan={torch.isnan(concept_logits).any().item()}")

    # Prediction result
    if NUM_CLASSES == 1:
        prob = torch.sigmoid(class_logits).item()
        pred_label = "Malignant" if prob >= 0.5 else "Benign"
        prediction_text = f"**{pred_label}** (Malignancy probability: {prob:.4f})"
        target_class_idx = 0
    else:
        probs = torch.softmax(class_logits, dim=-1).squeeze(0)
        top_k = min(3, NUM_CLASSES)
        top_probs, top_idxs = probs.topk(top_k)
        lines = []
        for i in range(top_k):
            idx = top_idxs[i].item()
            p = top_probs[i].item()
            name = TARGET_CLASSES[idx] if idx < len(TARGET_CLASSES) else f"Class {idx}"
            marker = ">" if i == 0 else " "
            lines.append(f"{marker} **{name}**: {p:.4f}")
            print(f"[Debug Inference] Top-{i+1} Class: {idx} ({name}) prob={p:.4f}")
        prediction_text = "\n".join(lines)
        target_class_idx = top_idxs[0].item()

    # Concept values
    concept_vals = concept_probs.squeeze(0).cpu().tolist()

    # Generate heatmaps
    img_np = _unnormalize_tensor(img_tensor.squeeze(0))
    _, _, H_img, W_img = img_tensor.shape

    attn_upsampled = F.interpolate(
        attn_weights, size=(H_img, W_img), mode='bilinear', align_corners=False
    )

    use_group_broadcasting = getattr(MODEL, "use_group_broadcasting", False)
    heatmap_gallery = []
    heatmap_arrays = []
    heatmap_names = []
    for group_idx, group in enumerate(CONCEPT_GROUPS):
        if group["type"] == "numerical":
            c_idx = group["flat_indices"][0]
            val = concept_vals[c_idx]
            # scale value to original physical range for visualization title
            orig_val = group["min"] + (group["max"] - group["min"]) * val
            if group["min"].is_integer() and group["max"].is_integer() and (group["max"] - group["min"]) > 2.0:
                orig_val = int(round(orig_val))
            else:
                orig_val = round(orig_val, 2)
            
            hm_idx = group_idx if use_group_broadcasting else c_idx
            hm = attn_upsampled[0, hm_idx].cpu().numpy()
            name = f"{group['name']}: {orig_val}"
            pil_img = _generate_single_heatmap(img_np, hm, name)
            heatmap_gallery.append((pil_img, name))
            heatmap_arrays.append(hm)
            heatmap_names.append(name)
        else:
            # Categorical concept
            probs = [concept_vals[idx] for idx in group["flat_indices"]]
            max_idx = np.argmax(probs)
            max_c_idx = group["flat_indices"][max_idx]
            selected_class = group["classes"][max_idx]
            max_prob = probs[max_idx]
            
            hm_idx = group_idx if use_group_broadcasting else max_c_idx
            hm = attn_upsampled[0, hm_idx].cpu().numpy()
            name = f"{group['name']}: {selected_class} ({max_prob:.2f})"
            pil_img = _generate_single_heatmap(img_np, hm, name)
            heatmap_gallery.append((pil_img, name))
            heatmap_arrays.append(hm)
            heatmap_names.append(name)

    # 1. Generate the grid image containing all heatmaps
    num_heatmaps = len(heatmap_arrays)
    if num_heatmaps > 0:
        cols = 4
        rows = math.ceil(num_heatmaps / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
            
        for i in range(len(axes)):
            if i < num_heatmaps:
                hm = heatmap_arrays[i]
                name = heatmap_names[i]
                h_min, h_max = hm.min(), hm.max()
                if h_max > h_min:
                    hm_norm = (hm - h_min) / (h_max - h_min + 1e-8)
                else:
                    hm_norm = np.zeros_like(hm)
                    
                axes[i].imshow(img_np)
                axes[i].imshow(hm_norm, cmap='jet', alpha=0.5)
                axes[i].set_title(name, fontsize=10, fontweight='bold', pad=6)
                axes[i].axis('off')
            else:
                axes[i].axis('off')
                
        plt.tight_layout()
        os.makedirs("scratch", exist_ok=True)
        grid_path = "scratch/heatmap_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # 2. Package everything into a ZIP file
        import zipfile
        zip_path = "scratch/all_heatmaps.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(grid_path, arcname="heatmap_grid.png")
            for idx, (pil_img, name) in enumerate(heatmap_gallery):
                # Sanitize name for filename
                clean_name = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name])
                clean_name = clean_name.replace(" ", "_")
                temp_path = f"scratch/temp_heatmap_{idx}_{clean_name}.png"
                pil_img.save(temp_path)
                zipf.write(temp_path, arcname=f"individual_heatmaps/{clean_name}.png")
                os.remove(temp_path)
    else:
        grid_path = None
        zip_path = None

    concept_logits_list = concept_logits.squeeze(0).cpu().tolist()

    contrib_img = plot_concept_contributions(MODEL, concept_logits, CONCEPT_NAMES, target_class_idx)

    return prediction_text, concept_vals, heatmap_gallery, img_np, concept_logits_list, contrib_img, grid_path, zip_path


def repredict_with_adjusted_concepts(original_state, *args):
    """
    Re-run only the classifier head with user-adjusted concept values.
    This is the human-in-the-loop intervention point.
    If the user has not modified a concept group, it preserves the original predicted logits.
    """
    if MODEL is None:
        return "No model loaded.", None

    is_prob = getattr(MODEL, "use_probabilistic_cbm", False)

    # Decode original_state (dict, list, or empty)
    if not original_state:
        original_logits = [0.0] * (NUM_CONCEPTS + (getattr(MODEL, "num_latent_concepts", 0) or 0))
        original_logvars = [-2.0] * len(original_logits)
    elif isinstance(original_state, dict):
        original_logits = original_state.get("logits", [])
        original_logvars = original_state.get("logvars", [])
    else:
        # Fallback if original_state is just a list of logits
        original_logits = original_state
        original_logvars = [-2.0] * len(original_logits)

    if len(original_logvars) < len(original_logits):
        original_logvars = original_logvars + [-2.0] * (len(original_logits) - len(original_logvars))

    # Reconstruct concept logits/logvars by preserving original values unless modified
    logits_mutated = torch.tensor(
        [original_logits], dtype=torch.float32, device=DEVICE
    )
    logvars_mutated = torch.tensor(
        [original_logvars], dtype=torch.float32, device=DEVICE
    )

    num_groups = len(CONCEPT_GROUPS)
    component_values = args[:num_groups]
    uncertainty_values = args[num_groups:] if is_prob else [0.0] * num_groups

    for i, group in enumerate(CONCEPT_GROUPS):
        val = component_values[i]
        unc_val = uncertainty_values[i] if is_prob else None
        flat_indices = group["flat_indices"]

        if group["type"] == "numerical":
            c_idx = flat_indices[0]
            # Original predicted probability and value
            orig_logit = original_logits[c_idx]
            orig_prob = 1.0 / (1.0 + math.exp(-orig_logit))
            orig_val = group["min"] + (group["max"] - group["min"]) * orig_prob
            
            # Check if slider value matches predicted value
            if abs(float(val) - orig_val) > 1e-4:
                # User intervened! Override with slider value
                norm_val = (float(val) - group["min"]) / (group["max"] - group["min"] + 1e-8)
                if is_prob:
                    norm_val = max(0.001, min(0.999, norm_val))
                else:
                    norm_val = max(0.05, min(0.95, norm_val))
                logits_mutated[0, c_idx] = math.log(norm_val / (1.0 - norm_val))
                
            if is_prob and unc_val is not None:
                orig_std = math.exp(0.5 * original_logvars[c_idx])
                if abs(float(unc_val) - orig_std) > 1e-4:
                    safe_unc_val = max(1e-6, float(unc_val))
                    logvars_mutated[0, c_idx] = 2.0 * math.log(safe_unc_val)
        else:
            # Categorical concept (Dropdown / Radio)
            group_logits = [original_logits[idx] for idx in flat_indices]
            
            is_exclusive = is_group_exclusive(group["name"])
            if is_exclusive and len(flat_indices) > 1:
                # Softmax
                exp_logits = [math.exp(l) for l in group_logits]
                sum_exp = sum(exp_logits)
                group_probs = [el / (sum_exp + 1e-8) for el in exp_logits]
            else:
                # Sigmoid fallback
                group_probs = [1.0 / (1.0 + math.exp(-l)) for l in group_logits]
                
            max_idx = np.argmax(group_probs)
            orig_selected_cls = group["classes"][max_idx]
                
            # If the user changed the choice, we intervene and override!
            if val != orig_selected_cls:
                selected_classes = val if isinstance(val, list) else [val]
                selected_classes = {str(c) for c in selected_classes}
                
                for cls_idx, cls_str in zip(flat_indices, group["classes"]):
                    if cls_str in selected_classes:
                        p = 0.999 if is_prob else 0.95
                    else:
                        p = 0.001 if is_prob else 0.05
                    logits_mutated[0, cls_idx] = math.log(p / (1.0 - p))

            if is_prob and unc_val is not None:
                group_stds = [math.exp(0.5 * original_logvars[idx]) for idx in flat_indices]
                orig_group_std = float(np.mean(group_stds))
                if abs(float(unc_val) - orig_group_std) > 1e-4:
                    safe_unc_val = max(1e-6, float(unc_val))
                    new_logvar = 2.0 * math.log(safe_unc_val)
                    for idx in flat_indices:
                        logvars_mutated[0, idx] = new_logvar

    # Evaluate using the mutated concept logits
    MODEL.eval()
    with torch.no_grad():
        if is_prob:
            # Perform Monte Carlo sampling to propagate uncertainty through the classifier head
            num_samples = 50
            std = torch.exp(0.5 * logvars_mutated)  # [1, num_concepts]
            
            accum_probs = None
            for _ in range(num_samples):
                eps = torch.randn_like(std)
                sampled_logits = logits_mutated + std * eps
                if getattr(MODEL, "use_multimodal", False):
                    inputs_sampled = torch.cat([sampled_logits, torch.zeros(sampled_logits.size(0), 3, device=sampled_logits.device)], dim=-1)
                else:
                    inputs_sampled = sampled_logits
                class_logits_sample = MODEL.classifier_head(inputs_sampled)
                
                if NUM_CLASSES == 1:
                    probs_sample = torch.sigmoid(class_logits_sample)
                else:
                    probs_sample = torch.softmax(class_logits_sample, dim=-1)
                    
                if accum_probs is None:
                    accum_probs = probs_sample
                else:
                    accum_probs += probs_sample
                    
            avg_probs = accum_probs / num_samples
            
            if NUM_CLASSES == 1:
                prob = avg_probs.item()
                pred_label = "Malignant" if prob >= 0.5 else "Benign"
                pred_text = f"**{pred_label}** (Malignancy probability: {prob:.4f})"
                target_class_idx = 0
            else:
                probs = avg_probs.squeeze(0)
                top_k = min(3, NUM_CLASSES)
                top_probs, top_idxs = probs.topk(top_k)
                lines = []
                for i in range(top_k):
                    idx = top_idxs[i].item()
                    p = top_probs[i].item()
                    name = TARGET_CLASSES[idx] if idx < len(TARGET_CLASSES) else f"Class {idx}"
                    marker = ">" if i == 0 else " "
                    lines.append(f"{marker} **{name}**: {p:.4f}")
                pred_text = "\n".join(lines)
                target_class_idx = top_idxs[0].item()
        else:
            if getattr(MODEL, "use_multimodal", False):
                inputs_mutated = torch.cat([logits_mutated, torch.zeros(logits_mutated.size(0), 3, device=logits_mutated.device)], dim=-1)
            else:
                inputs_mutated = logits_mutated
            class_logits = MODEL.classifier_head(inputs_mutated)
            if NUM_CLASSES == 1:
                prob = torch.sigmoid(class_logits).item()
                pred_label = "Malignant" if prob >= 0.5 else "Benign"
                pred_text = f"**{pred_label}** (Malignancy probability: {prob:.4f})"
                target_class_idx = 0
            else:
                probs = torch.softmax(class_logits, dim=-1).squeeze(0)
                top_k = min(3, NUM_CLASSES)
                top_probs, top_idxs = probs.topk(top_k)
                lines = []
                for i in range(top_k):
                    idx = top_idxs[i].item()
                    p = top_probs[i].item()
                    name = TARGET_CLASSES[idx] if idx < len(TARGET_CLASSES) else f"Class {idx}"
                    marker = ">" if i == 0 else " "
                    lines.append(f"{marker} **{name}**: {p:.4f}")
                pred_text = "\n".join(lines)
                target_class_idx = top_idxs[0].item()

    contrib_img = plot_concept_contributions(MODEL, logits_mutated, CONCEPT_NAMES, target_class_idx)
    return pred_text, contrib_img


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
)

APP_CSS = """
.main-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}
/* Extremely tight & compact concept intervention container */
.concept-slider-group {
    max-height: 380px !important;
    overflow-y: auto !important;
    overflow-x: auto !important;
    padding: 4px !important;
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
}
/* Grid layout to keep the 5 columns side-by-side without squishing them */
.columns-container {
    display: grid !important;
    grid-template-columns: repeat(5, minmax(210px, 1fr)) !important;
    gap: 6px !important;
    width: 100% !important;
    overflow-x: auto !important;
}
.columns-container > div {
    display: flex !important;
    flex-direction: column !important;
    min-width: 0 !important;
    margin: 0 !important;
}
/* Compact accordion summary with no-wrap and alignment */
.concept-slider-group details summary,
.concept-slider-group .accordion-trigger,
.concept-slider-group summary {
    font-size: 0.72rem !important;
    padding: 3px 6px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    white-space: nowrap !important;
}
.concept-slider-group details summary span,
.concept-slider-group .accordion-trigger span {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 160px !important;
}
/* Reduce vertical spacing between sliders */
.concept-slider-group .compact-comp {
    padding: 1px 4px !important;
    margin: 0px !important;
}
/* Shrink slider labels and value texts */
.concept-slider-group label {
    font-size: 0.72rem !important;
    padding: 0px 2px !important;
}
.concept-slider-group label span {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
}
/* Shrink slider track and handle */
.concept-slider-group input[type="range"] {
    height: 3px !important;
    margin-top: 1px !important;
    margin-bottom: 1px !important;
}
.concept-slider-group input[type="number"] {
    font-size: 0.72rem !important;
    height: 18px !important;
    padding: 1px 2px !important;
    width: 45px !important;
}
/* Shrink radios and dropdowns */
.concept-slider-group .gr-radio-group, .concept-slider-group .gr-dropdown {
    font-size: 0.72rem !important;
    padding: 1px 2px !important;
}
.concept-slider-group .gr-radio-group label {
    font-size: 0.7rem !important;
    padding: 1px 4px !important;
}
/* Tight block margins */
.concept-slider-group div.block {
    padding: 1px 2px !important;
    margin: 1px !important;
    border-radius: 4px !important;
    background-color: #ffffff !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
}
"""


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks interface."""

    with gr.Blocks(title="Attention-CBM Explorer") as app:

        # ---- Header ----
        gr.HTML("<div class='main-title'>Attention-CBM Explorer</div>")
        gr.HTML("<div class='subtitle'>Interactive Concept Bottleneck Model — Inference & Human-in-the-Loop</div>")

        # ---- State for concept values ----
        concept_state = gr.State([])

        with gr.Row():
            # ==== Column 1: Input ====
            with gr.Column(scale=1):
                gr.Markdown("### Input Image")
                
                # Determine image upload label based on dataset
                if "cub" in DATASET_NAME.lower():
                    image_label = "Upload a bird image"
                elif "derm" in DATASET_NAME.lower():
                    image_label = "Upload a dermoscopy image"
                elif "milk" in DATASET_NAME.lower():
                    image_label = "Upload a milk bottle image"
                else:
                    image_label = "Upload an image"
                    
                input_image = gr.Image(
                    type="pil",
                    label=image_label,
                    height=280
                )
                run_btn = gr.Button(
                    "Run Inference",
                    variant="primary",
                    size="lg"
                )

            # ==== Column 2: Model Prediction & Contributions ====
            with gr.Column(scale=1):
                
                gr.Markdown("### Model Prediction")
                prediction_output = gr.Markdown(
                    value="*Upload an image and click Run Inference*"
                )
                
                gr.Markdown("### Concept Decision Contributions")
                contrib_output = gr.Image(
                    label="Concept Contribution Plot",
                    type="pil",
                    interactive=False,
                    height=280
                )

            # ==== Column 3: Heatmaps ====
            with gr.Column(scale=2):
                gr.Markdown("### Concept Attention Heatmaps")
                heatmap_gallery = gr.Gallery(
                    label="Per-concept attention maps",
                    columns=4,
                    rows=2,
                    height=520,
                    object_fit="contain"
                )
                gr.Markdown("### Download Heatmaps")
                with gr.Row():
                    grid_download = gr.File(
                        label="Download Heatmap Grid (PNG)",
                        interactive=False
                    )
                    zip_download = gr.File(
                        label="Download All Heatmaps (ZIP)",
                        interactive=False
                    )

        gr.Markdown("---")

        # ---- Human-in-the-Loop Section ----
        gr.Markdown("### Human-in-the-Loop: Concept Intervention")
        gr.Markdown(
            "Adjust the concept values below and click **Re-predict** to see how "
            "changes in individual concepts affect the final classification. "
            "This lets you explore the model's decision-making process interactively."
        )

        with gr.Row():
            with gr.Column(scale=3, elem_classes="concept-slider-group"):
                group_name_to_comp = {}
                group_name_to_unc = {}
                group_name_to_accordion = {}
                num_cols = 5
                cols_groups = [CONCEPT_GROUPS[i::num_cols] for i in range(num_cols)]
                
                with gr.Row(elem_classes="columns-container"):
                    for col_idx in range(num_cols):
                        with gr.Column():
                            for group in cols_groups[col_idx]:
                                with gr.Accordion(label=group["name"], open=False) as accordion:
                                    if group["type"] == "numerical":
                                        is_int = (group["min"].is_integer() and group["max"].is_integer() and (group["max"] - group["min"]) > 2.0)
                                        step = 1.0 if is_int else 0.01
                                        comp = gr.Slider(
                                            minimum=group["min"],
                                            maximum=group["max"],
                                            step=step,
                                            value=(group["min"] + group["max"]) / 2,
                                            label=group["name"],
                                            show_label=False,
                                            interactive=True,
                                            elem_classes="compact-comp"
                                        )
                                    else:
                                        # Categorical Concept Component
                                        choices = group["classes"] + ["Not Visible / Occluded"]
                                        default_val = group["classes"][0]
                                        if len(group["classes"]) <= 3:
                                            comp = gr.Radio(
                                                choices=choices,
                                                value=default_val,
                                                label=group["name"],
                                                show_label=False,
                                                interactive=True,
                                                elem_classes="compact-comp"
                                            )
                                        else:
                                            comp = gr.Dropdown(
                                                choices=choices,
                                                value=default_val,
                                                label=group["name"],
                                                show_label=False,
                                                multiselect=False,
                                                interactive=True,
                                                elem_classes="compact-comp"
                                            )
                                    
                                    # Create uncertainty slider (hidden by default unless model is probabilistic)
                                    is_prob = MODEL is not None and getattr(MODEL, "use_probabilistic_cbm", False)
                                    unc_comp = gr.Slider(
                                        minimum=0.0,
                                        maximum=2.0,
                                        value=0.0,
                                        label="Uncertainty (σ)",
                                        interactive=True,
                                        visible=is_prob,
                                        elem_classes="compact-comp"
                                    )
                                    group_name_to_comp[group["name"]] = comp
                                    group_name_to_unc[group["name"]] = unc_comp
                                    group_name_to_accordion[group["name"]] = accordion
                
                # Reconstruct concept_components and concept_accordions in the exact order of CONCEPT_GROUPS
                concept_components = [group_name_to_comp[g["name"]] for g in CONCEPT_GROUPS]
                uncertainty_components = [group_name_to_unc[g["name"]] for g in CONCEPT_GROUPS]
                concept_accordions = [group_name_to_accordion[g["name"]] for g in CONCEPT_GROUPS]

            with gr.Column(scale=1):
                repredict_btn = gr.Button(
                    "Re-predict with adjusted concepts",
                    variant="secondary",
                    size="lg"
                )
                adjusted_prediction = gr.Markdown(
                    value="*Adjust concepts and click Re-predict*"
                )

        # ---- Event Handlers ----
        def on_inference(image):
            if image is None:
                return (
                    "*Please upload an image first.*",
                    gr.update(),
                    [],
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    *[gr.update() for _ in concept_components],
                    *[gr.update() for _ in uncertainty_components],
                    *[gr.update() for _ in concept_accordions]
                )

            pred_text, concept_vals, gallery, _, concept_logits_list, contrib_img, grid_path, zip_path = run_inference(image)

            # Build updates to reflect predicted values and accordion open/closed state
            component_updates = []
            uncertainty_updates = []
            accordion_updates = []
            
            # Fetch uncertainty standard deviations if model is probabilistic
            if MODEL is not None and getattr(MODEL, "use_probabilistic_cbm", False) and getattr(MODEL, "last_logvar", None) is not None:
                std_vals = torch.exp(0.5 * MODEL.last_logvar).squeeze(0).cpu().tolist()
                concept_logvars_list = MODEL.last_logvar.squeeze(0).cpu().tolist()
            else:
                std_vals = [0.0] * NUM_CONCEPTS
                concept_logvars_list = [0.0] * NUM_CONCEPTS
                
            for group in CONCEPT_GROUPS:
                flat_indices = group["flat_indices"]
                group_std = float(np.mean([std_vals[idx] for idx in flat_indices]))
                uncertainty_updates.append(gr.update(value=group_std))
                
                if group["type"] == "numerical":
                    val = concept_vals[flat_indices[0]]
                    # Scale val [0, 1] to [min, max]
                    scaled_val = group["min"] + (group["max"] - group["min"]) * val
                    # Round value for clean display
                    if group["min"].is_integer() and group["max"].is_integer() and (group["max"] - group["min"]) > 2.0:
                        scaled_val = int(round(scaled_val))
                    else:
                        scaled_val = round(scaled_val, 4)
                    component_updates.append(gr.update(value=scaled_val))
                    accordion_updates.append(gr.update(open=False))
                else:
                    # Categorical concept: select the highest probability class
                    probs = [concept_vals[idx] for idx in flat_indices]
                    max_idx = np.argmax(probs)
                    
                    # Extract raw predicted logits for this group
                    group_logits = [concept_logits_list[idx] for idx in flat_indices]
                    max_logit = max(group_logits)
                    
                    # Fetch optimal validation threshold in logit space from model buffer
                    if MODEL is not None and hasattr(MODEL, "concept_thresholds") and MODEL.concept_thresholds is not None:
                        g_threshold = MODEL.concept_thresholds[flat_indices].mean().item()
                    else:
                        g_threshold = 0.0  # Fallback to logit 0.0
                    
                    # Intervene (open Accordion) if predicted max logit is below threshold + 0.30 logit margin
                    selected_cls = group["classes"][max_idx]
                    if max_logit <= (g_threshold + 0.30):
                        accordion_updates.append(gr.update(open=True))
                    else:
                        accordion_updates.append(gr.update(open=False))
                    
                    component_updates.append(gr.update(value=selected_cls))

            state_dict = {"logits": concept_logits_list, "logvars": concept_logvars_list}
            return (pred_text, state_dict, gallery, contrib_img, grid_path, zip_path, *component_updates, *uncertainty_updates, *accordion_updates)

        run_btn.click(
            fn=on_inference,
            inputs=[input_image],
            outputs=[prediction_output, concept_state, heatmap_gallery, contrib_output, grid_download, zip_download, *concept_components, *uncertainty_components, *concept_accordions]
        )

        repredict_btn.click(
            fn=repredict_with_adjusted_concepts,
            inputs=[concept_state, *concept_components, *uncertainty_components],
            outputs=[adjusted_prediction, contrib_output]
        )

    return app


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------
def parse_app_args():
    parser = argparse.ArgumentParser(description="Gradio Inference App for Attention-CBM")
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        '--concept_config_path', type=str,
        default='data/MILK10K/concept_config.json',
        help="Path to concept configuration JSON"
    )
    parser.add_argument(
        '--backbone_type', type=str, default='timm',
        choices=['timm', 'clip', 'torchxrayvision']
    )
    parser.add_argument(
        '--backbone_name', type=str, default='resnet50'
    )
    parser.add_argument(
        '--num_classes', type=int, default=1
    )
    parser.add_argument(
        '--target_classes', type=str, default="",
        help="Comma-separated target class names (e.g. 'AKIEC,BCC,BEN_OTH,...')"
    )
    parser.add_argument(
        '--latent_concepts', type=int, default=0,
        help="Number of latent concepts (automatically detected from checkpoint if not specified)"
    )
    parser.add_argument(
        '--use_lora', action='store_true', default=False,
        help="Enable LoRA adapters for ViT backbone"
    )
    # (use_dino_mask and dino_mask_threshold CLI arguments removed)
    parser.add_argument(
        '--no_grouping', action='store_true', default=False,
        help="Disable mutually exclusive concept grouping (defaults to auto-detecting from checkpoint)"
    )
    parser.add_argument(
        '--use_group_broadcasting', action='store_true', default=False,
        help="Use GroupToConceptAttention layout (group queries → independent BCE classifiers based on concept_config)"
    )
    parser.add_argument(
        '--use_gated_nam', type=str2bool, default=False,
        help="Activate Gated Sparse NAM head"
    )
    parser.add_argument(
        '--use_pairwise_nam', type=str2bool, default=False,
        help="Activate Pairwise Interaction NAM^2 head"
    )
    parser.add_argument(
        '--use_probabilistic_cbm', type=str2bool, default=False,
        help="Convert Concept Extractor to Probabilistic"
    )
    parser.add_argument(
        '--use_concept_attention', type=str2bool, default=False,
        help="Activate Patch token-based Cross-Attention"
    )
    parser.add_argument(
        '--lora_r', type=int, default=8
    )
    parser.add_argument(
        '--lora_alpha', type=float, default=16.0
    )
    parser.add_argument(
        '--use_multimodal', type=str2bool, default=False,
        help="Enable late fusion with tabular features (Age, Sex) by concatenating them INTO classifier head input"
    )
    parser.add_argument(
        '--age_sex_skip_connection', type=str2bool, default=False,
        help="Bypass NAM with Age/Sex: add a dedicated linear(tabular -> classes) directly to final logits"
    )
    parser.add_argument(
        '--port', type=int, default=7860,
        help="Port to serve the Gradio app on"
    )
    return parser.parse_args()

def main():
    global MODEL, DEVICE, CONCEPT_NAMES, CONCEPT_CONFIG, CONCEPT_GROUPS, TARGET_CLASSES, NUM_CONCEPTS, NUM_CLASSES, DATASET_NAME

    # Pre-warm matplotlib to prevent first-run slow renderer/font cache loading
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        plt.close(fig)
        print("[Config] Matplotlib pre-warmed successfully.")
    except Exception as e:
        print(f"[Config] Matplotlib pre-warming failed: {e}")

    args = parse_app_args()

    # Load checkpoint first to inspect dimensions and metadata for dynamic concept filtering
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(args.checkpoint):
        try:
            loaded_checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
            if isinstance(loaded_checkpoint, dict) and "state_dict" in loaded_checkpoint:
                state_dict = loaded_checkpoint["state_dict"]
            else:
                state_dict = loaded_checkpoint
                
            use_filtered = False
            if isinstance(loaded_checkpoint, dict):
                ckpt_args = loaded_checkpoint.get("args", {})
                ckpt_config = loaded_checkpoint.get("config", {})
                
                # Check for config key structure
                if ckpt_config and isinstance(ckpt_config, dict):
                    if "dataset" in ckpt_config and isinstance(ckpt_config["dataset"], dict):
                        ckpt_args.update(ckpt_config["dataset"])
                    if "training" in ckpt_config and isinstance(ckpt_config["training"], dict):
                        ckpt_args.update(ckpt_config["training"])
                    if "backbone" in ckpt_config and isinstance(ckpt_config["backbone"], dict):
                        ckpt_args.update(ckpt_config["backbone"])
                
                # Auto-detect dataset name
                global DATASET_NAME
                detected_ds = ckpt_args.get("dataset") or ckpt_args.get("name") or ckpt_args.get("dataset_name")
                if detected_ds:
                    DATASET_NAME = str(detected_ds).lower()
                    print(f"[Config] Auto-detected dataset from checkpoint: {DATASET_NAME}")
                else:
                    # Fallback detection
                    if "chexpert" in str(ckpt_args.get("concept_config_path", "")).lower() or "chexpert" in str(args.checkpoint).lower():
                        DATASET_NAME = "chexpert"
                    elif ckpt_args.get("num_classes") in [5, 20]:
                        DATASET_NAME = "derm7pt"
                    elif ckpt_args.get("num_classes") == 200:
                        DATASET_NAME = "cub"
                    elif "derm7pt" in str(ckpt_args.get("concept_config_path", "")).lower():
                        DATASET_NAME = "derm7pt"
                    elif "cub" in str(ckpt_args.get("concept_config_path", "")).lower():
                        DATASET_NAME = "cub"
                    else:
                        DATASET_NAME = "milk10k"
                    print(f"[Config] Fallback-detected dataset: {DATASET_NAME}")
                
                # 1. Auto-detect num_classes
                if "num_classes" in ckpt_args:
                    args.num_classes = ckpt_args["num_classes"]
                    print(f"[Config] Auto-detected num_classes from checkpoint: {args.num_classes}")
                    
                # 2. Auto-detect concept_config_path
                if "concept_config_path" in ckpt_args:
                    args.concept_config_path = ckpt_args["concept_config_path"]
                    print(f"[Config] Auto-detected concept_config_path from checkpoint: {args.concept_config_path}")
                
                # 3. Auto-detect filtering settings
                if ckpt_args.get("filter_rare_concepts", False) or ckpt_args.get("use_paper_preprocessing", False):
                    use_filtered = True
                    
            if not use_filtered and "classifier_head.weight" in state_dict:
                checkpoint_dims = state_dict["classifier_head.weight"].shape[1]
                # Let's count how many concepts the original concept config has
                if os.path.exists(args.concept_config_path):
                    with open(args.concept_config_path, 'r', encoding='utf-8') as f:
                        orig_cfg = json.load(f)
                    orig_dims = 0
                    for n, inf in orig_cfg.items():
                        if inf.get("type") == "categorical":
                            orig_dims += len(inf.get("classes", []))
                        else:
                            orig_dims += 1
                    if checkpoint_dims < orig_dims:
                        use_filtered = True
                        print(f"[Config] Dimension mismatch detected: Checkpoint has {checkpoint_dims} dimensions, but original config has {orig_dims}. Attempting to use filtered config.")
                        
            if use_filtered:
                filtered_path = args.concept_config_path.replace(".json", "_filtered.json")
                if not os.path.exists(filtered_path):
                    print(f"[Config] Filtered config not found at {filtered_path}. Generating it dynamically by instantiating dataset...")
                    try:
                        from src.data.cub import CUB2011Dataset
                        _ = CUB2011Dataset(
                            split='test',
                            config={
                                "num_concepts": 312,
                                "num_classes": 200,
                                "concepts": [],
                                "target_col": 'class_id',
                                "default_csv_path": 'data/CUB_200_2011/images.txt',
                                "default_image_dir": 'data/CUB_200_2011/images',
                                "filter_rare_concepts": False,
                                "use_paper_preprocessing": True,
                                "concept_config_path": args.concept_config_path
                            }
                        )
                    except Exception as ex:
                        print(f"[Config] Failed to generate filtered concept configuration: {ex}")
                
                if os.path.exists(filtered_path):
                    print(f"[Config] Automatically redirecting concept config to: {filtered_path}")
                    args.concept_config_path = filtered_path
                else:
                    print(f"[Config] Warning: Checkpoint indicates rare concept filtering, but filtered config was not found at: {filtered_path}")
        except Exception as e:
            print(f"[Config] Pre-loading checkpoint failed to auto-detect filtering settings: {e}")

    # 1. Load concept config
    if (args.concept_config_path == 'data/MILK10K/concept_config.json' or not os.path.exists(args.concept_config_path)) and DATASET_NAME == 'chexpert':
        args.concept_config_path = 'data/CheXpert/concept_config.json'
        print(f"[Config] Automatically redirecting concept config to: {args.concept_config_path}")

    if not os.path.exists(args.concept_config_path):
        raise FileNotFoundError(f"Concept config not found: {args.concept_config_path}")

    with open(args.concept_config_path, 'r', encoding='utf-8') as f:
        CONCEPT_CONFIG = json.load(f)

    # Build concepts_flat & CONCEPT_GROUPS (same logic as dataset classes)
    concepts_flat = []
    total_dims = 0
    CONCEPT_GROUPS = []
    
    for name, info in CONCEPT_CONFIG.items():
        ctype = info.get("type", "numerical")
        if ctype == "categorical":
            classes = info.get("classes", [])
            classes_str = [str(c) for c in classes]
            group = {
                "name": name,
                "type": "categorical",
                "classes": classes_str,
                "flat_indices": list(range(total_dims, total_dims + len(classes)))
            }
            for cls_val in classes:
                concepts_flat.append(f"{name}_{cls_val}")
            total_dims += len(classes)
        else:
            group = {
                "name": name,
                "type": "numerical",
                "min": float(info.get("min", 0.0)),
                "max": float(info.get("max", 1.0)),
                "flat_indices": [total_dims]
            }
            concepts_flat.append(name)
            total_dims += 1
        CONCEPT_GROUPS.append(group)

    CONCEPT_NAMES = concepts_flat
    NUM_CONCEPTS = total_dims
    NUM_CLASSES = args.num_classes

    # Target class names for multi-class display
    if args.target_classes:
        TARGET_CLASSES = [c.strip() for c in args.target_classes.split(',')]
    elif NUM_CLASSES > 1:
        if DATASET_NAME == "chexpert":
            TARGET_CLASSES = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
        elif NUM_CLASSES in [5, 20]:
            try:
                from src.data.derm7pt import Derm7PtDataset
                dataset = Derm7PtDataset(cache_in_memory=False)
                TARGET_CLASSES = sorted(list(dataset.target_to_idx.keys()))
            except Exception as e:
                TARGET_CLASSES = [f"Class {i}" for i in range(NUM_CLASSES)]
        elif NUM_CLASSES == 200:
            try:
                classes_path = "data/CUB_200_2011/classes.txt"
                classes = []
                with open(classes_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 2:
                                name = " ".join(parts[1:])
                                if '.' in name:
                                    name = name.split('.', 1)[1]
                                name = name.replace('_', ' ')
                                classes.append(name)
                TARGET_CLASSES = classes
            except Exception as e:
                TARGET_CLASSES = [f"Class {i}" for i in range(NUM_CLASSES)]
        else:
            # Default MILK10K classes
            from src.data.milk10k import MILK10KDataset
            TARGET_CLASSES = MILK10KDataset.GT_LABEL_COLS
    else:
        TARGET_CLASSES = []

    print(f"Loaded {NUM_CONCEPTS} concepts from {args.concept_config_path}")
    if TARGET_CLASSES:
        print(f"Target classes ({len(TARGET_CLASSES)}): {TARGET_CLASSES}")

    # 2. Inspect checkpoint first to automatically detect model parameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded_checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if isinstance(loaded_checkpoint, dict) and "state_dict" in loaded_checkpoint:
        state_dict = loaded_checkpoint["state_dict"]
    else:
        state_dict = loaded_checkpoint
    
    # Auto-detect number of latent concepts and NAM head configuration from checkpoint keys
    latent_concepts = args.latent_concepts
    use_nam_head = False
    nam_hidden_dim = 64

    # 1. Check if NAM head is used
    if "classifier_head.concept_gates" in state_dict:
        use_nam_head = True
        # GatedSparseNAMHead uses conv1 grouped conv. Out channels is num_concepts * hidden_dim
        if "classifier_head.conv1.weight" in state_dict:
            out_ch = state_dict["classifier_head.conv1.weight"].shape[0]
            num_nam_gates = state_dict["classifier_head.concept_gates"].shape[0]
            nam_hidden_dim = out_ch // num_nam_gates
            print(f"[Config] Auto-detected GatedSparseNAMHead: use_nam_head=True, nam_hidden_dim={nam_hidden_dim}")
        
        # Detect latent concepts in NAM: check if latent_linear layer weights exist
        if "classifier_head.latent_linear.weight" in state_dict:
            latent_concepts = state_dict["classifier_head.latent_linear.weight"].shape[1]
            print(f"[Config] Auto-detected latent concepts from NAM latent_linear: {latent_concepts}")
        else:
            latent_concepts = 0
            print(f"[Config] No latent concepts found in NAM head. Setting latent_concepts=0")
    # 2. Otherwise fallback to standard linear head
    elif "classifier_head.weight" in state_dict:
        checkpoint_dims = state_dict["classifier_head.weight"].shape[1]
        detected_latent = checkpoint_dims - NUM_CONCEPTS
        if detected_latent >= 0:
            latent_concepts = detected_latent
            print(f"[Config] Auto-detected standard head with latent concepts: {latent_concepts} (Total dimensions: {checkpoint_dims})")
        else:
            print(f"[Config] Warning: Checkpoint dimensions ({checkpoint_dims}) are less than supervised concepts ({NUM_CONCEPTS}). Using args.latent_concepts={args.latent_concepts}.")
    else:
        print(f"[Config] Warning: 'classifier_head.weight' or 'classifier_head.concept_gates' not found in checkpoint. Using args.latent_concepts={args.latent_concepts}.")

    use_lora = getattr(args, 'use_lora', False)
    lora_r = getattr(args, 'lora_r', 8)
    lora_alpha = getattr(args, 'lora_alpha', 16.0)
    backbone_name = args.backbone_name
    backbone_type = args.backbone_type
    use_concept_groups = True
    use_cosine_attention = getattr(args, 'use_cosine_attention', False)
    use_group_broadcasting = getattr(args, 'use_group_broadcasting', False)
    # (use_dino_mask and dino_mask_threshold variables removed)
    use_gated_nam = getattr(args, 'use_gated_nam', False)
    use_pairwise_nam = getattr(args, 'use_pairwise_nam', False)
    use_probabilistic_cbm = getattr(args, 'use_probabilistic_cbm', False)
    use_concept_attention = getattr(args, 'use_concept_attention', False)
    use_multimodal = getattr(args, 'use_multimodal', False)
    age_sex_skip_connection = getattr(args, 'age_sex_skip_connection', False)
    
    if isinstance(loaded_checkpoint, dict) and 'args' in loaded_checkpoint:
        checkpoint_args = loaded_checkpoint['args']
        if 'use_lora' in checkpoint_args:
            use_lora = checkpoint_args['use_lora']
            lora_r = checkpoint_args.get('lora_r', 8)
            lora_alpha = checkpoint_args.get('lora_alpha', 16.0)
        if 'use_cosine_attention' in checkpoint_args:
            use_cosine_attention = checkpoint_args['use_cosine_attention']
        if 'use_group_broadcasting' in checkpoint_args:
            use_group_broadcasting = checkpoint_args['use_group_broadcasting']
        if 'backbone_name' in checkpoint_args:
            backbone_name = checkpoint_args['backbone_name']
            backbone_type = checkpoint_args.get('backbone_type', 'timm')
        if 'use_concept_groups' in checkpoint_args:
            use_concept_groups = checkpoint_args['use_concept_groups']
        pass
        if 'use_gated_nam' in checkpoint_args:
            use_gated_nam = checkpoint_args['use_gated_nam']
        elif 'use_nam_head' in checkpoint_args:
            use_gated_nam = checkpoint_args['use_nam_head']
        if 'use_pairwise_nam' in checkpoint_args:
            use_pairwise_nam = checkpoint_args['use_pairwise_nam']
        if 'use_probabilistic_cbm' in checkpoint_args:
            use_probabilistic_cbm = checkpoint_args['use_probabilistic_cbm']
        if 'use_concept_attention' in checkpoint_args:
            use_concept_attention = checkpoint_args['use_concept_attention']
        if 'use_multimodal' in checkpoint_args:
            use_multimodal = checkpoint_args['use_multimodal']
        if 'age_sex_skip_connection' in checkpoint_args:
            age_sex_skip_connection = checkpoint_args['age_sex_skip_connection']
        print(f"[Config] Auto-detected Config from checkpoint args: backbone={backbone_name} ({backbone_type}), use_lora={use_lora}, r={lora_r}, alpha={lora_alpha}, use_concept_groups={use_concept_groups}, use_group_broadcasting={use_group_broadcasting}, use_probabilistic_cbm={use_probabilistic_cbm}, use_concept_attention={use_concept_attention}, use_multimodal={use_multimodal}, age_sex_skip_connection={age_sex_skip_connection}")
    elif isinstance(loaded_checkpoint, dict) and 'config' in loaded_checkpoint:
        checkpoint_cfg = loaded_checkpoint['config']
        bb_cfg = checkpoint_cfg.get('backbone', {})
        ds_cfg = checkpoint_cfg.get('dataset', {})
        tr_cfg = checkpoint_cfg.get('training', {})
        if 'use_lora' in bb_cfg:
            use_lora = bb_cfg['use_lora']
            lora_r = bb_cfg.get('lora_r', 8)
            lora_alpha = bb_cfg.get('lora_alpha', 16.0)
        if 'use_cosine_attention' in bb_cfg:
            use_cosine_attention = bb_cfg['use_cosine_attention']
        if 'use_group_broadcasting' in bb_cfg:
            use_group_broadcasting = bb_cfg['use_group_broadcasting']
        if 'backbone_name' in bb_cfg:
            backbone_name = bb_cfg['backbone_name']
            backbone_type = bb_cfg.get('backbone_type', 'timm')
        if 'use_concept_groups' in ds_cfg:
            use_concept_groups = ds_cfg['use_concept_groups']
        pass
        if 'use_gated_nam' in tr_cfg:
            use_gated_nam = tr_cfg['use_gated_nam']
        elif 'use_nam_head' in tr_cfg:
            use_gated_nam = tr_cfg['use_nam_head']
        if 'use_pairwise_nam' in tr_cfg:
            use_pairwise_nam = tr_cfg['use_pairwise_nam']
        if 'use_probabilistic_cbm' in tr_cfg:
            use_probabilistic_cbm = tr_cfg['use_probabilistic_cbm']
        if 'use_concept_attention' in bb_cfg:
            use_concept_attention = bb_cfg['use_concept_attention']
        if 'use_multimodal' in ds_cfg:
            use_multimodal = ds_cfg['use_multimodal']
        elif 'use_multimodal' in tr_cfg:
            use_multimodal = tr_cfg['use_multimodal']
        if 'age_sex_skip_connection' in ds_cfg:
            age_sex_skip_connection = ds_cfg['age_sex_skip_connection']
        elif 'age_sex_skip_connection' in tr_cfg:
            age_sex_skip_connection = tr_cfg['age_sex_skip_connection']
        print(f"[Config] Auto-detected Config from checkpoint config: backbone={backbone_name} ({backbone_type}), use_lora={use_lora}, r={lora_r}, alpha={lora_alpha}, use_concept_groups={use_concept_groups}, use_group_broadcasting={use_group_broadcasting}, use_probabilistic_cbm={use_probabilistic_cbm}, use_concept_attention={use_concept_attention}, use_multimodal={use_multimodal}, age_sex_skip_connection={age_sex_skip_connection}")
    else:
        # Fallback to key scanning: if "backbone.vit.blocks.0.attn.qkv.lora_A" exists, LoRA must be True!
        has_lora_keys = any("lora_" in key for key in state_dict.keys())
        if has_lora_keys:
            use_lora = True
        # Detect cosine attention from state_dict keys (cosine path has q_proj/k_proj, standard has cross_attention)
        has_cosine_keys = any(".q_proj.weight" in k and "supervised_attention" in k for k in state_dict.keys())
        has_mha_keys    = any(".cross_attention." in k for k in state_dict.keys())
        if has_cosine_keys and not has_mha_keys:
            use_cosine_attention = True
        # Detect group broadcasting
        has_group_queries = any("supervised_attention.group_queries" in k for k in state_dict.keys())
        if has_group_queries:
            use_group_broadcasting = True
        # Dynamic inference of ViT shape based on weight keys
        is_vit_keys = any("blocks." in key for key in state_dict.keys())
        if is_vit_keys:
            backbone_name = "vit_base_patch16_224"
        # Fallback multimodal / skip-connection detection from state_dict keys
        if "classifier_head.concept_gates" in state_dict:
            num_supervised_gates = state_dict["classifier_head.concept_gates"].shape[0]
            if num_supervised_gates > NUM_CONCEPTS and DATASET_NAME == 'chexpert':
                use_multimodal = True
        elif "classifier_head.weight" in state_dict:
            checkpoint_dims = state_dict["classifier_head.weight"].shape[1]
            if checkpoint_dims > NUM_CONCEPTS and DATASET_NAME == 'chexpert':
                use_multimodal = True
        if 'tabular_skip_head.weight' in state_dict:
            age_sex_skip_connection = True
        print(f"[Config] Auto-detected from state_dict keys: backbone={backbone_name}, use_lora={use_lora}, use_cosine_attention={use_cosine_attention}, use_concept_groups=True, use_group_broadcasting={use_group_broadcasting}, use_multimodal={use_multimodal}, age_sex_skip_connection={age_sex_skip_connection}")
    
    # Command line argument override
    if getattr(args, 'no_grouping', False):
        use_concept_groups = False
        print("[Config] Command line override: Disabling concept grouping.")

    # Build concept_groups_info for dynamic softmax activation integration
    global USE_CONCEPT_GROUPS
    concept_groups_info = None
    if use_concept_groups:
        target_groups = None
        if isinstance(use_concept_groups, str):
            if use_concept_groups.lower() == 'true':
                target_groups = None
                USE_CONCEPT_GROUPS = True
            elif use_concept_groups.lower() == 'false':
                use_concept_groups = False
                USE_CONCEPT_GROUPS = False
            else:
                target_groups = {name.strip() for name in use_concept_groups.split(',')}
                USE_CONCEPT_GROUPS = target_groups
        elif isinstance(use_concept_groups, list):
            target_groups = {str(name).strip() for name in use_concept_groups}
            USE_CONCEPT_GROUPS = target_groups
        else:
            USE_CONCEPT_GROUPS = use_concept_groups
            
        if use_concept_groups:
            concept_groups_info = []
            grouped_count = 0
            for group in CONCEPT_GROUPS:
                start = group["flat_indices"][0]
                num = len(group["flat_indices"])
                name = group["name"]
                if target_groups is not None and name not in target_groups:
                    # Treat each class as an individual sigmoid category (group of size 1)
                    for i in range(num):
                        concept_groups_info.append((start + i, 1))
                else:
                    concept_groups_info.append((start, num))
                    if num > 1:
                        grouped_count += 1
            print(f"[Config] Configured Group-level Softmax over {grouped_count} mutually exclusive groups out of {len(CONCEPT_GROUPS)} total categories.")
    
    if not use_concept_groups or concept_groups_info is None:
        concept_groups_info = None
        USE_CONCEPT_GROUPS = False
        print("[Config] Group-level Softmax Activation is DISABLED (Sigmoid activation fallback active).")

    # 2b. Build group_mapping for GroupToConceptAttention (if requested)
    group_mapping = None
    num_groups    = len(CONCEPT_GROUPS)
    if use_group_broadcasting:
        group_mapping = []
        for group_idx, group in enumerate(CONCEPT_GROUPS):
            num_in_group = len(group["flat_indices"])
            group_mapping.extend([group_idx] * num_in_group)
        assert len(group_mapping) == NUM_CONCEPTS, (
            f"group_mapping length {len(group_mapping)} != NUM_CONCEPTS {NUM_CONCEPTS}"
        )
        # When using group broadcasting, disable Group Softmax (which conflicts with independent BCE)
        concept_groups_info = None
        USE_CONCEPT_GROUPS = False
        print(f"[Config] Group Broadcasting: {num_groups} anatomical groups → {NUM_CONCEPTS} independent BCE classifiers (Group Softmax disabled).")

    # 3. Initialize model with correct parameters
    MODEL = UniversalFlexibleCBM(
        backbone_type=backbone_type,
        backbone_name=backbone_name,
        num_supervised_concepts=NUM_CONCEPTS,
        num_classes=NUM_CLASSES,
        num_latent_concepts=latent_concepts,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        concept_groups_info=concept_groups_info,
        use_cosine_attention=use_cosine_attention,
        use_group_broadcasting=use_group_broadcasting,
        num_groups=num_groups,
        group_mapping=group_mapping,
        # use_dino_mask and dino_mask_threshold parameters removed
        use_nam_head=use_nam_head or use_gated_nam,
        nam_hidden_dim=nam_hidden_dim,
        use_probabilistic_cbm=use_probabilistic_cbm,
        use_concept_attention=use_concept_attention,
        use_pairwise_nam=use_pairwise_nam,
        use_multimodal=use_multimodal,
        age_sex_skip_connection=age_sex_skip_connection,
    )

    # ── State-dict migration: old MHA → new Cosine Attention keys ─────────────
    # Checkpoints trained before the Cosine Attention refactor contain
    # nn.MultiheadAttention sub-keys (cross_attention.in_proj_weight, …).
    # We remap them to the new explicit q_proj/k_proj/v_proj/out_proj layout
    # so that old checkpoints can still be loaded into the updated architecture.
    def _migrate_state_dict(sd: dict) -> dict:
        """Remap legacy MHA keys to cosine-attention keys (in-place copy)."""
        migrated = {}
        for k, v in sd.items():
            # Pattern: <prefix>.cross_attention.<suffix> → handled below
            if ".cross_attention.in_proj_weight" in k:
                prefix = k.replace(".cross_attention.in_proj_weight", "")
                D = v.shape[0] // 3
                # nn.MHA packs Q/K/V into a single in_proj_weight [3D, D]
                migrated[f"{prefix}.q_proj.weight"] = v[:D].clone()
                migrated[f"{prefix}.k_proj.weight"] = v[D:2*D].clone()
                migrated[f"{prefix}.v_proj.weight"] = v[2*D:].clone()
                print(f"  [Migration] Migrated in_proj_weight → q/k/v_proj.weight  ({prefix})")
            elif ".cross_attention.in_proj_bias" in k:
                # Bias is not used in the new projections (bias=False); skip.
                print(f"  [Migration] Skipped in_proj_bias (new proj layers have no bias): {k}")
            elif ".cross_attention.out_proj.weight" in k:
                new_k = k.replace(".cross_attention.out_proj.weight", ".out_proj.weight")
                migrated[new_k] = v
                print(f"  [Migration] Migrated out_proj.weight: {k} → {new_k}")
            elif ".cross_attention.out_proj.bias" in k:
                # out_proj bias not present in new arch; skip.
                print(f"  [Migration] Skipped out_proj.bias (new out_proj has no bias): {k}")
            else:
                migrated[k] = v
        return migrated

    old_keys = {k for k in state_dict if ".cross_attention." in k}
    if old_keys and use_cosine_attention:
        print(f"[Migration] Checkpoint contains {len(old_keys)} legacy MHA key(s) and use_cosine_attention is True. Running migration...")
        state_dict = _migrate_state_dict(state_dict)
        print("[Migration] State-dict migration complete.")

    # Load with strict=False so that new parameters (temperature, …) that are
    # absent from old checkpoints are left at their default initialization.
    missing, unexpected = MODEL.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Config] New parameters not found in checkpoint (initialized from scratch): {missing}")
    if unexpected:
        print(f"[Config] Unexpected keys ignored during loading: {unexpected}")
    MODEL.to(DEVICE)
    MODEL.eval()

    print(f"Model loaded from {checkpoint_path} on {DEVICE}")

    # 4. Build and launch app
    # TODO(security): In production, add authentication and restrict to trusted users.
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        theme=APP_THEME,
        css=APP_CSS
    )


if __name__ == "__main__":
    main()
