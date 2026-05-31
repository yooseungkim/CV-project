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
    return Image.open(buf)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def generate_segmentation_overlay(img_np: np.ndarray, attn: Optional[torch.Tensor], use_mask: bool, threshold: float) -> Image.Image:
    """Generate the binarized silhouette mask overlay or soft CLS-attention map overlay."""
    if attn is None:
        return Image.fromarray((img_np * 255).astype(np.uint8))
        
    try:
        # cls_attn: [1, num_heads, N_patches]
        cls_attn = attn[:, :, 0, 1:]
        mean_attn = cls_attn.mean(dim=1)  # [1, N_patches]
        
        # Min-max normalization per image
        min_val = mean_attn.min(dim=1, keepdim=True)[0]
        max_val = mean_attn.max(dim=1, keepdim=True)[0]
        norm_attn = (mean_attn - min_val) / (max_val - min_val + 1e-8)
        
        N_patches = norm_attn.size(1)
        H_grid = int(math.sqrt(N_patches))
        norm_attn_2d = norm_attn.view(1, 1, H_grid, H_grid)
        
        # Upsample to 224x224
        norm_attn_upsampled = F.interpolate(norm_attn_2d, size=(224, 224), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
        
        if use_mask:
            # Binarized mask overlay: darken the background (multiply original image by mask * 0.75 + 0.25)
            mask = (norm_attn_upsampled > threshold).astype(np.float32)
            overlay = img_np * np.expand_dims(mask * 0.75 + 0.25, axis=-1)
            overlay_img = (overlay * 255).astype(np.uint8)
            return Image.fromarray(overlay_img)
        else:
            # Soft attention overlay
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(img_np)
            ax.imshow(norm_attn_upsampled, cmap='jet', alpha=0.45)
            ax.axis('off')
            plt.tight_layout(pad=0)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            buf.seek(0)
            plt.close(fig)
            return Image.open(buf)
    except Exception as e:
        print(f"Error generating segmentation overlay: {e}")
        return Image.fromarray((img_np * 255).astype(np.uint8))


def _unnormalize_tensor(img_tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization and convert to numpy HWC [0,1]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


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

    # Forward pass (extract intermediate representations)
    MODEL.eval()
    with torch.no_grad():
        class_logits, concept_logits, attn_weights = MODEL(img_tensor)
        concept_probs = MODEL.concept_activation(concept_logits)  # [1, num_concepts]

    # Prediction result
    if NUM_CLASSES == 1:
        prob = torch.sigmoid(class_logits).item()
        pred_label = "Malignant" if prob >= 0.5 else "Benign"
        prediction_text = f"**{pred_label}** (Malignancy probability: {prob:.4f})"
    else:
        probs = torch.softmax(class_logits, dim=-1).squeeze(0)
        top_k = min(3, NUM_CLASSES)
        top_probs, top_idxs = probs.topk(top_k)
        lines = []
        for i in range(top_k):
            idx = top_idxs[i].item()
            p = top_probs[i].item()
            name = TARGET_CLASSES[idx] if idx < len(TARGET_CLASSES) else f"Class {idx}"
            marker = "🔴" if i == 0 else "⚪"
            lines.append(f"{marker} **{name}**: {p:.4f}")
        prediction_text = "\n".join(lines)

    # Concept values
    concept_vals = concept_probs.squeeze(0).cpu().tolist()

    # Generate heatmaps
    img_np = _unnormalize_tensor(img_tensor.squeeze(0))
    _, _, H_img, W_img = img_tensor.shape

    attn_upsampled = F.interpolate(
        attn_weights, size=(H_img, W_img), mode='bilinear', align_corners=False
    )

    heatmap_gallery = []
    for group in CONCEPT_GROUPS:
        if group["type"] == "numerical":
            c_idx = group["flat_indices"][0]
            val = concept_vals[c_idx]
            # scale value to original physical range for visualization title
            orig_val = group["min"] + (group["max"] - group["min"]) * val
            if group["min"].is_integer() and group["max"].is_integer() and (group["max"] - group["min"]) > 2.0:
                orig_val = int(round(orig_val))
            else:
                orig_val = round(orig_val, 2)
            
            hm = attn_upsampled[0, c_idx].cpu().numpy()
            name = f"{group['name']}: {orig_val}"
            pil_img = _generate_single_heatmap(img_np, hm, name)
            heatmap_gallery.append((pil_img, name))
        else:
            # Categorical concept
            probs = [concept_vals[idx] for idx in group["flat_indices"]]
            max_idx = np.argmax(probs)
            max_c_idx = group["flat_indices"][max_idx]
            selected_class = group["classes"][max_idx]
            max_prob = probs[max_idx]
            
            hm = attn_upsampled[0, max_c_idx].cpu().numpy()
            name = f"{group['name']}: {selected_class} ({max_prob:.2f})"
            pil_img = _generate_single_heatmap(img_np, hm, name)
            heatmap_gallery.append((pil_img, name))

    # Intercept final block's attention weights to generate segmentation overlay
    attn = None
    use_mask = False
    threshold = 0.35
    
    if hasattr(MODEL, "backbone") and hasattr(MODEL.backbone, "vit"):
        attn = getattr(MODEL.backbone.vit.blocks[-1].attn, "last_attn_weights", None)
        use_mask = getattr(MODEL.backbone, "use_dino_mask", False)
        threshold = getattr(MODEL.backbone, "mask_threshold", 0.35)
        
    seg_pil = generate_segmentation_overlay(img_np, attn, use_mask, threshold)

    return prediction_text, concept_vals, heatmap_gallery, seg_pil, img_np


def repredict_with_adjusted_concepts(*component_values):
    """
    Re-run only the classifier head with user-adjusted concept values.
    This is the human-in-the-loop intervention point.
    """
    if MODEL is None:
        return "No model loaded."

    # Reconstruct flat concept tensor in [0, 1]
    concept_probs_list = [0.0] * NUM_CONCEPTS
    
    for i, group in enumerate(CONCEPT_GROUPS):
        val = component_values[i]
        if group["type"] == "numerical":
            # Scale down from [min, max] to [0, 1]
            min_val = group["min"]
            max_val = group["max"]
            norm_val = (float(val) - min_val) / (max_val - min_val + 1e-8)
            norm_val = max(0.0, min(1.0, norm_val))
            concept_probs_list[group["flat_indices"][0]] = norm_val
        else:
            # Categorical: val can be a list of strings (CheckboxGroup / Multi-select Dropdown) or a single string
            selected_classes = val if isinstance(val, list) else [val]
            selected_classes = {str(c) for c in selected_classes}
            
            for cls_idx, cls_str in zip(group["flat_indices"], group["classes"]):
                if cls_str in selected_classes:
                    concept_probs_list[cls_idx] = 1.0
                else:
                    concept_probs_list[cls_idx] = 0.0

    # If the model has latent concepts, pad them with zeros for classification
    if MODEL is not None and getattr(MODEL, "num_latent_concepts", 0) > 0:
        concept_probs_list.extend([0.0] * MODEL.num_latent_concepts)

    concept_probs = torch.tensor(
        [concept_probs_list], dtype=torch.float32, device=DEVICE
    )

    MODEL.eval()
    with torch.no_grad():
        class_logits = MODEL.classifier_head(concept_probs)

    if NUM_CLASSES == 1:
        prob = torch.sigmoid(class_logits).item()
        pred_label = "Malignant" if prob >= 0.5 else "Benign"
        return f"**{pred_label}** (Malignancy probability: {prob:.4f})"
    else:
        probs = torch.softmax(class_logits, dim=-1).squeeze(0)
        top_k = min(3, NUM_CLASSES)
        top_probs, top_idxs = probs.topk(top_k)
        lines = []
        for i in range(top_k):
            idx = top_idxs[i].item()
            p = top_probs[i].item()
            name = TARGET_CLASSES[idx] if idx < len(TARGET_CLASSES) else f"Class {idx}"
            marker = "🔴" if i == 0 else "⚪"
            lines.append(f"{marker} **{name}**: {p:.4f}")
        return "\n".join(lines)


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
    overflow-y: auto;
    padding: 4px !important;
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
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
        gr.HTML("<div class='main-title'>🔬 Attention-CBM Explorer</div>")
        gr.HTML("<div class='subtitle'>Interactive Concept Bottleneck Model — Inference & Human-in-the-Loop</div>")

        # ---- State for concept values ----
        concept_state = gr.State([])

        with gr.Row():
            # ==== Column 1: Input ====
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input Image")
                input_image = gr.Image(
                    type="pil",
                    label="Upload a dermoscopy image",
                    height=280
                )
                run_btn = gr.Button(
                    "🚀 Run Inference",
                    variant="primary",
                    size="lg"
                )

            # ==== Column 2: DINOv2 Foreground Segmentation Overlay ====
            with gr.Column(scale=1):
                gr.Markdown("### 👁️ DINOv2 Foreground Segmentation")
                seg_output = gr.Image(
                    label="Foreground Mask Overlay",
                    height=280,
                    interactive=False
                )
                
                gr.Markdown("### 🎯 Model Prediction")
                prediction_output = gr.Markdown(
                    value="*Upload an image and click Run Inference*"
                )

            # ==== Column 3: Heatmaps ====
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Concept Attention Heatmaps")
                heatmap_gallery = gr.Gallery(
                    label="Per-concept attention maps",
                    columns=4,
                    rows=2,
                    height=360,
                    object_fit="contain"
                )

        gr.Markdown("---")

        # ---- Human-in-the-Loop Section ----
        gr.Markdown("### 🎛️ Human-in-the-Loop: Concept Intervention")
        gr.Markdown(
            "Adjust the concept values below and click **Re-predict** to see how "
            "changes in individual concepts affect the final classification. "
            "This lets you explore the model's decision-making process interactively."
        )

        with gr.Row():
            with gr.Column(scale=3, elem_classes="concept-slider-group"):
                group_name_to_comp = {}
                num_cols = 3
                cols_groups = [CONCEPT_GROUPS[i::num_cols] for i in range(num_cols)]
                
                with gr.Row():
                    for col_idx in range(num_cols):
                        with gr.Column():
                            for group in cols_groups[col_idx]:
                                if group["type"] == "numerical":
                                    is_int = (group["min"].is_integer() and group["max"].is_integer() and (group["max"] - group["min"]) > 2.0)
                                    step = 1.0 if is_int else 0.01
                                    comp = gr.Slider(
                                        minimum=group["min"],
                                        maximum=group["max"],
                                        step=step,
                                        value=(group["min"] + group["max"]) / 2,
                                        label=group["name"],
                                        interactive=True,
                                        elem_classes="compact-comp"
                                    )
                                else:
                                    # Categorical Concept Component
                                    choices = group["classes"] + ["Not Visible / Occluded"]
                                    default_val = "Not Visible / Occluded"
                                    if len(group["classes"]) <= 3:
                                        comp = gr.Radio(
                                            choices=choices,
                                            value=default_val,
                                            label=group["name"],
                                            interactive=True,
                                            elem_classes="compact-comp"
                                        )
                                    else:
                                        comp = gr.Dropdown(
                                            choices=choices,
                                            value=default_val,
                                            label=group["name"],
                                            multiselect=False,
                                            interactive=True,
                                            elem_classes="compact-comp"
                                        )
                                group_name_to_comp[group["name"]] = comp
                
                # Reconstruct concept_components in the exact order of CONCEPT_GROUPS
                concept_components = [group_name_to_comp[g["name"]] for g in CONCEPT_GROUPS]

            with gr.Column(scale=1):
                repredict_btn = gr.Button(
                    "🔄 Re-predict with adjusted concepts",
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
                    *[gr.update() for _ in concept_components]
                )

            pred_text, concept_vals, gallery, seg_pil, _ = run_inference(image)

            # Build updates to reflect predicted values
            component_updates = []
            for group in CONCEPT_GROUPS:
                if group["type"] == "numerical":
                    val = concept_vals[group["flat_indices"][0]]
                    # Scale val [0, 1] to [min, max]
                    scaled_val = group["min"] + (group["max"] - group["min"]) * val
                    # Round value for clean display
                    if group["min"].is_integer() and group["max"].is_integer() and (group["max"] - group["min"]) > 2.0:
                        scaled_val = int(round(scaled_val))
                    else:
                        scaled_val = round(scaled_val, 4)
                    component_updates.append(gr.update(value=scaled_val))
                else:
                    # Categorical concept: select the highest probability class if above threshold,
                    # otherwise fallback to "Not Visible / Occluded" (all zeros)
                    probs = [concept_vals[idx] for idx in group["flat_indices"]]
                    max_idx = np.argmax(probs)
                    max_prob = probs[max_idx]
                    
                    # Threshold of 0.5 (confidence threshold for probability)
                    if max_prob <= 0.5:
                        selected_cls = "Not Visible / Occluded"
                    else:
                        selected_cls = group["classes"][max_idx]
                    
                    component_updates.append(gr.update(value=selected_cls))

            return (pred_text, concept_vals, gallery, seg_pil, *component_updates)

        run_btn.click(
            fn=on_inference,
            inputs=[input_image],
            outputs=[prediction_output, concept_state, heatmap_gallery, seg_output, *concept_components]
        )

        repredict_btn.click(
            fn=repredict_with_adjusted_concepts,
            inputs=concept_components,
            outputs=[adjusted_prediction]
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
        choices=['timm', 'clip']
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
    parser.add_argument(
        '--use_dino_mask', type=str2bool, default=None,
        help="Use DINOv2 self-attention to generate a silhouette foreground mask for background suppression"
    )
    parser.add_argument(
        '--dino_mask_threshold', type=float, default=None,
        help="Threshold to binarize DINOv2 attention map for silhouette mask"
    )
    parser.add_argument(
        '--no_grouping', action='store_true', default=False,
        help="Disable mutually exclusive concept grouping (defaults to auto-detecting from checkpoint)"
    )
    parser.add_argument(
        '--use_group_broadcasting', action='store_true', default=False,
        help="Use GroupToConceptAttention layout (group queries → independent BCE classifiers based on concept_config)"
    )
    parser.add_argument(
        '--lora_r', type=int, default=8
    )
    parser.add_argument(
        '--lora_alpha', type=float, default=16.0
    )
    parser.add_argument(
        '--port', type=int, default=7860,
        help="Port to serve the Gradio app on"
    )
    return parser.parse_args()

def main():
    global MODEL, DEVICE, CONCEPT_NAMES, CONCEPT_CONFIG, CONCEPT_GROUPS, TARGET_CLASSES, NUM_CONCEPTS, NUM_CLASSES

    args = parse_app_args()

    # 1. Load concept config
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
        if NUM_CLASSES == 20:
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
    
    # Auto-detect number of latent concepts from classifier_head weight shape
    latent_concepts = args.latent_concepts
    if "classifier_head.weight" in state_dict:
        checkpoint_dims = state_dict["classifier_head.weight"].shape[1]
        detected_latent = checkpoint_dims - NUM_CONCEPTS
        if detected_latent >= 0:
            latent_concepts = detected_latent
            print(f"🔮 Auto-detected latent concepts from checkpoint: {latent_concepts} (Total dimensions: {checkpoint_dims})")
        else:
            print(f"⚠️ Warning: Checkpoint dimensions ({checkpoint_dims}) are less than supervised concepts ({NUM_CONCEPTS}). Using args.latent_concepts={args.latent_concepts}.")
        print(f"⚠️ Warning: 'classifier_head.weight' not found in checkpoint. Using args.latent_concepts={args.latent_concepts}.")
    use_lora = getattr(args, 'use_lora', False)
    lora_r = getattr(args, 'lora_r', 8)
    lora_alpha = getattr(args, 'lora_alpha', 16.0)
    backbone_name = args.backbone_name
    backbone_type = args.backbone_type
    use_concept_groups = True
    use_cosine_attention = getattr(args, 'use_cosine_attention', False)
    use_group_broadcasting = getattr(args, 'use_group_broadcasting', False)
    use_dino_mask = False
    dino_mask_threshold = 0.35
    
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
        if 'use_dino_mask' in checkpoint_args:
            use_dino_mask = checkpoint_args['use_dino_mask']
            dino_mask_threshold = checkpoint_args.get('dino_mask_threshold', 0.35)
        print(f"🔮 Auto-detected Config from checkpoint args: backbone={backbone_name} ({backbone_type}), use_lora={use_lora}, r={lora_r}, alpha={lora_alpha}, use_concept_groups={use_concept_groups}, use_group_broadcasting={use_group_broadcasting}, use_dino_mask={use_dino_mask}")
    elif isinstance(loaded_checkpoint, dict) and 'config' in loaded_checkpoint:
        checkpoint_cfg = loaded_checkpoint['config']
        bb_cfg = checkpoint_cfg.get('backbone', {})
        ds_cfg = checkpoint_cfg.get('dataset', {})
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
        if 'use_dino_mask' in bb_cfg:
            use_dino_mask = bb_cfg['use_dino_mask']
            dino_mask_threshold = bb_cfg.get('dino_mask_threshold', 0.35)
        print(f"🔮 Auto-detected Config from checkpoint config: backbone={backbone_name} ({backbone_type}), use_lora={use_lora}, r={lora_r}, alpha={lora_alpha}, use_concept_groups={use_concept_groups}, use_group_broadcasting={use_group_broadcasting}, use_dino_mask={use_dino_mask}")
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
        print(f"🔮 Auto-detected from state_dict keys: backbone={backbone_name}, use_lora={use_lora}, use_cosine_attention={use_cosine_attention}, use_concept_groups=True, use_group_broadcasting={use_group_broadcasting}, use_dino_mask={use_dino_mask}")
    
    # Command line argument override
    if getattr(args, 'no_grouping', False):
        use_concept_groups = False
        print("🔮 Command line override: Disabling concept grouping.")
    if getattr(args, 'use_dino_mask', None) is not None:
        use_dino_mask = args.use_dino_mask
    if getattr(args, 'dino_mask_threshold', None) is not None:
        dino_mask_threshold = args.dino_mask_threshold

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
            print(f"🔮 Configured Group-level Softmax over {grouped_count} mutually exclusive groups out of {len(CONCEPT_GROUPS)} total categories.")
    
    if not use_concept_groups or concept_groups_info is None:
        concept_groups_info = None
        USE_CONCEPT_GROUPS = False
        print("🔮 Group-level Softmax Activation is DISABLED (Sigmoid activation fallback active).")

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
        print(f"🔮 Group Broadcasting: {num_groups} anatomical groups → {NUM_CONCEPTS} independent BCE classifiers (Group Softmax disabled).")

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
        use_dino_mask=use_dino_mask,
        dino_mask_threshold=dino_mask_threshold
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
                print(f"  🔄 Migrated in_proj_weight → q/k/v_proj.weight  ({prefix})")
            elif ".cross_attention.in_proj_bias" in k:
                # Bias is not used in the new projections (bias=False); skip.
                print(f"  ⏭️  Skipped in_proj_bias (new proj layers have no bias): {k}")
            elif ".cross_attention.out_proj.weight" in k:
                new_k = k.replace(".cross_attention.out_proj.weight", ".out_proj.weight")
                migrated[new_k] = v
                print(f"  🔄 Migrated out_proj.weight: {k} → {new_k}")
            elif ".cross_attention.out_proj.bias" in k:
                # out_proj bias not present in new arch; skip.
                print(f"  ⏭️  Skipped out_proj.bias (new out_proj has no bias): {k}")
            else:
                migrated[k] = v
        return migrated

    old_keys = {k for k in state_dict if ".cross_attention." in k}
    if old_keys:
        print(f"⚠️  Checkpoint contains {len(old_keys)} legacy MHA key(s). Running migration…")
        state_dict = _migrate_state_dict(state_dict)
        print("✅  State-dict migration complete.")

    # Load with strict=False so that new parameters (temperature, …) that are
    # absent from old checkpoints are left at their default initialization.
    missing, unexpected = MODEL.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"ℹ️  New parameters not found in checkpoint (initialized from scratch): {missing}")
    if unexpected:
        print(f"⚠️  Unexpected keys ignored during loading: {unexpected}")
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
