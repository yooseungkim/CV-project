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
        concept_probs = torch.sigmoid(concept_logits)  # [1, num_concepts]

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

    return prediction_text, concept_vals, heatmap_gallery, img_np


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
            # Categorical dropdown: val is a string representing the selected class
            # One-hot encode
            selected_cls = str(val)
            for cls_idx, cls_str in zip(group["flat_indices"], group["classes"]):
                if cls_str == selected_cls:
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
.concept-slider-group {
    max-height: 420px;
    overflow-y: auto;
    padding: 6px !important;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background-color: #f9fafb;
}
/* Extremely tight padding for components in the grid */
.concept-slider-group .compact-comp {
    padding: 2px 6px !important;
    margin: 0 !important;
    background-color: #ffffff;
}
/* Reduce default Gradio block paddings/margins inside container */
.concept-slider-group .block {
    padding: 4px 6px !important;
    margin: 1px 0 !important;
    min-width: 0 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
/* Make labels small, bold, and clean */
.concept-slider-group label span {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    margin-bottom: 1px !important;
    color: #374151 !important;
}
/* Tight range inputs */
.concept-slider-group input[type="range"] {
    margin-top: 1px !important;
    margin-bottom: 1px !important;
    height: 4px !important;
}
/* Tight dropdown fields */
.concept-slider-group .wrap {
    padding: 1px 4px !important;
    font-size: 0.8rem !important;
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
            # ==== Left Column: Input ====
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input Image")
                input_image = gr.Image(
                    type="pil",
                    label="Upload a dermoscopy image",
                    height=300
                )
                run_btn = gr.Button(
                    "🚀 Run Inference",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("### 🎯 Model Prediction")
                prediction_output = gr.Markdown(
                    value="*Upload an image and click Run Inference*"
                )

            # ==== Right Column: Heatmaps ====
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Concept Attention Heatmaps")
                heatmap_gallery = gr.Gallery(
                    label="Per-concept attention maps",
                    columns=4,
                    rows=2,
                    height=500,
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
                col1_groups = CONCEPT_GROUPS[::2]
                col2_groups = CONCEPT_GROUPS[1::2]
                
                with gr.Row():
                    with gr.Column():
                        for group in col1_groups:
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
                                comp = gr.Dropdown(
                                    choices=group["classes"],
                                    value=group["classes"][0],
                                    label=group["name"],
                                    interactive=True,
                                    elem_classes="compact-comp"
                                )
                            group_name_to_comp[group["name"]] = comp
                            
                    with gr.Column():
                        for group in col2_groups:
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
                                comp = gr.Dropdown(
                                    choices=group["classes"],
                                    value=group["classes"][0],
                                    label=group["name"],
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
                    *[gr.update() for _ in concept_components]
                )

            pred_text, concept_vals, gallery, _ = run_inference(image)

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
                    # Find highest probability index
                    probs = [concept_vals[idx] for idx in group["flat_indices"]]
                    max_idx = np.argmax(probs)
                    selected_cls = group["classes"][max_idx]
                    component_updates.append(gr.update(value=selected_cls))

            return (pred_text, concept_vals, gallery, *component_updates)

        run_btn.click(
            fn=on_inference,
            inputs=[input_image],
            outputs=[prediction_output, concept_state, heatmap_gallery, *concept_components]
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
    else:
        print(f"⚠️ Warning: 'classifier_head.weight' not found in checkpoint. Using args.latent_concepts={args.latent_concepts}.")

    # 3. Initialize model with correct parameters
    MODEL = UniversalFlexibleCBM(
        backbone_type=args.backbone_type,
        backbone_name=args.backbone_name,
        num_supervised_concepts=NUM_CONCEPTS,
        num_classes=NUM_CLASSES,
        num_latent_concepts=latent_concepts
    )

    MODEL.load_state_dict(state_dict)
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
