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
        features = MODEL.backbone(img_tensor)
        if isinstance(features, tuple):
            features = features[0]

        concept_logits, attn_weights = MODEL.concept_attention(features)
        concept_probs = torch.sigmoid(concept_logits)  # [1, num_concepts]
        class_logits = MODEL.classifier_head(concept_probs)  # [1, num_classes]

    # Prediction result
    if NUM_CLASSES == 1:
        prob = torch.sigmoid(class_logits).item()
        pred_label = "Malignant" if prob >= 0.5 else "Benign"
        prediction_text = f"**{pred_label}** (Malignancy probability: {prob:.4f})"
    else:
        probs = torch.softmax(class_logits, dim=-1).squeeze(0)
        pred_idx = probs.argmax().item()
        prediction_text = f"**Predicted class: {pred_idx}** (confidence: {probs[pred_idx]:.4f})"

    # Concept values
    concept_vals = concept_probs.squeeze(0).cpu().tolist()

    # Generate heatmaps
    img_np = _unnormalize_tensor(img_tensor.squeeze(0))
    _, _, H_img, W_img = img_tensor.shape

    attn_upsampled = F.interpolate(
        attn_weights, size=(H_img, W_img), mode='bilinear', align_corners=False
    )

    heatmap_gallery = []
    for c in range(NUM_CONCEPTS):
        hm = attn_upsampled[0, c].cpu().numpy()
        name = CONCEPT_NAMES[c] if c < len(CONCEPT_NAMES) else f"Concept {c}"
        pil_img = _generate_single_heatmap(img_np, hm, name)
        heatmap_gallery.append((pil_img, name))

    return prediction_text, concept_vals, heatmap_gallery, img_np


def repredict_with_adjusted_concepts(*slider_values):
    """
    Re-run only the classifier head with user-adjusted concept values.
    This is the human-in-the-loop intervention point.
    """
    if MODEL is None:
        return "No model loaded."

    # Build concept tensor from slider values
    concept_probs = torch.tensor(
        [list(slider_values)], dtype=torch.float32, device=DEVICE
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
        pred_idx = probs.argmax().item()
        return f"**Predicted class: {pred_idx}** (confidence: {probs[pred_idx]:.4f})"


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
    max-height: 500px;
    overflow-y: auto;
    padding: 8px;
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
                # Build sliders dynamically based on concepts
                concept_sliders = []
                for i, name in enumerate(CONCEPT_NAMES):
                    slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.5,
                        label=name,
                        interactive=True
                    )
                    concept_sliders.append(slider)

            with gr.Column(scale=1):
                repredict_btn = gr.Button(
                    "🔄 Re-predict with adjusted concepts",
                    variant="secondary",
                    size="lg"
                )
                adjusted_prediction = gr.Markdown(
                    value="*Adjust sliders and click Re-predict*"
                )

        # ---- Event Handlers ----
        def on_inference(image):
            if image is None:
                return (
                    "*Please upload an image first.*",
                    gr.update(),
                    [],
                    *[gr.update() for _ in concept_sliders]
                )

            pred_text, concept_vals, gallery, _ = run_inference(image)

            # Build slider updates to reflect predicted values
            slider_updates = []
            for i in range(len(concept_sliders)):
                val = concept_vals[i] if i < len(concept_vals) else 0.5
                slider_updates.append(gr.update(value=round(val, 4)))

            return (pred_text, concept_vals, gallery, *slider_updates)

        run_btn.click(
            fn=on_inference,
            inputs=[input_image],
            outputs=[prediction_output, concept_state, heatmap_gallery, *concept_sliders]
        )

        repredict_btn.click(
            fn=repredict_with_adjusted_concepts,
            inputs=concept_sliders,
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
        '--port', type=int, default=7860,
        help="Port to serve the Gradio app on"
    )
    return parser.parse_args()


def main():
    global MODEL, DEVICE, CONCEPT_NAMES, CONCEPT_CONFIG, NUM_CONCEPTS, NUM_CLASSES

    args = parse_app_args()

    # 1. Load concept config
    if not os.path.exists(args.concept_config_path):
        raise FileNotFoundError(f"Concept config not found: {args.concept_config_path}")

    with open(args.concept_config_path, 'r', encoding='utf-8') as f:
        CONCEPT_CONFIG = json.load(f)

    # Build concepts_flat (same logic as dataset classes)
    concepts_flat = []
    total_dims = 0
    for name, info in CONCEPT_CONFIG.items():
        ctype = info.get("type", "numerical")
        if ctype == "categorical":
            classes = info.get("classes", [])
            for cls_val in classes:
                concepts_flat.append(f"{name}_{cls_val}")
            total_dims += len(classes)
        else:
            concepts_flat.append(name)
            total_dims += 1

    CONCEPT_NAMES = concepts_flat
    NUM_CONCEPTS = total_dims
    NUM_CLASSES = args.num_classes

    print(f"Loaded {NUM_CONCEPTS} concepts from {args.concept_config_path}")

    # 2. Initialize model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL = UniversalFlexibleCBM(
        backbone_type=args.backbone_type,
        backbone_name=args.backbone_name,
        num_concepts=NUM_CONCEPTS,
        num_classes=NUM_CLASSES
    )

    # 3. Load checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
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
