import io
import math
from typing import List
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def generate_concept_heatmaps(
    image_tensor: torch.Tensor,       # [B, 3, H, W]
    attn_weights: torch.Tensor,       # [B, num_concepts, H_attn, W_attn]
    concept_names: List[str],
    colormap: str = 'jet',
    alpha: float = 0.5
) -> List[Image.Image]:
    """
    Generates concept attention heatmap overlays on original normalized images.
    
    Args:
        image_tensor: Normalized input images of shape [B, 3, H, W].
        attn_weights: Concept attention weights of shape [B, num_concepts, H_attn, W_attn].
        concept_names: List of concept names of length num_concepts.
        colormap: Matplotlib colormap name to use for the heatmap overlay (e.g., 'jet', 'viridis').
        alpha: Blending transparency factor for the heatmap overlay.
        
    Returns:
        List of PIL Images, each representing a grid plot of the original image 
        and its respective concept-specific attention overlays.
    """
    B, C_img, H_img, W_img = image_tensor.shape
    _, num_concepts, H_attn, W_attn = attn_weights.shape
    
    # 1. Reverse ImageNet Normalization
    # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)  # [1, 3, 1, 1]
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)   # [1, 3, 1, 1]
    
    # [B, 3, H, W]
    unnormalized = image_tensor * std + mean
    unnormalized = torch.clamp(unnormalized, 0.0, 1.0)
    
    # 2. Upsample Attention weights to Image size
    # [B, num_concepts, H_img, W_img]
    attn_upsampled = F.interpolate(
        attn_weights,
        size=(H_img, W_img),
        mode='bilinear',
        align_corners=False
    )
    
    pil_images = []
    
    # 3. Process each image in the batch
    for b in range(B):
        # Extract un-normalized RGB image: [3, H_img, W_img] -> [H_img, W_img, 3]
        img_np = unnormalized[b].permute(1, 2, 0).cpu().numpy()
        
        # Calculate dynamic subplot grid layout (max 8 columns per row)
        cols = min(8, 1 + num_concepts)
        rows = math.ceil((1 + num_concepts) / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        # Plot Original un-normalized Image in the first subplot
        axes_flat[0].imshow(img_np)
        axes_flat[0].set_title("Original Image", fontsize=10, fontweight='bold')
        axes_flat[0].axis('off')
        
        # Plot each concept's upsampled attention heatmap overlay
        for c in range(num_concepts):
            ax = axes_flat[c + 1]
            
            # Extract heatmap slice: [H_img, W_img]
            heatmap = attn_upsampled[b, c].cpu().numpy()
            
            # Min-Max normalization of the heatmap for visual enhancement
            h_min, h_max = heatmap.min(), heatmap.max()
            if h_max > h_min:
                heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)
            else:
                heatmap = np.zeros_like(heatmap)
                
            # Plot the base image
            ax.imshow(img_np)
            # Overlay the colormapped heatmap
            ax.imshow(heatmap, cmap=colormap, alpha=alpha)
            
            # Set concept name as title
            concept_title = concept_names[c]
            # Truncate title if it's too long for a subplot
            if len(concept_title) > 25:
                concept_title = concept_title[:22] + "..."
            ax.set_title(concept_title, fontsize=9)
            ax.axis('off')
            
        # Hide any unused subplots
        for i in range(1 + num_concepts, len(axes_flat)):
            axes_flat[i].axis('off')
            
        plt.tight_layout()
        
        # Convert Matplotlib figure directly to PIL Image using a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        pil_img = Image.open(buf)
        pil_images.append(pil_img)
        
    return pil_images
