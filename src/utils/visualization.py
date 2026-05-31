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
    attn_weights: torch.Tensor,       # [B, num_concepts, H_attn, W_attn] or [B, num_concepts] (max_indices)
    concept_names: List[str],
    colormap: str = 'jet',
    alpha: float = 0.5
) -> List[Image.Image]:
    """
    Generates concept attention heatmap overlays on original normalized images.
    Supports both dense attention weights and 1D patch indices (max_indices).
    
    Args:
        image_tensor: Normalized input images of shape [B, 3, H, W].
        attn_weights: Concept attention weights or winning patch indices.
        concept_names: List of concept names of length num_concepts.
        colormap: Matplotlib colormap name to use for the heatmap overlay.
        alpha: Blending transparency factor for the heatmap overlay.
        
    Returns:
        List of PIL Images, each representing a grid plot of the original image 
        and its respective concept-specific attention overlays.
    """
    B, C_img, H_img, W_img = image_tensor.shape
    
    # 1. Resolve inputs: extract max_indices if input is 4D attention maps
    if attn_weights.dim() == 2:
        max_indices = attn_weights
        num_concepts = max_indices.shape[1]
        max_idx_val = max_indices.max().item()
        # Find smallest perfect square H_attn^2 > max_idx_val, at least 14
        H_attn = max(int(math.ceil(math.sqrt(max_idx_val + 1))), 14)
        N_patches = H_attn * H_attn
    elif attn_weights.dim() == 4:
        B_att, num_concepts, H_attn, W_attn = attn_weights.shape
        N_patches = H_attn * W_attn
        # Flatten spatial dimensions to find the peak indices (values 0..N_patches-1)
        flat_attn = attn_weights.view(B_att, num_concepts, -1)
        max_indices = torch.argmax(flat_attn, dim=-1)
    else:
        raise ValueError(f"Unsupported attn_weights shape: {attn_weights.shape}")
        
    # 2. Sparse Map Generation: Create a dynamically-sized H_attn x H_attn grid of zeros and set peak index to 1.0
    device = max_indices.device
    sparse_maps = torch.zeros(B, num_concepts, N_patches, device=device)
    sparse_maps.scatter_(2, max_indices.unsqueeze(-1), 1.0)
    sparse_maps = sparse_maps.view(B, num_concepts, H_attn, H_attn)
    
    # 3. Smoothing: Apply Gaussian blur to the 14x14 grid
    from torchvision.transforms.functional import gaussian_blur
    smoothed_maps = gaussian_blur(sparse_maps, kernel_size=[3, 3], sigma=[1.0, 1.0])
    
    # 4. Upsampling: Resize to original image size [H_img, W_img] using bicubic interpolation
    attn_upsampled = F.interpolate(
        smoothed_maps,
        size=(H_img, W_img),
        mode='bicubic',
        align_corners=False
    )
    attn_upsampled = torch.clamp(attn_upsampled, min=0.0)
    
    # 5. Reverse ImageNet Normalization
    # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)  # [1, 3, 1, 1]
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)   # [1, 3, 1, 1]
    
    # [B, 3, H, W]
    unnormalized = image_tensor * std + mean
    unnormalized = torch.clamp(unnormalized, 0.0, 1.0)
    
    pil_images = []
    
    # 6. Process each image in the batch and overlay colormap
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
