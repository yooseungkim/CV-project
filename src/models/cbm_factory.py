import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import open_clip
import math
from typing import Optional, List, Tuple

# ANSI terminal colors for highlighting
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

class GatedSparseNAMHead(nn.Module):
    def __init__(self, num_concepts: int = 312, num_classes: int = 200, 
                 hidden_dim: int = 64, num_latent_concepts: int = 0):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_latent_concepts = num_latent_concepts
        
        # Parallel MLPs using Grouped Conv1d (groups=num_concepts)
        self.conv1 = nn.Conv1d(
            in_channels=num_concepts,
            out_channels=num_concepts * hidden_dim,
            kernel_size=1,
            groups=num_concepts
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=num_concepts * hidden_dim,
            out_channels=num_concepts * num_classes,
            kernel_size=1,
            groups=num_concepts
        )
        
        # Learnable gating parameter initialized to 1.0
        self.concept_gates = nn.Parameter(torch.ones(num_concepts))
        
        if self.num_latent_concepts > 0:
            self.latent_linear = nn.Linear(num_latent_concepts, num_classes)
        else:
            self.latent_linear = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        supervised_x = x[:, :self.num_concepts].unsqueeze(-1)
        
        h = F.relu(self.conv1(supervised_x))
        y = self.conv2(h) # Shape: [Batch, num_concepts * num_classes, 1]
        
        y = y.view(batch_size, self.num_concepts, self.num_classes)
        gated_y = y * self.concept_gates.view(1, self.num_concepts, 1)
        supervised_out = gated_y.sum(dim=1)
        
        if self.num_latent_concepts > 0 and self.latent_linear is not None:
            latent_x = x[:, self.num_concepts:]
            latent_out = self.latent_linear(latent_x)
            return supervised_out + latent_out
            
        return supervised_out

    def get_sparsity_loss(self) -> torch.Tensor:
        return torch.sum(torch.abs(self.concept_gates))

class ConceptAttentionLayer(nn.Module):
    def __init__(self, feature_dim: int, num_concepts: int, num_heads: int = 4):
        """Concept-specific Spatial Cosine Attention Layer for CNN (ResNet) backbones.
        Replaces dot-product attention with L2-normalized cosine attention to suppress
        high-norm border-patch outliers produced by DINOv2 on tightly cropped images.
        Each concept learns its own unique 1x1 conv mapping to spatial location and
        individual feature projection.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.feature_dim = feature_dim
        
        # 1x1 Conv to produce per-concept spatial query vectors
        self.attention_conv = nn.Conv2d(feature_dim, num_concepts, kernel_size=1)
        
        # Concept-specific weight projections: [num_concepts, feature_dim]
        self.concept_proj = nn.Parameter(torch.randn(num_concepts, feature_dim))
        self.concept_bias = nn.Parameter(torch.zeros(num_concepts))
        
        # Learnable temperature for cosine similarity scaling (init=1 -> no-op at start)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, features: torch.Tensor):
        # features: [B, C, H, W]
        B, C, H, W = features.shape
        
        # 1. Compute per-concept spatial attention via Cosine Attention
        # Produce raw logit maps and L2-normalize along the channel dimension
        attn_logits = self.attention_conv(features)  # [B, num_concepts, H, W]
        # L2-normalize feature patches along channel dim to remove magnitude bias
        features_flat = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        features_norm = F.normalize(features_flat, p=2, dim=-1)           # [B, H*W, C]
        # L2-normalize the attention conv weights (used as per-concept queries)
        attn_queries = self.attention_conv.weight.squeeze(-1).squeeze(-1)  # [num_concepts, C]
        attn_queries_norm = F.normalize(attn_queries, p=2, dim=-1)         # [num_concepts, C]
        # Cosine similarity: [B, num_concepts, H*W]
        cosine_logits = torch.bmm(
            attn_queries_norm.unsqueeze(0).expand(B, -1, -1),  # [B, num_concepts, C]
            features_norm.transpose(1, 2)                       # [B, C, H*W]
        ) / self.temperature.clamp(min=1e-4)                   # scale by learnable T
        
        # Softmax over spatial dimension to get attention weights
        attn_weights = torch.softmax(cosine_logits, dim=-1)               # [B, num_concepts, H*W]
        attn_weights_2d = attn_weights.view(B, self.num_concepts, H, W)   # [B, num_concepts, H, W]
        
        # 2. Weighted Sum: [B, num_concepts, H*W] x [B, H*W, C] -> [B, num_concepts, C]
        weighted_features = torch.bmm(attn_weights, features_norm)  # [B, num_concepts, C]
        
        # 3. Concept Logits via cosine similarity with learnable concept projections
        concept_proj_norm = F.normalize(self.concept_proj, p=2, dim=-1)  # [num_concepts, C]
        concept_logits = (
            (weighted_features * concept_proj_norm.unsqueeze(0)).sum(dim=-1)
            / self.temperature.clamp(min=1e-4)
            + self.concept_bias.unsqueeze(0)
        )  # [B, num_concepts]
        
        return concept_logits, attn_weights_2d, weighted_features


class GroupSoftmaxActivation(nn.Module):
    def __init__(self, groups_info: List[Tuple[int, int]]):
        """Dynamic Group-level Softmax Activation for mutually exclusive conceptual attributes.
        Applies softmax within multi-class concept groups to enforce probability sum bounds to exactly 1.0,
        while falling back to standard sigmoid for numerical/binary single-dimension features.
        Any dimensions beyond the configured groups (e.g., latent concepts) are automatically
        activated using standard Sigmoid to preserve downstream dimensionality integrity.
        groups_info: List of (start_idx, num_feats) representing attribute groupings.
        """
        super().__init__()
        self.groups_info = groups_info
        # Calculate the total number of supervised concepts covered by the groups
        self.total_group_feats = sum(num_feats for _, num_feats in groups_info) if groups_info else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, total_concepts] (supervised + latent)
        outputs = []
        for start_idx, num_feats in self.groups_info:
            group_logits = x[:, start_idx : start_idx + num_feats]
            if num_feats > 1:
                # Mutually exclusive attribute groups
                group_probs = torch.softmax(group_logits, dim=-1)
            else:
                # Binary/numerical fallback
                group_probs = torch.sigmoid(group_logits)
            outputs.append(group_probs)
            
        # If there are latent or remaining concepts beyond the defined groups, apply sigmoid fallback
        if x.shape[1] > self.total_group_feats:
            remaining_logits = x[:, self.total_group_feats:]
            remaining_probs = torch.sigmoid(remaining_logits)
            outputs.append(remaining_probs)
            
        return torch.cat(outputs, dim=1)


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.05):
        """Low-Rank Adaptation (LoRA) linear wrapper module.
        Freezes the original_linear projection layer and adds parallel learnable rank-r adapters.
        """
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Original projection layer parameters are strictly frozen
        for param in self.original_linear.parameters():
            param.requires_grad = False
            
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # Trainable low-rank adapter weights
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # Initialize A to uniform kaiming and B to zero so that the initial output mismatch is exactly 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original_linear(x)
        lora_output = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return original_output + lora_output * self.scaling


def inject_lora_to_vit(model: nn.Module, r: int = 8, lora_alpha: float = 16.0) -> list:
    """Recursively replaces standard nn.Linear layers inside vit blocks' qkv attention with LoRALinear modules."""
    injected_modules = []
    
    # Traverse named modules to find and surgical replace attention projections
    for name, module in model.named_modules():
        if name.endswith('.attn'):
            if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
                original_qkv = module.qkv
                lora_qkv = LoRALinear(original_qkv, r=r, lora_alpha=lora_alpha)
                module.qkv = lora_qkv
                injected_modules.append(lora_qkv)
                
    print(f"{BOLD}{GREEN}[LoRA Injector]{RESET} Successfully injected LoRA (r={r}, alpha={lora_alpha}) into {len(injected_modules)} ViT attention blocks.")
    return injected_modules


class ViTBackboneWrapper(nn.Module):
    def __init__(self, vit_model, use_dino_mask: bool = False, mask_threshold: float = 0.35):
        """Wrapper for Vision Transformer to dynamically extract patch tokens while ignoring the CLS token.
        Ensures compatibility with internal feature representations and main training pipelines.
        Supports DINOv2 attention silhouette foreground masking.
        """
        super().__init__()
        self.vit = vit_model
        self.use_dino_mask = use_dino_mask
        self.mask_threshold = mask_threshold
        
        if self.use_dino_mask:
            if hasattr(self.vit, "blocks") and len(self.vit.blocks) > 0 and hasattr(self.vit.blocks[-1], "attn"):
                attn_module = self.vit.blocks[-1].attn
                attn_module.fused_attn = False  # Disable fused attention to force explicit weight computation
                
                import types
                def custom_forward(attn_self, x, attn_mask=None, is_causal=False):
                    B, N, C = x.shape
                    qkv = attn_self.qkv(x).reshape(B, N, 3, attn_self.num_heads, attn_self.head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    q, k = attn_self.q_norm(q), attn_self.k_norm(k)
                    
                    q = q * attn_self.scale
                    attn = q @ k.transpose(-2, -1)
                    
                    if attn_mask is not None:
                        from timm.layers.attention import resolve_self_attn_mask, maybe_add_mask
                        attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
                        attn = maybe_add_mask(attn, attn_bias)
                        
                    attn = attn.softmax(dim=-1)
                    attn_self.last_attn_weights = attn
                    
                    attn = attn_self.attn_drop(attn)
                    x = attn @ v
                    
                    x = x.transpose(1, 2).reshape(B, N, attn_self.attn_dim)
                    x = attn_self.norm(x)
                    x = attn_self.proj(x)
                    x = attn_self.proj_drop(x)
                    return x
                
                attn_module.forward = types.MethodType(custom_forward, attn_module)
                print(f"{BOLD}{GREEN}[DINOv2 Masking]{RESET} Successfully patched final attention block of {self.vit.__class__.__name__} to extract self-attention maps (threshold={mask_threshold}).")
            else:
                print(f"{BOLD}{YELLOW}[DINOv2 Masking]{RESET} The backbone does not support attention patching (missing 'blocks' or 'attn'). Masking disabled.")
                self.use_dino_mask = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        if self.use_dino_mask:
            # ── PASS 1: Extract DINOv2 self-attention map ──
            # Run a forward pass to capture the attention maps of the final block
            with torch.no_grad():
                _ = self.vit.forward_features(x)
                
            attn = getattr(self.vit.blocks[-1].attn, "last_attn_weights", None)
            if attn is not None:
                # attn shape: [B, num_heads, N_tokens, N_tokens]
                # CLS attention to all other patch tokens is at index 0 (CLS) to index 1: (all other tokens)
                cls_attn = attn[:, :, 0, 1:]  # [B, num_heads, N_patches]
                mean_attn = cls_attn.mean(dim=1)  # [B, N_patches]
                
                # Min-max normalization per-image to [0, 1] range to handle dynamic scale differences
                min_val = mean_attn.min(dim=1, keepdim=True)[0]
                max_val = mean_attn.max(dim=1, keepdim=True)[0]
                norm_attn = (mean_attn - min_val) / (max_val - min_val + 1e-8)  # [B, 196]
                
                # Reshape to [B, 1, grid, grid] assuming grid x grid patches
                grid_size = int(norm_attn.shape[1] ** 0.5)
                norm_attn_grid = norm_attn.view(B, 1, grid_size, grid_size)
                
                # Upsample the mask to match raw image resolution [B, 1, H, W]
                segmentation_mask = F.interpolate(
                    norm_attn_grid, 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Binarize mask using the configured threshold
                segmentation_mask = (segmentation_mask > self.mask_threshold).float()  # [B, 1, H, W]
                
                # Apply background blur to the input image batch x
                import torchvision.transforms.functional as TF
                # kernel_size must be odd
                blurred_x = TF.gaussian_blur(x, kernel_size=[21, 21])
                
                # Blend foreground (original) and background (blurred)
                blurred_input = (x * segmentation_mask) + (blurred_x * (1.0 - segmentation_mask))
                
                # ── PASS 2: Extract features from the blended image ──
                feats = self.vit.forward_features(blurred_input)
            else:
                # Fallback if attention map could not be captured
                feats = self.vit.forward_features(x)
        else:
            feats = self.vit.forward_features(x)
            
        if isinstance(feats, tuple):
            feats = feats[0]
            
        # Assert type and shape safety to prevent silencing subtle shape bugs or non-standard timm outputs
        assert isinstance(feats, torch.Tensor), f"Expected backbone features to be a torch.Tensor, but got {type(feats)}"
        assert len(feats.shape) == 3, (
            f"Expected 3D patch token tensor [B, N_tokens, C] from ViT backbone, but got shape {feats.shape}. "
            "Ensure global pooling is disabled in timm and your model returns sequence-level features."
        )
        
        patch_features = feats[:, 1:]
        return patch_features



class ViTCrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_concepts: int, num_heads: int = 4,
                 use_cosine_attention: bool = False):
        """Concept-specific Multihead Cross-Attention Layer for ViT backbones.

        Two attention modes are supported via `use_cosine_attention`:
          - False (default): Standard nn.MultiheadAttention (stable, pretrained-checkpoint compatible).
          - True: L2-normalized Cosine Attention with learnable temperature, which suppresses
            the high-norm border-patch outliers produced by DINOv2 on tightly cropped images
            ("Vision Transformers Need Registers", ICLR 2024).
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim
        self.use_cosine_attention = use_cosine_attention
        
        # Learnable concept queries: [1, num_concepts, embed_dim]
        self.concept_queries = nn.Parameter(torch.randn(1, num_concepts, embed_dim))
        
        # Concept-specific projections & biases: [num_concepts, embed_dim]
        self.concept_proj = nn.Parameter(torch.randn(num_concepts, embed_dim))
        self.concept_bias = nn.Parameter(torch.zeros(num_concepts))
        
        # Surgical initialization of bias to prevent Focal Loss logit explosion (RetinaNet Prior pi=0.01)
        pi = 0.01
        bias_init = -math.log((1 - pi) / pi)
        nn.init.constant_(self.concept_bias, bias_init)

        if use_cosine_attention:
            # Explicit Q / K / V projections for cosine attention path
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            # Learnable temperature (init=1 -> no-op at start)
            self.temperature = nn.Parameter(torch.ones(1))
            for linear in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                nn.init.xavier_uniform_(linear.weight)
        else:
            # Standard multihead cross-attention
            self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, features: torch.Tensor):
        # features: patch tokens [B, N_patches, embed_dim]
        B, N, D = features.shape
        queries = self.concept_queries.expand(B, -1, -1)  # [B, num_concepts, D]

        if self.use_cosine_attention:
            # --- Cosine Attention path ---
            Q = self.q_proj(queries)   # [B, num_concepts, D]
            K = self.k_proj(features)  # [B, N, D]
            V = self.v_proj(features)  # [B, N, D]

            Q_norm = F.normalize(Q, p=2, dim=-1)
            K_norm = F.normalize(K, p=2, dim=-1)

            attn_scores = torch.bmm(Q_norm, K_norm.transpose(1, 2)) / self.temperature.clamp(min=1e-4)
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, num_concepts, N]
            attn_out = self.out_proj(torch.bmm(attn_weights, V))  # [B, num_concepts, D]

            concept_proj_norm = F.normalize(self.concept_proj, p=2, dim=-1)
            attn_out_norm = F.normalize(attn_out, p=2, dim=-1)
            concept_logits = (
                (attn_out_norm * concept_proj_norm.unsqueeze(0)).sum(dim=-1)
                / self.temperature.clamp(min=1e-4)
                + self.concept_bias.unsqueeze(0)
            )
        else:
            # --- Standard MultiheadAttention path ---
            attn_out, attn_weights = self.cross_attention(
                query=queries,
                key=features,
                value=features
            )  # attn_out: [B, num_concepts, D], attn_weights: [B, num_concepts, N]

            concept_logits = (
                (attn_out * self.concept_proj.unsqueeze(0)).sum(dim=-1)
                + self.concept_bias.unsqueeze(0)
            )

        # Reshape attention weights: [B, num_concepts, N] -> [B, num_concepts, sqrt(N), sqrt(N)]
        H_attn = int(math.sqrt(N))
        attn_weights_2d = attn_weights.view(B, self.num_concepts, H_attn, H_attn)

        return concept_logits, attn_weights_2d, attn_out


class GroupToConceptAttention(nn.Module):
    """Attention Broadcasting: Spatial Localization (28 groups) → Semantic Classification (312 concepts).

    Architecture (3-step pipeline):
      Step 1 — Spatial Localization:
        28 learnable group queries attend to DINOv2 patch tokens via Cosine Attention.
        Output: group_features [B, 28, embed_dim]

      Step 2 — Attention Broadcasting:
        group_features are index-selected via `group_mapping` (int array of length 312,
        each value in [0..27]) to produce concept_features [B, 312, embed_dim].
        Concepts sharing the same anatomical part share the same spatial feature —
        but each has its own independent classifier below.

      Step 3 — Independent Classification:
        Each of the 312 concepts has its own Linear(embed_dim → 1) with a dedicated bias.
        Outputs raw logits suitable for BCEWithLogitsLoss (no sigmoid/softmax applied here).

    This design separates "where to look" (group attention) from "what is present"
    (per-concept binary classifiers), fixing both TPR Collapse (multi-label) and
    TNR Collapse (occlusion / all-zero GT) caused by Group Softmax.
    """

    def __init__(self, embed_dim: int, num_groups: int, num_concepts: int,
                 group_mapping: List[int]):
        """
        Args:
            embed_dim:     DINOv2 patch token dimension (e.g. 768).
            num_groups:    Number of anatomical groups (28 for CUB).
            num_concepts:  Total supervised concepts (312 for CUB).
            group_mapping: List of length `num_concepts` mapping each concept → group index.
        """
        super().__init__()
        self.num_groups   = num_groups
        self.num_concepts = num_concepts
        assert len(group_mapping) == num_concepts, (
            f"group_mapping length {len(group_mapping)} must equal num_concepts {num_concepts}"
        )

        # Register group_mapping as a buffer so it moves with .to(device) and is saved in state_dict
        self.register_buffer('group_mapping', torch.tensor(group_mapping, dtype=torch.long))

        # ── Step 1: 28 Group Queries (Spatial Localization) ─────────────────────
        # Learnable group queries: [1, num_groups, embed_dim]
        self.group_queries = nn.Parameter(torch.randn(1, num_groups, embed_dim))
        nn.init.trunc_normal_(self.group_queries, std=0.02)

        # Learnable temperature for cosine similarity (init=1 → no-op, learned during training)
        self.temperature = nn.Parameter(torch.ones(1))

        # Value projection for the attention output
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # ── Step 3: 312 Independent Binary Classifiers ──────────────────────────
        # Batched as a single weight matrix [num_concepts, embed_dim] + bias [num_concepts]
        # Equivalent to 312 separate Linear(embed_dim, 1) layers but computed in one bmm.
        self.concept_weight = nn.Parameter(torch.randn(num_concepts, embed_dim))
        self.concept_bias   = nn.Parameter(torch.zeros(num_concepts))
        nn.init.xavier_uniform_(self.concept_weight.unsqueeze(0))  # treat as [1, C, D]

        # Surgical prior initialization: bias = log((1-pi)/pi) for pi=0.01
        # Prevents Focal Loss / BCE logit explosion at the start of training.
        pi = 0.01
        nn.init.constant_(self.concept_bias, -math.log((1 - pi) / pi))

    def forward(self, patch_tokens: torch.Tensor):
        """
        Args:
            patch_tokens: DINOv2 patch features [B, N_patches, embed_dim]
        Returns:
            concept_logits:    [B, num_concepts]   — raw BCE logits, no activation applied
            attn_weights_2d:   [B, num_groups, H, H] — group-level spatial attention maps
            concept_features:  [B, num_concepts, embed_dim] — broadcasted concept features
        """
        B, N, D = patch_tokens.shape

        # ── Step 1: Cosine Attention over patch tokens ────────────────────────────
        # L2-normalize group queries and patch keys to remove magnitude bias
        Q = F.normalize(self.group_queries.expand(B, -1, -1), p=2, dim=-1)  # [B, 28, D]
        K = F.normalize(patch_tokens, p=2, dim=-1)                           # [B, N, D]

        # Cosine similarity scores, scaled by learnable temperature
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.temperature.clamp(min=1e-4)  # [B, 28, N]
        attn_weights = torch.softmax(attn_scores, dim=-1)                                   # [B, 28, N]

        # Aggregate values: weighted sum of projected patch tokens
        V = self.v_proj(patch_tokens)                             # [B, N, D]
        group_features = self.out_proj(torch.bmm(attn_weights, V))  # [B, 28, D]

        # Reshape attention maps to 2D grid for visualization
        H_attn = int(math.sqrt(N))
        attn_weights_2d = attn_weights.view(B, self.num_groups, H_attn, H_attn)  # [B, 28, H, H]

        # ── Step 2: Attention Broadcasting (28 groups → 312 concepts) ─────────────
        # group_mapping: [312] — index_select along the group dimension
        concept_features = group_features[:, self.group_mapping, :]  # [B, 312, D]

        # ── Step 3: Independent Binary Classification ─────────────────────────────
        # Batched dot-product: [B, 312, D] · [312, D] → [B, 312] (einsum for clarity)
        concept_logits = (
            torch.einsum('bcd,cd->bc', concept_features, self.concept_weight)
            + self.concept_bias.unsqueeze(0)
        )  # [B, 312]

        return concept_logits, attn_weights_2d, concept_features


class PatchWiseMLPConceptHead(nn.Module):
    def __init__(self, feature_dim: int, num_concepts: int, hidden_dim: int = 384):
        """Patch-wise MLP Concept Head with Global Max Pooling.
        Processes each visual patch token independently through a shared non-linear MLP,
        and applies Global Max Pooling over the spatial dimension to extract the peak
        activation and its patch location index.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_concepts = num_concepts
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_concepts)
        )
        
        # Surgical prior initialization to prevent BCE logit explosion (RetinaNet Prior pi=0.01)
        pi = 0.01
        bias_init = -math.log((1 - pi) / pi)
        nn.init.constant_(self.mlp[-1].bias, bias_init)

    def forward(self, patch_features: torch.Tensor, k: int = 3, return_weights: bool = False):
        # patch_features: [B, N_patches, feature_dim]
        # MLP maps [B, N_patches, feature_dim] -> [B, N_patches, num_concepts]
        logits_per_patch = self.mlp(patch_features)
        
        # Top-K Pooling over spatial dimension (N_patches)
        # logits_per_patch shape: [B, N_patches, num_concepts]
        topk_logits, topk_indices = torch.topk(logits_per_patch, k=k, dim=1)
        
        # Softmax-weighted pooling to prevent downward bias in logits
        weights = torch.softmax(topk_logits, dim=1) # [B, k, num_concepts]
        concept_logits = torch.sum(topk_logits * weights, dim=1) # [B, num_concepts]
        
        if return_weights:
            return concept_logits, topk_indices, weights
        return concept_logits, topk_indices


class UniversalFlexibleCBM(nn.Module):
    def __init__(
        self,
        backbone_type: str,
        backbone_name: str,
        num_supervised_concepts: int,
        num_classes: int,
        num_latent_concepts: int = 0,
        pretrained: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        concept_groups_info: Optional[List[Tuple[int, int]]] = None,
        use_cosine_attention: bool = False,
        # ── Group Broadcasting Architecture ──────────────────────────────────
        use_group_broadcasting: bool = False,
        num_groups: int = 28,
        group_mapping: Optional[List[int]] = None,  # required when use_group_broadcasting=True
        # ── DINOv2 Attention Silhouette Mask Settings ────────────────────────
        use_dino_mask: bool = False,
        dino_mask_threshold: float = 0.35,
        # ── Gated Sparse NAM Head Settings ───────────────────────────────────
        use_nam_head: bool = False,
        nam_hidden_dim: int = 64,
    ):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.backbone_name = backbone_name.lower()
        self.num_supervised_concepts = num_supervised_concepts
        self.num_latent_concepts = num_latent_concepts
        self.num_concepts = num_supervised_concepts + num_latent_concepts
        self.num_classes = num_classes
        self.lora_active = False
        self.use_group_broadcasting = use_group_broadcasting

        if use_group_broadcasting:
            if group_mapping is None:
                raise ValueError("`group_mapping` must be provided when `use_group_broadcasting=True`.")
            if len(group_mapping) != num_supervised_concepts:
                raise ValueError(
                    f"`group_mapping` length ({len(group_mapping)}) must equal "
                    f"`num_supervised_concepts` ({num_supervised_concepts})."
                )

        # 1. Initialize Backbone based on architecture
        if self.backbone_name.startswith('resnet'):
            # Load CNN backbone
            self.backbone = timm.create_model(
                backbone_name, 
                pretrained=pretrained, 
                num_classes=0, 
                global_pool=''
            )
            
            # Apply True 14x14 Spatial Surgery to preserve spatial detail (beaks, eyes, etc.)
            if hasattr(self.backbone, 'layer4'):
                try:
                    # For ResNet18/34 (BasicBlock): stride is in conv1
                    if hasattr(self.backbone.layer4[0], 'conv1') and self.backbone.layer4[0].conv1.stride == (2, 2):
                        self.backbone.layer4[0].conv1.stride = (1, 1)
                    # For ResNet50/101/152 (BottleneckBlock): stride is in conv2
                    if hasattr(self.backbone.layer4[0], 'conv2') and self.backbone.layer4[0].conv2.stride == (2, 2):
                        self.backbone.layer4[0].conv2.stride = (1, 1)
                    
                    if hasattr(self.backbone.layer4[0], 'downsample') and self.backbone.layer4[0].downsample is not None:
                        if hasattr(self.backbone.layer4[0].downsample[0], 'stride'):
                            self.backbone.layer4[0].downsample[0].stride = (1, 1)
                            
                    # Apply Dilated Convolutions (Atrous Conv) to maintain receptive field alignment
                    for block in self.backbone.layer4:
                        if hasattr(block, 'conv1') and block.conv1.kernel_size == (3, 3):
                            block.conv1.dilation = (2, 2)
                            block.conv1.padding = (2, 2)
                        if hasattr(block, 'conv2') and block.conv2.kernel_size == (3, 3):
                            block.conv2.dilation = (2, 2)
                            block.conv2.padding = (2, 2)
                            
                    print(f"{BOLD}{GREEN}[Dilated Surgery]{RESET} Modified layer4 strides to (1, 1) and conv dilations to (2, 2) to preserve receptive field alignment.")
                except Exception as e:
                    print(f"{BOLD}{YELLOW}[Warning]{RESET} Stride surgery failed on resnet backbone: {e}. Falling back to standard stride.")
            
            # Dynamic Feature Dimension Inference
            feature_dim, _, _ = self._infer_feature_dim()
            
            # Layer Construction for ResNet
            self.supervised_attention = ConceptAttentionLayer(
                feature_dim=feature_dim, 
                num_concepts=num_supervised_concepts
            )
            if self.num_latent_concepts > 0:
                self.latent_attention = ConceptAttentionLayer(
                    feature_dim=feature_dim, 
                    num_concepts=num_latent_concepts
                )
                
        elif self.backbone_name.startswith('vit') or 'dinov2' in self.backbone_name:
            # Check timm registry to provide clear diagnostic error if naming convention is unsupported
            if not timm.is_model(backbone_name):
                available_dinov2 = [m for m in timm.list_models('*dinov2*')]
                raise ValueError(
                    f"Model '{backbone_name}' not found in timm registry. "
                    f"Available DINOv2 models in local timm registry: {available_dinov2}"
                )
                
            # Load ViT / DINOv2 backbone with positional embedding interpolation enabled (dynamic_img_size=True)
            # This cleanly supports 224x224 input without hardcoded image sizing or silencing TypeErrors.
            vit_model = timm.create_model(backbone_name, pretrained=pretrained, dynamic_img_size=True)
            self.backbone = ViTBackboneWrapper(vit_model, use_dino_mask=use_dino_mask, mask_threshold=dino_mask_threshold)
            
            # Apply LoRA 어댑터 주입 if requested
            self.lora_active = use_lora
            if use_lora:
                inject_lora_to_vit(self.backbone, r=lora_r, lora_alpha=lora_alpha)
            
            # Extract embed_dim from the Vit model
            embed_dim = vit_model.embed_dim if hasattr(vit_model, 'embed_dim') else 768
            print(f"{BOLD}{BLUE}[Backbone Factory]{RESET} Configured Cross-Attention CBM for {backbone_name} (embed_dim: {embed_dim}, use_lora: {use_lora})")

            # Create PatchWiseMLPConceptHead as the new concept head to prevent attention collapse
            print(f"{BOLD}{BLUE}[Concept Head]{RESET} PatchWiseMLPConceptHead ({embed_dim} -> 384 -> {num_supervised_concepts})")
            self.supervised_attention = PatchWiseMLPConceptHead(
                feature_dim=embed_dim,
                num_concepts=num_supervised_concepts,
                hidden_dim=384
            )
            if self.num_latent_concepts > 0:
                self.latent_attention = PatchWiseMLPConceptHead(
                    feature_dim=embed_dim,
                    num_concepts=self.num_latent_concepts,
                    hidden_dim=384
                )
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}. ResNet and ViT backbones are supported.")

        # Common CBM classification/activation layers
        if use_group_broadcasting:
            # GroupToConceptAttention outputs raw BCE logits — no activation needed at bottleneck.
            # Sigmoid is applied implicitly inside BCEWithLogitsLoss during training.
            # For inference / visualization we use sigmoid explicitly.
            self.concept_activation = nn.Sigmoid()
            print(f"{BOLD}{BLUE}[Backbone Factory]{RESET} Group Broadcasting mode — BCEWithLogitsLoss compatible (Sigmoid activation for inference).")
        elif concept_groups_info is not None:
            self.concept_activation = GroupSoftmaxActivation(concept_groups_info)
            print(f"{BOLD}{BLUE}[Backbone Factory]{RESET} Activated Mutual Exclusive Group Softmax over {len(concept_groups_info)} groups.")
        else:
            self.concept_activation = nn.Sigmoid()
            print(f"{BOLD}{BLUE}[Backbone Factory]{RESET} Activated Standard Flat Sigmoid Activation.")
            
        self.dropout = nn.Dropout(p=0.2)
        if use_nam_head:
            print(f"{BOLD}{BLUE}[Classifier Head]{RESET} GatedSparseNAMHead (concepts={num_supervised_concepts} -> hidden={nam_hidden_dim} -> classes={num_classes})")
            self.classifier_head = GatedSparseNAMHead(
                num_concepts=num_supervised_concepts,
                num_classes=num_classes,
                hidden_dim=nam_hidden_dim,
                num_latent_concepts=num_latent_concepts
            )
        else:
            self.classifier_head = nn.Linear(self.num_concepts, num_classes)
        
        # Register a buffer to store dynamically-found optimal validation logit thresholds
        self.register_buffer('concept_thresholds', torch.zeros(self.num_supervised_concepts))

    def load_state_dict(self, state_dict, strict=True):
        # If 'concept_thresholds' is not in the loaded state_dict, inject it to prevent strict loading errors
        if 'concept_thresholds' not in state_dict:
            state_dict['concept_thresholds'] = torch.zeros(self.num_supervised_concepts)
        return super().load_state_dict(state_dict, strict=strict)

    def _infer_feature_dim(self) -> tuple[int, int, int]:
        """Pass a dummy tensor through the backbone to dynamically infer feature dimensions (ResNet only)."""
        dummy_tensor = torch.randn(1, 3, 224, 224)
        
        was_training = self.backbone.training
        self.backbone.eval()
        
        with torch.no_grad():
            features = self.backbone(dummy_tensor)
            if isinstance(features, tuple):
                features = features[0]
            
            if len(features.shape) != 4:
                raise ValueError(
                    f"Expected 4D output from CNN backbone [B, C, H, W], but got shape {features.shape}. "
                    "Ensure the backbone does not apply global average pooling."
                )
                
            _, C, H, W = features.shape
            
        if was_training:
            self.backbone.train()
            
        return C, H, W

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # Input tensor shape x: [B, 3, H, W]
        features = self.backbone(x)  # [B, C, H_attn, W_attn] (ResNet) or [B, 196, embed_dim] (ViT)
        if isinstance(features, tuple):
            features = features[0]
            
        if self.backbone_name.startswith('resnet'):
            # ResNet still uses spatial attention conv
            supervised_logits, supervised_attn, supervised_features = self.supervised_attention(features)
            if self.num_latent_concepts > 0:
                latent_logits, latent_attn, latent_features = self.latent_attention(features)
                concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
                attn_weights = torch.cat([supervised_attn, latent_attn], dim=1)
            else:
                concept_logits = supervised_logits
                attn_weights = supervised_attn
                latent_features = None
        else:
            # ViT / DINOv2 uses PatchWiseMLPConceptHead
            k_val = 3
            supervised_logits, supervised_topk_indices, supervised_weights = self.supervised_attention(features, k=k_val, return_weights=True)
            
            # Convert supervised_topk_indices [B, k, num_supervised_concepts] to sparse/smoothed dynamically-sized maps
            B = supervised_topk_indices.size(0)
            N_patches = features.size(1) # e.g. 256 or 196
            H_attn = int(math.sqrt(N_patches))
            device = supervised_topk_indices.device
            D = features.size(-1)
            
            # Reshape topk indices from [B, k, num_supervised_concepts] to [B, num_supervised_concepts, k] for scatter
            indices_transposed = supervised_topk_indices.permute(0, 2, 1) # [B, num_supervised_concepts, k]
            weights_transposed = supervised_weights.permute(0, 2, 1) # [B, num_supervised_concepts, k]
            
            sparse_maps = torch.zeros(B, self.num_supervised_concepts, N_patches, device=device)
            # Scatter dynamic softmax weights to each top-k patch position
            sparse_maps.scatter_(2, indices_transposed, weights_transposed)
            sparse_maps = sparse_maps.view(B, self.num_supervised_concepts, H_attn, H_attn)
            
            from torchvision.transforms.functional import gaussian_blur
            supervised_attn = gaussian_blur(sparse_maps, kernel_size=[3, 3], sigma=[1.0, 1.0])
            
            # Memory-Efficient Top-k Gathering along dim=1 (N_patches) using flattened indices [B, C * k]
            flat_indices = indices_transposed.reshape(B, self.num_supervised_concepts * k_val)
            gathered_flat = torch.gather(
                features,
                dim=1,
                index=flat_indices.unsqueeze(-1).expand(-1, -1, D)
            ) # [B, C * k, D]
            
            # Reshape gathered features back to [B, num_supervised_concepts, k, D]
            gathered_features = gathered_flat.view(B, self.num_supervised_concepts, k_val, D)
            
            # Softmax-Weighted average over the top-k dimension
            supervised_features = torch.sum(gathered_features * weights_transposed.unsqueeze(-1), dim=2) # [B, num_supervised_concepts, D]
            
            if self.num_latent_concepts > 0:
                latent_logits, latent_topk_indices, latent_weights = self.latent_attention(features, k=k_val, return_weights=True)
                
                latent_indices_transposed = latent_topk_indices.permute(0, 2, 1)
                latent_weights_transposed = latent_weights.permute(0, 2, 1)
                
                sparse_latent_maps = torch.zeros(B, self.num_latent_concepts, N_patches, device=device)
                sparse_latent_maps.scatter_(2, latent_indices_transposed, latent_weights_transposed)
                sparse_latent_maps = sparse_latent_maps.view(B, self.num_latent_concepts, H_attn, H_attn)
                latent_attn = gaussian_blur(sparse_latent_maps, kernel_size=[3, 3], sigma=[1.0, 1.0])
                
                latent_flat_indices = latent_indices_transposed.reshape(B, self.num_latent_concepts * k_val)
                latent_gathered_flat = torch.gather(
                    features,
                    dim=1,
                    index=latent_flat_indices.unsqueeze(-1).expand(-1, -1, D)
                )
                latent_gathered_features = latent_gathered_flat.view(B, self.num_latent_concepts, k_val, D)
                latent_features = torch.sum(latent_gathered_features * latent_weights_transposed.unsqueeze(-1), dim=2)
                
                concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
                attn_weights = torch.cat([supervised_attn, latent_attn], dim=1)
            else:
                concept_logits = supervised_logits
                attn_weights = supervised_attn
                latent_features = None
                
        # Apply dropout to regularize concept predictions (in logit space)
        concept_logits_dropout = self.dropout(concept_logits)
        
        # Final classification target output logits (Inverse Sigmoid / Logit Intervention SOTA design)
        class_logits = self.classifier_head(concept_logits_dropout)  # [B, num_classes]
        
        if return_features:
            return class_logits, concept_logits, attn_weights, supervised_features, latent_features
        return class_logits, concept_logits, attn_weights

    def freeze_backbone(self):
        """Freezes the vision backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes the vision backbone parameters.
        If LoRA is active, unfreezes ONLY the LoRA adapter parameters to preserve ImageNet weights.
        """
        if getattr(self, 'lora_active', False):
            for name, param in self.backbone.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f"{BOLD}{GREEN}[LoRA Unfreeze]{RESET} Activated only LoRA adapter parameters for training.")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print(f"{BOLD}{GREEN}[Full Unfreeze]{RESET} Activated all backbone parameters for training.")

    def freeze_classifier(self):
        """Freezes the classifier head parameters."""
        for param in self.classifier_head.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        """Unfreezes the classifier head parameters."""
        for param in self.classifier_head.parameters():
            param.requires_grad = True

    def freeze_supervised_attention(self):
        """Freezes the supervised concept attention parameters."""
        for param in self.supervised_attention.parameters():
            param.requires_grad = False

    def unfreeze_supervised_attention(self):
        """Unfreezes the supervised concept attention parameters."""
        for param in self.supervised_attention.parameters():
            param.requires_grad = True

    def freeze_latent_attention(self):
        """Freezes the latent concept attention parameters."""
        if self.num_latent_concepts > 0:
            for param in self.latent_attention.parameters():
                param.requires_grad = False

    def unfreeze_latent_attention(self):
        """Unfreezes the latent concept attention parameters."""
        if self.num_latent_concepts > 0:
            for param in self.latent_attention.parameters():
                param.requires_grad = True
