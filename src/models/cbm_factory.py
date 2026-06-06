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
                 hidden_dim: int = 64, num_latent_concepts: int = 0,
                 use_pairwise_nam: bool = False, max_pairs: int = 128,
                 input_dropout_p: float = 0.2):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_latent_concepts = num_latent_concepts
        self.use_pairwise_nam = use_pairwise_nam
        self.max_pairs = max_pairs
        
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
        
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.dropout = nn.Dropout(p=0.2)
        
        # Learnable gating parameter initialized to 1.0
        self.concept_gates = nn.Parameter(torch.ones(num_concepts))
        
        if self.num_latent_concepts > 0:
            self.latent_linear = nn.Linear(num_latent_concepts, num_classes)
            self.latent_gates = nn.Parameter(torch.ones(num_latent_concepts))
        else:
            self.latent_linear = None
            self.latent_gates = None

        # Weight Initialization
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
            
        nn.init.xavier_uniform_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
            
        if self.num_latent_concepts > 0 and self.latent_linear is not None:
            nn.init.xavier_uniform_(self.latent_linear.weight)
            nn.init.zeros_(self.latent_linear.bias)

        # 2. Pairwise 2D MLPs
        if use_pairwise_nam:
            self.num_pairs = num_concepts * (num_concepts - 1) // 2
            self.M = min(self.num_pairs, max_pairs)
            
            pair_indices = []
            for i in range(num_concepts):
                for j in range(i + 1, num_concepts):
                    pair_indices.append((i, j))
            self.pair_indices = pair_indices
            
            self.pairwise_gates = nn.Parameter(torch.zeros(self.num_pairs))
            nn.init.normal_(self.pairwise_gates, std=0.01)
            
            self.pairwise_conv1 = nn.Conv1d(
                in_channels=self.M * 2,
                out_channels=self.M * hidden_dim,
                kernel_size=1,
                groups=self.M
            )
            self.pairwise_conv2 = nn.Conv1d(
                in_channels=self.M * hidden_dim,
                out_channels=self.M * num_classes,
                kernel_size=1,
                groups=self.M
            )
            
            nn.init.kaiming_uniform_(self.pairwise_conv1.weight, a=math.sqrt(5))
            if self.pairwise_conv1.bias is not None:
                nn.init.zeros_(self.pairwise_conv1.bias)
            nn.init.xavier_uniform_(self.pairwise_conv2.weight)
            if self.pairwise_conv2.bias is not None:
                nn.init.zeros_(self.pairwise_conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Apply explicit Input Dropout specifically to concatenated features before entering sub-networks
        x_dropped = self.input_dropout(x)
        supervised_x = x_dropped[:, :self.num_concepts].unsqueeze(-1)
        
        h = F.relu(self.conv1(supervised_x))
        h = self.dropout(h)
        y = self.conv2(h) # Shape: [Batch, num_concepts * num_classes, 1]
        
        y = y.view(batch_size, self.num_concepts, self.num_classes)
        gated_y = y * self.concept_gates.view(1, self.num_concepts, 1)
        supervised_out = gated_y.sum(dim=1)
        
        if self.use_pairwise_nam:
            # Select top M active pairs dynamically to prevent parameter explosion
            topk_vals, topk_indices = torch.topk(torch.abs(self.pairwise_gates), k=self.M)
            selected_gates = self.pairwise_gates[topk_indices]
            
            pair_inputs = []
            for idx in topk_indices.tolist():
                i, j = self.pair_indices[idx]
                c_i = x_dropped[:, i].unsqueeze(-1)
                c_j = x_dropped[:, j].unsqueeze(-1)
                pair_inputs.append(torch.cat([c_i, c_j], dim=-1))
                
            pair_features = torch.cat(pair_inputs, dim=1).unsqueeze(-1) # [B, M * 2, 1]
            
            h_p = F.relu(self.pairwise_conv1(pair_features))
            h_p = self.dropout(h_p)
            y_p = self.pairwise_conv2(h_p)
            y_p = y_p.view(batch_size, self.M, self.num_classes)
            
            gated_yp = y_p * selected_gates.view(1, self.M, 1)
            pairwise_out = gated_yp.sum(dim=1)
            supervised_out = supervised_out + pairwise_out
            
        if self.num_latent_concepts > 0 and self.latent_linear is not None:
            latent_x = x_dropped[:, self.num_concepts:]
            if self.latent_gates is not None:
                latent_x = latent_x * self.latent_gates.view(1, -1)
            latent_out = self.latent_linear(latent_x)
            return supervised_out + latent_out
            
        return supervised_out

    def get_sparsity_loss(self, latent_penalty_scale: float = 1.0) -> torch.Tensor:
        loss = torch.sum(torch.abs(self.concept_gates))
        if self.num_latent_concepts > 0 and self.latent_gates is not None:
            loss = loss + latent_penalty_scale * torch.sum(torch.abs(self.latent_gates))
        if self.use_pairwise_nam:
            loss = loss + torch.sum(torch.abs(self.pairwise_gates))
        return loss

class ConceptAttentionLayer(nn.Module):
    def __init__(self, feature_dim: int, num_concepts: int, num_heads: int = 4, probabilistic: bool = False):
        super().__init__()
        self.num_concepts = num_concepts
        self.feature_dim = feature_dim
        self.probabilistic = probabilistic
        
        self.attention_conv = nn.Conv2d(feature_dim, num_concepts, kernel_size=1)
        self.concept_proj = nn.Parameter(torch.randn(num_concepts, feature_dim))
        self.concept_bias = nn.Parameter(torch.zeros(num_concepts))
        self.temperature = nn.Parameter(torch.ones(1))

        nn.init.xavier_uniform_(self.attention_conv.weight)
        if self.attention_conv.bias is not None:
            nn.init.zeros_(self.attention_conv.bias)
        nn.init.xavier_uniform_(self.concept_proj)

        if probabilistic:
            self.concept_proj_logvar = nn.Parameter(torch.randn(num_concepts, feature_dim))
            self.concept_bias_logvar = nn.Parameter(torch.zeros(num_concepts))
            nn.init.xavier_uniform_(self.concept_proj_logvar)
            nn.init.constant_(self.concept_bias_logvar, -2.0)

    def forward(self, features: torch.Tensor):
        B, C, H, W = features.shape
        
        attn_logits = self.attention_conv(features)
        features_flat = features.flatten(2).transpose(1, 2)
        features_norm = F.normalize(features_flat, p=2, dim=-1)
        attn_queries = self.attention_conv.weight.squeeze(-1).squeeze(-1)
        attn_queries_norm = F.normalize(attn_queries, p=2, dim=-1)
        cosine_logits = torch.bmm(
            attn_queries_norm.unsqueeze(0).expand(B, -1, -1),
            features_norm.transpose(1, 2)
        ) / self.temperature.clamp(min=1e-4)
        
        attn_weights = torch.softmax(cosine_logits, dim=-1)
        attn_weights_2d = attn_weights.view(B, self.num_concepts, H, W)
        
        weighted_features = torch.bmm(attn_weights, features_norm)
        
        concept_proj_norm = F.normalize(self.concept_proj, p=2, dim=-1)
        concept_mean = (
            (weighted_features * concept_proj_norm.unsqueeze(0)).sum(dim=-1)
            / self.temperature.clamp(min=1e-4)
            + self.concept_bias.unsqueeze(0)
        )
        if self.probabilistic:
            concept_proj_logvar_norm = F.normalize(self.concept_proj_logvar, p=2, dim=-1)
            concept_logvar = (
                (weighted_features * concept_proj_logvar_norm.unsqueeze(0)).sum(dim=-1)
                / self.temperature.clamp(min=1e-4)
                + self.concept_bias_logvar.unsqueeze(0)
            )
            concept_logvar = torch.clamp(concept_logvar, min=-10.0, max=10.0)
            return concept_mean, concept_logvar, attn_weights_2d, weighted_features
            
        return concept_mean, attn_weights_2d, weighted_features


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
    def __init__(self, vit_model):
        """Wrapper for Vision Transformer to dynamically extract patch tokens while ignoring the CLS token.
        Ensures compatibility with internal feature representations and main training pipelines.
        """
        super().__init__()
        self.vit = vit_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ConvNeXtBackboneWrapper(nn.Module):
    def __init__(self, convnext_model):
        """Wrapper for ConvNeXt to extract 2D features and rearrange them as a token sequence [B, H*W, C].
        This makes ConvNeXt compatible with ViT attention heads.
        """
        super().__init__()
        self.convnext = convnext_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features without global pooling
        feats = self.convnext(x)
            
        if isinstance(feats, tuple):
            feats = feats[0]
            
        assert isinstance(feats, torch.Tensor), f"Expected backbone features to be a torch.Tensor, but got {type(feats)}"
        assert len(feats.shape) == 4, (
            f"Expected 4D feature map tensor [B, C, H, W] from ConvNeXt backbone, but got shape {feats.shape}. "
            "Ensure global pooling is disabled in timm and your model returns spatial features."
        )
        
        # Rearrange shape: [B, C, H, W] -> [B, H*W, C]
        # Equivalent to einops.rearrange(feats, 'b c h w -> b (h w) c')
        B, C, H, W = feats.shape
        feats = feats.flatten(2).transpose(1, 2)
        return feats



class ViTCrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_concepts: int, num_heads: int = 4,
                 use_cosine_attention: bool = False, probabilistic: bool = False):
        super().__init__()
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim
        self.use_cosine_attention = use_cosine_attention
        self.probabilistic = probabilistic
        
        # Learnable concept queries: [1, num_concepts, embed_dim]
        self.concept_queries = nn.Parameter(torch.randn(1, num_concepts, embed_dim))
        
        # Concept-specific projections & biases: [num_concepts, embed_dim]
        self.concept_proj = nn.Parameter(torch.randn(num_concepts, embed_dim))
        self.concept_bias = nn.Parameter(torch.zeros(num_concepts))
        
        # Surgical initialization of bias to prevent Focal Loss logit explosion (RetinaNet Prior pi=0.01)
        pi = 0.01
        bias_init = -math.log((1 - pi) / pi)
        nn.init.constant_(self.concept_bias, bias_init)

        # Weight Initialization
        nn.init.trunc_normal_(self.concept_queries, std=0.02)
        nn.init.xavier_uniform_(self.concept_proj)

        if probabilistic:
            self.concept_proj_logvar = nn.Parameter(torch.randn(num_concepts, embed_dim))
            self.concept_bias_logvar = nn.Parameter(torch.zeros(num_concepts))
            nn.init.xavier_uniform_(self.concept_proj_logvar)
            nn.init.constant_(self.concept_bias_logvar, -2.0)

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
            concept_mean = (
                (attn_out_norm * concept_proj_norm.unsqueeze(0)).sum(dim=-1)
                / self.temperature.clamp(min=1e-4)
                + self.concept_bias.unsqueeze(0)
            )
            if self.probabilistic:
                concept_proj_logvar_norm = F.normalize(self.concept_proj_logvar, p=2, dim=-1)
                concept_logvar = (
                    (attn_out_norm * concept_proj_logvar_norm.unsqueeze(0)).sum(dim=-1)
                    / self.temperature.clamp(min=1e-4)
                    + self.concept_bias_logvar.unsqueeze(0)
                )
        else:
            # --- Standard MultiheadAttention path ---
            attn_out, attn_weights = self.cross_attention(
                query=queries,
                key=features,
                value=features
            )  # attn_out: [B, num_concepts, D], attn_weights: [B, num_concepts, N]

            concept_mean = (
                (attn_out * self.concept_proj.unsqueeze(0)).sum(dim=-1)
                + self.concept_bias.unsqueeze(0)
            )
            if self.probabilistic:
                concept_logvar = (
                    (attn_out * self.concept_proj_logvar.unsqueeze(0)).sum(dim=-1)
                    + self.concept_bias_logvar.unsqueeze(0)
                )

        # Reshape attention weights: [B, num_concepts, N] -> [B, num_concepts, sqrt(N), sqrt(N)]
        H_attn = int(math.sqrt(N))
        if H_attn * H_attn == N:
            attn_weights_2d = attn_weights.view(B, self.num_concepts, H_attn, H_attn)
        else:
            N_single = N // 2
            H_attn_single = int(math.sqrt(N_single))
            attn_clinic = attn_weights[:, :, :N_single].view(B, self.num_concepts, H_attn_single, H_attn_single)
            attn_derm = attn_weights[:, :, N_single:].view(B, self.num_concepts, H_attn_single, H_attn_single)
            attn_weights_2d = torch.cat([attn_clinic, attn_derm], dim=3)

        if self.probabilistic:
            concept_logvar = torch.clamp(concept_logvar, min=-10.0, max=10.0)
            return concept_mean, concept_logvar, attn_weights_2d, attn_out

        return concept_mean, attn_weights_2d, attn_out


class GroupToConceptAttention(nn.Module):
    def __init__(self, embed_dim: int, num_groups: int, num_concepts: int,
                 group_mapping: List[int], probabilistic: bool = False):
        super().__init__()
        self.num_groups   = num_groups
        self.num_concepts = num_concepts
        self.probabilistic = probabilistic
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
        self.concept_weight = nn.Parameter(torch.randn(num_concepts, embed_dim))
        self.concept_bias   = nn.Parameter(torch.zeros(num_concepts))
        nn.init.xavier_uniform_(self.concept_weight.unsqueeze(0))  # treat as [1, C, D]

        # Surgical prior initialization: bias = log((1-pi)/pi) for pi=0.01
        pi = 0.01
        nn.init.constant_(self.concept_bias, -math.log((1 - pi) / pi))

        if probabilistic:
            self.concept_weight_logvar = nn.Parameter(torch.randn(num_concepts, embed_dim))
            self.concept_bias_logvar = nn.Parameter(torch.zeros(num_concepts))
            nn.init.xavier_uniform_(self.concept_weight_logvar.unsqueeze(0))
            nn.init.constant_(self.concept_bias_logvar, -2.0)

    def forward(self, patch_tokens: torch.Tensor):
        B, N, D = patch_tokens.shape

        # ── Step 1: Cosine Attention over patch tokens ────────────────────────────
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
        if H_attn * H_attn == N:
            attn_weights_2d = attn_weights.view(B, self.num_groups, H_attn, H_attn)  # [B, 28, H, H]
        else:
            N_single = N // 2
            H_attn_single = int(math.sqrt(N_single))
            attn_clinic = attn_weights[:, :, :N_single].view(B, self.num_groups, H_attn_single, H_attn_single)
            attn_derm = attn_weights[:, :, N_single:].view(B, self.num_groups, H_attn_single, H_attn_single)
            attn_weights_2d = torch.cat([attn_clinic, attn_derm], dim=3)

        # ── Step 2: Attention Broadcasting (28 groups → 312 concepts) ─────────────
        concept_features = group_features[:, self.group_mapping, :]  # [B, 312, D]

        # ── Step 3: Independent Binary Classification ─────────────────────────────
        concept_mean = (
            torch.einsum('bcd,cd->bc', concept_features, self.concept_weight)
            + self.concept_bias.unsqueeze(0)
        )  # [B, 312]

        if self.probabilistic:
            concept_logvar = (
                torch.einsum('bcd,cd->bc', concept_features, self.concept_weight_logvar)
                + self.concept_bias_logvar.unsqueeze(0)
            )
            concept_logvar = torch.clamp(concept_logvar, min=-10.0, max=10.0)
            return concept_mean, concept_logvar, attn_weights_2d, concept_features

        return concept_mean, attn_weights_2d, concept_features


class PatchWiseMLPConceptHead(nn.Module):
    def __init__(self, feature_dim: int, num_concepts: int, hidden_dim: int = 384, probabilistic: bool = False):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_concepts = num_concepts
        self.probabilistic = probabilistic
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_concepts)
        )
        
        # Weight Initialization
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Surgical prior initialization to prevent BCE logit explosion (RetinaNet Prior pi=0.01)
        pi = 0.01
        bias_init = -math.log((1 - pi) / pi)
        nn.init.constant_(self.mlp[-1].bias, bias_init)

        if probabilistic:
            self.mlp_logvar = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, num_concepts)
            )
            for m in self.mlp_logvar:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.constant_(self.mlp_logvar[-1].bias, -2.0)

    def forward(self, patch_features: torch.Tensor, k: int = 3, return_weights: bool = False):
        # patch_features: [B, N_patches, feature_dim]
        logits_per_patch = self.mlp(patch_features)
        
        # Top-K Pooling over spatial dimension (N_patches)
        topk_logits, topk_indices = torch.topk(logits_per_patch, k=k, dim=1)
        
        # Softmax-weighted pooling to prevent downward bias in logits
        weights = torch.softmax(topk_logits, dim=1) # [B, k, num_concepts]
        concept_mean = torch.sum(topk_logits * weights, dim=1) # [B, num_concepts]
        
        if self.probabilistic:
            logvar_per_patch = self.mlp_logvar(patch_features)
            # Gather logvar values at the top-k indices of the mean path
            logvar_gathered = torch.gather(logvar_per_patch, dim=1, index=topk_indices) # [B, k, num_concepts]
            concept_logvar = torch.sum(logvar_gathered * weights, dim=1)
            concept_logvar = torch.clamp(concept_logvar, min=-10.0, max=10.0)
            
            if return_weights:
                return concept_mean, concept_logvar, topk_indices, weights
            return concept_mean, concept_logvar, topk_indices
            
        if return_weights:
            return concept_mean, topk_indices, weights
        return concept_mean, topk_indices


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
        # ── Gated Sparse NAM Head Settings ───────────────────────────────────
        use_nam_head: bool = False,
        nam_hidden_dim: int = 64,
        # ── Advanced Integration Settings ────────────────────────────────────
        use_probabilistic_cbm: bool = False,
        use_concept_attention: bool = False,
        use_pairwise_nam: bool = False,
        use_multimodal: bool = False,
        # ── Age/Sex Skip Connection ───────────────────────────────────────────
        age_sex_skip_connection: bool = False,
        num_tabular_features: int = 3,
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
        self.use_probabilistic_cbm = use_probabilistic_cbm
        self.use_concept_attention = use_concept_attention
        self.use_pairwise_nam = use_pairwise_nam
        self.use_multimodal = use_multimodal
        self.age_sex_skip_connection = age_sex_skip_connection
        self.num_tabular_features = num_tabular_features
        
        # Placeholders for VAE loss calculation
        self.last_mean = None
        self.last_logvar = None

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
            
            # Apply True 14x14 Spatial Surgery to preserve spatial detail (beaks, eyes, etc.) - ResNet only
            if self.backbone_name.startswith('resnet') and hasattr(self.backbone, 'layer4'):
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
            print(f"{BOLD}{BLUE}[Concept Head]{RESET} ConceptAttentionLayer ({feature_dim} -> {num_supervised_concepts}) | Mode: {'Probabilistic (VAE-style)' if use_probabilistic_cbm else 'Deterministic'}")
            self.supervised_attention = ConceptAttentionLayer(
                feature_dim=feature_dim, 
                num_concepts=num_supervised_concepts,
                probabilistic=use_probabilistic_cbm
            )
            if self.num_latent_concepts > 0:
                print(f"{BOLD}{BLUE}[Latent Concept Head]{RESET} ConceptAttentionLayer ({feature_dim} -> {num_latent_concepts}) | Mode: {'Probabilistic (VAE-style)' if use_probabilistic_cbm else 'Deterministic'}")
                self.latent_attention = ConceptAttentionLayer(
                    feature_dim=feature_dim, 
                    num_concepts=num_latent_concepts,
                    probabilistic=use_probabilistic_cbm
                )
                
        elif self.backbone_name.startswith('vit') or 'dinov2' in self.backbone_name or 'convnext' in self.backbone_name:
            if 'convnext' in self.backbone_name:
                raw_backbone = timm.create_model(
                    backbone_name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool=''
                )
                self.backbone = raw_backbone
                embed_dim, _, _ = self._infer_feature_dim()
                self.backbone = ConvNeXtBackboneWrapper(raw_backbone)
                print(f"{BOLD}{BLUE}[Backbone Factory]{RESET} Configured Tokenized ConvNeXt for {backbone_name} (embed_dim: {embed_dim})")
            else:
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
                self.backbone = ViTBackboneWrapper(vit_model)
                
                # Apply LoRA 어댑터 주입 if requested
                self.lora_active = use_lora
                if use_lora:
                    inject_lora_to_vit(self.backbone, r=lora_r, lora_alpha=lora_alpha)
                
                # Extract embed_dim from the Vit model
                embed_dim = vit_model.embed_dim if hasattr(vit_model, 'embed_dim') else 768
                print(f"{BOLD}{BLUE}[Backbone Factory]{RESET} Configured Cross-Attention CBM for {backbone_name} (embed_dim: {embed_dim}, use_lora: {use_lora})")

            if use_group_broadcasting:
                print(f"{BOLD}{BLUE}[Concept Head]{RESET} GroupToConceptAttention ({embed_dim} -> groups={num_groups} -> concepts={num_supervised_concepts}) | Mode: {'Probabilistic (VAE-style)' if use_probabilistic_cbm else 'Deterministic'}")
                self.supervised_attention = GroupToConceptAttention(
                    embed_dim=embed_dim,
                    num_groups=num_groups,
                    num_concepts=num_supervised_concepts,
                    group_mapping=group_mapping,
                    probabilistic=use_probabilistic_cbm
                )
            elif use_concept_attention:
                print(f"{BOLD}{BLUE}[Concept Head]{RESET} ViTCrossAttentionLayer ({embed_dim} -> {num_supervised_concepts}) | Mode: {'Probabilistic (VAE-style)' if use_probabilistic_cbm else 'Deterministic'}")
                self.supervised_attention = ViTCrossAttentionLayer(
                    embed_dim=embed_dim,
                    num_concepts=num_supervised_concepts,
                    probabilistic=use_probabilistic_cbm
                )
            else:
                # Create PatchWiseMLPConceptHead as the new concept head to prevent attention collapse
                print(f"{BOLD}{BLUE}[Concept Head]{RESET} PatchWiseMLPConceptHead ({embed_dim} -> 384 -> {num_supervised_concepts}) | Mode: {'Probabilistic (VAE-style)' if use_probabilistic_cbm else 'Deterministic'}")
                self.supervised_attention = PatchWiseMLPConceptHead(
                    feature_dim=embed_dim,
                    num_concepts=num_supervised_concepts,
                    hidden_dim=384,
                    probabilistic=use_probabilistic_cbm
                )
            if self.num_latent_concepts > 0:
                if use_group_broadcasting:
                    raise NotImplementedError("Latent concepts are not supported with group broadcasting.")
                print(f"{BOLD}{BLUE}[Latent Concept Head]{RESET} PatchWiseMLPConceptHead ({embed_dim} -> 384 -> {self.num_latent_concepts}) | Mode: {'Probabilistic (VAE-style)' if use_probabilistic_cbm else 'Deterministic'}")
                self.latent_attention = PatchWiseMLPConceptHead(
                    feature_dim=embed_dim,
                    num_concepts=self.num_latent_concepts,
                    hidden_dim=384,
                    probabilistic=use_probabilistic_cbm
                )
        elif self.backbone_type == 'torchxrayvision':
            from src.models.xray_backbone import XRayDenseNetBackboneWrapper, XRayConceptHead
            # Default to densenet121 wrapper (we can add more xrv models here in the future)
            self.backbone = XRayDenseNetBackboneWrapper(pretrained=pretrained, return_spatial=use_concept_attention)
            embed_dim = 1024
            if use_concept_attention:
                print(f"{BOLD}{BLUE}[Concept Head]{RESET} ConceptAttentionLayer ({embed_dim} -> {num_supervised_concepts}) | Mode: Deterministic")
                self.supervised_attention = ConceptAttentionLayer(
                    feature_dim=embed_dim,
                    num_concepts=num_supervised_concepts,
                    probabilistic=False
                )
            else:
                self.supervised_attention = XRayConceptHead(embed_dim=embed_dim, num_concepts=num_supervised_concepts)
            print(f"{BOLD}{BLUE}[Backbone Factory]{RESET} Configured TorchXRayVision {backbone_name} (embed_dim: {embed_dim})")
            if self.num_latent_concepts > 0:
                raise NotImplementedError("Latent concepts are not supported with torchxrayvision backbone yet.")
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type} / backbone_name: {backbone_name}. ResNet, ConvNeXt, ViT, and TorchXRayVision backbones are supported.")

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
        
        # age_sex_skip_connection=True: tabular features are added AFTER the classifier head
        # (skip over NAM). In this case, the classifier head never sees tabular features.
        # use_multimodal and age_sex_skip_connection are mutually exclusive.
        if age_sex_skip_connection and use_multimodal:
            raise ValueError("`age_sex_skip_connection` and `use_multimodal` are mutually exclusive. "
                             "Set only one of them to True.")
        
        if use_nam_head:
            num_nam_inputs = num_supervised_concepts + (3 if use_multimodal else 0)
            print(f"{BOLD}{BLUE}[Classifier Head]{RESET} GatedSparseNAMHead (concepts={num_nam_inputs} -> hidden={nam_hidden_dim} -> classes={num_classes}, use_pairwise_nam={use_pairwise_nam})")
            self.classifier_head = GatedSparseNAMHead(
                num_concepts=num_nam_inputs,
                num_classes=num_classes,
                hidden_dim=nam_hidden_dim,
                num_latent_concepts=num_latent_concepts,
                use_pairwise_nam=use_pairwise_nam
            )
        else:
            num_linear_inputs = self.num_concepts + (3 if use_multimodal else 0)
            self.classifier_head = nn.Linear(num_linear_inputs, num_classes)

        # Independent tabular skip-connection head (Age/Sex -> num_classes)
        # Initialized to near-zero so it starts as identity (no bias for the x-ray channel)
        if age_sex_skip_connection:
            self.tabular_skip_head = nn.Linear(num_tabular_features, num_classes)
            nn.init.normal_(self.tabular_skip_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.tabular_skip_head.bias)
            print(f"{BOLD}{BLUE}[Skip Connection]{RESET} Age/Sex Skip Connection enabled: "
                  f"tabular_skip_head({num_tabular_features} -> {num_classes}) added directly to final logits (bypasses NAM).")
        else:
            self.tabular_skip_head = None
        
        # Register a buffer to store dynamically-found optimal validation logit thresholds
        self.register_buffer('concept_thresholds', torch.zeros(self.num_supervised_concepts))

    def load_state_dict(self, state_dict, strict=True):
        # If 'concept_thresholds' is not in the loaded state_dict, inject it to prevent strict loading errors
        if 'concept_thresholds' not in state_dict:
            state_dict['concept_thresholds'] = torch.zeros(self.num_supervised_concepts)
            
        # Backward compatibility for concept_gates and latent_gates
        if hasattr(self.classifier_head, 'concept_gates'):
            key = 'classifier_head.concept_gates'
            if key not in state_dict:
                state_dict[key] = torch.ones(self.classifier_head.num_concepts)
        if hasattr(self.classifier_head, 'latent_gates') and self.classifier_head.latent_gates is not None:
            key = 'classifier_head.latent_gates'
            if key not in state_dict:
                state_dict[key] = torch.ones(self.classifier_head.num_latent_concepts)
                
        ret = super().load_state_dict(state_dict, strict=strict)
        
        # Post-load Gated NAM concept gates pruning (threshold = 0.05)
        if hasattr(self.classifier_head, 'concept_gates'):
            with torch.no_grad():
                gates = self.classifier_head.concept_gates
                total_gates = gates.numel()
                original_active = (gates.abs() > 0.0).sum().item()
                
                # Zero out gates <= 0.05
                pruned_mask = gates.abs() <= 0.05
                gates[pruned_mask] = 0.0
                
                final_active = (gates.abs() > 0.0).sum().item()
                print(f"[Gated NAM Pruning] Pruned gates with absolute value <= 0.05")
                print(f"  Original active gates (>0.0): {original_active} / {total_gates}")
                print(f"  Final remaining gates (>0.05): {final_active} / {total_gates}")
                
        # Post-load Gated NAM latent gates pruning (threshold = 0.05)
        if hasattr(self.classifier_head, 'latent_gates') and self.classifier_head.latent_gates is not None:
            with torch.no_grad():
                l_gates = self.classifier_head.latent_gates
                l_total_gates = l_gates.numel()
                l_original_active = (l_gates.abs() > 0.0).sum().item()
                
                # Zero out gates <= 0.05
                l_pruned_mask = l_gates.abs() <= 0.05
                l_gates[l_pruned_mask] = 0.0
                
                l_final_active = (l_gates.abs() > 0.0).sum().item()
                print(f"[Gated NAM Latent Pruning] Pruned latent gates with absolute value <= 0.05")
                print(f"  Original active latent gates (>0.0): {l_original_active} / {l_total_gates}")
                print(f"  Final remaining latent gates (>0.05): {l_final_active} / {l_total_gates}")
                
        return ret

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

    def forward(self, x: torch.Tensor, tabular_features: Optional[torch.Tensor] = None, return_features: bool = False, stochastic: bool = False):
        # Input tensor shape x: [B, 3, H, W] or [B, 2, 3, H, W] (multimodal)
        is_multimodal = (x.dim() == 5)
        if is_multimodal:
            B, M, C, H, W = x.shape
            x_flat = x.view(B * M, C, H, W)
            features_flat = self.backbone(x_flat)
            if isinstance(features_flat, tuple):
                features_flat = features_flat[0]
            
            if self.backbone_name.startswith('resnet'):
                _, C_feat, H_feat, W_feat = features_flat.shape
                features = features_flat.view(B, M, C_feat, H_feat, W_feat)
                features = torch.cat([features[:, 0], features[:, 1]], dim=2) # [B, C_feat, 2 * H_feat, W_feat]
            else:
                _, N_patches, D = features_flat.shape
                features = features_flat.view(B, M, N_patches, D)
                features = torch.cat([features[:, 0], features[:, 1]], dim=1) # [B, 2 * N_patches, D]
        else:
            features = self.backbone(x)  # [B, C, H_attn, W_attn] (ResNet) or [B, N_patches, embed_dim] (ViT/ConvNeXt Wrapper)
            if isinstance(features, tuple):
                features = features[0]
            
        if self.backbone_type == 'torchxrayvision':
            if self.use_probabilistic_cbm:
                raise NotImplementedError("Probabilistic CBM is not supported with torchxrayvision yet.")
            supervised_logits, supervised_attn, supervised_features = self.supervised_attention(features)
            concept_logits = supervised_logits
            attn_weights = supervised_attn
            latent_features = None
        elif self.backbone_name.startswith('resnet'):
            # ResNet still uses spatial attention conv
            if self.use_probabilistic_cbm:
                supervised_mean, supervised_logvar, supervised_attn, supervised_features = self.supervised_attention(features)
                self.last_mean = supervised_mean
                self.last_logvar = supervised_logvar
                if self.training or stochastic:
                    std = torch.exp(0.5 * supervised_logvar)
                    eps = torch.randn_like(std)
                    supervised_logits = supervised_mean + std * eps
                else:
                    supervised_logits = supervised_mean
                
                if self.num_latent_concepts > 0:
                    latent_mean, latent_logvar, latent_attn, latent_features = self.latent_attention(features)
                    if self.training or stochastic:
                        std_l = torch.exp(0.5 * latent_logvar)
                        eps_l = torch.randn_like(std_l)
                        latent_logits = latent_mean + std_l * eps_l
                    else:
                        latent_logits = latent_mean
                    concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
                    attn_weights = torch.cat([supervised_attn, latent_attn], dim=1)
                else:
                    concept_logits = supervised_logits
                    attn_weights = supervised_attn
                    latent_features = None
            else:
                supervised_logits, supervised_attn, supervised_features = self.supervised_attention(features)
                if self.num_latent_concepts > 0:
                    latent_logits, latent_attn, latent_features = self.latent_attention(features)
                    concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
                    attn_weights = torch.cat([supervised_attn, latent_attn], dim=1)
                else:
                    concept_logits = supervised_logits
                    attn_weights = supervised_attn
                    latent_features = None
        elif self.use_group_broadcasting:
            # ViT / DINOv2 with Group Broadcasting
            if self.use_probabilistic_cbm:
                supervised_mean, supervised_logvar, supervised_attn, supervised_features = self.supervised_attention(features)
                self.last_mean = supervised_mean
                self.last_logvar = supervised_logvar
                if self.training or stochastic:
                    std = torch.exp(0.5 * supervised_logvar)
                    eps = torch.randn_like(std)
                    supervised_logits = supervised_mean + std * eps
                else:
                    supervised_logits = supervised_mean
                concept_logits = supervised_logits
                attn_weights = supervised_attn
                latent_features = None
            else:
                supervised_logits, supervised_attn, supervised_features = self.supervised_attention(features)
                concept_logits = supervised_logits
                attn_weights = supervised_attn
                latent_features = None
        elif self.use_concept_attention:
            # ViT / DINOv2 with Cross Attention
            if self.use_probabilistic_cbm:
                supervised_mean, supervised_logvar, supervised_attn, supervised_features = self.supervised_attention(features)
                self.last_mean = supervised_mean
                self.last_logvar = supervised_logvar
                if self.training or stochastic:
                    std = torch.exp(0.5 * supervised_logvar)
                    eps = torch.randn_like(std)
                    supervised_logits = supervised_mean + std * eps
                else:
                    supervised_logits = supervised_mean
            else:
                supervised_logits, supervised_attn, supervised_features = self.supervised_attention(features)

            if self.num_latent_concepts > 0:
                # Latent attention is always PatchWiseMLPConceptHead for ViT
                k_val = 3
                B = features.size(0)
                N_patches = features.size(1)
                H_attn = int(math.sqrt(N_patches))
                device = features.device
                D = features.size(-1)

                if self.use_probabilistic_cbm:
                    latent_mean, latent_logvar, latent_topk_indices, latent_weights = self.latent_attention(features, k=k_val, return_weights=True)
                    if self.training or stochastic:
                        std_l = torch.exp(0.5 * latent_logvar)
                        eps_l = torch.randn_like(std_l)
                        latent_logits = latent_mean + std_l * eps_l
                    else:
                        latent_logits = latent_mean
                else:
                    latent_logits, latent_topk_indices, latent_weights = self.latent_attention(features, k=k_val, return_weights=True)

                latent_indices_transposed = latent_topk_indices.permute(0, 2, 1)
                latent_weights_transposed = latent_weights.permute(0, 2, 1)

                from torchvision.transforms.functional import gaussian_blur
                sparse_latent_maps = torch.zeros(B, self.num_latent_concepts, N_patches, device=device)
                sparse_latent_maps.scatter_(2, latent_indices_transposed, latent_weights_transposed)
                if H_attn * H_attn == N_patches:
                    sparse_latent_maps = sparse_latent_maps.view(B, self.num_latent_concepts, H_attn, H_attn)
                    latent_attn = gaussian_blur(sparse_latent_maps, kernel_size=[3, 3], sigma=[1.0, 1.0])
                else:
                    N_single = N_patches // 2
                    H_attn_single = int(math.sqrt(N_single))
                    sparse_l_clinic = sparse_latent_maps[:, :, :N_single].view(B, self.num_latent_concepts, H_attn_single, H_attn_single)
                    sparse_l_derm = sparse_latent_maps[:, :, N_single:].view(B, self.num_latent_concepts, H_attn_single, H_attn_single)
                    l_attn_clinic = gaussian_blur(sparse_l_clinic, kernel_size=[3, 3], sigma=[1.0, 1.0])
                    l_attn_derm = gaussian_blur(sparse_l_derm, kernel_size=[3, 3], sigma=[1.0, 1.0])
                    latent_attn = torch.cat([l_attn_clinic, l_attn_derm], dim=3)

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
        else:
            # ViT / DINOv2 uses PatchWiseMLPConceptHead
            k_val = 3
            if self.use_probabilistic_cbm:
                supervised_mean, supervised_logvar, supervised_topk_indices, supervised_weights = self.supervised_attention(features, k=k_val, return_weights=True)
                self.last_mean = supervised_mean
                self.last_logvar = supervised_logvar
                if self.training or stochastic:
                    std = torch.exp(0.5 * supervised_logvar)
                    eps = torch.randn_like(std)
                    supervised_logits = supervised_mean + std * eps
                else:
                    supervised_logits = supervised_mean
            else:
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
            
            from torchvision.transforms.functional import gaussian_blur
            if H_attn * H_attn == N_patches:
                sparse_maps = sparse_maps.view(B, self.num_supervised_concepts, H_attn, H_attn)
                supervised_attn = gaussian_blur(sparse_maps, kernel_size=[3, 3], sigma=[1.0, 1.0])
            else:
                N_single = N_patches // 2
                H_attn_single = int(math.sqrt(N_single))
                sparse_maps_clinic = sparse_maps[:, :, :N_single].view(B, self.num_supervised_concepts, H_attn_single, H_attn_single)
                sparse_maps_derm = sparse_maps[:, :, N_single:].view(B, self.num_supervised_concepts, H_attn_single, H_attn_single)
                attn_clinic = gaussian_blur(sparse_maps_clinic, kernel_size=[3, 3], sigma=[1.0, 1.0])
                attn_derm = gaussian_blur(sparse_maps_derm, kernel_size=[3, 3], sigma=[1.0, 1.0])
                supervised_attn = torch.cat([attn_clinic, attn_derm], dim=3)
            
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
                if self.use_probabilistic_cbm:
                    latent_mean, latent_logvar, latent_topk_indices, latent_weights = self.latent_attention(features, k=k_val, return_weights=True)
                    if self.training or stochastic:
                        std_l = torch.exp(0.5 * latent_logvar)
                        eps_l = torch.randn_like(std_l)
                        latent_logits = latent_mean + std_l * eps_l
                    else:
                        latent_logits = latent_mean
                else:
                    latent_logits, latent_topk_indices, latent_weights = self.latent_attention(features, k=k_val, return_weights=True)
                
                latent_indices_transposed = latent_topk_indices.permute(0, 2, 1)
                latent_weights_transposed = latent_weights.permute(0, 2, 1)
                
                sparse_latent_maps = torch.zeros(B, self.num_latent_concepts, N_patches, device=device)
                sparse_latent_maps.scatter_(2, latent_indices_transposed, latent_weights_transposed)
                if H_attn * H_attn == N_patches:
                    sparse_latent_maps = sparse_latent_maps.view(B, self.num_latent_concepts, H_attn, H_attn)
                    latent_attn = gaussian_blur(sparse_latent_maps, kernel_size=[3, 3], sigma=[1.0, 1.0])
                else:
                    N_single = N_patches // 2
                    H_attn_single = int(math.sqrt(N_single))
                    sparse_l_clinic = sparse_latent_maps[:, :, :N_single].view(B, self.num_latent_concepts, H_attn_single, H_attn_single)
                    sparse_l_derm = sparse_latent_maps[:, :, N_single:].view(B, self.num_latent_concepts, H_attn_single, H_attn_single)
                    l_attn_clinic = gaussian_blur(sparse_l_clinic, kernel_size=[3, 3], sigma=[1.0, 1.0])
                    l_attn_derm = gaussian_blur(sparse_l_derm, kernel_size=[3, 3], sigma=[1.0, 1.0])
                    latent_attn = torch.cat([l_attn_clinic, l_attn_derm], dim=3)
                
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
        if self.use_multimodal:
            # Legacy path: tabular features concatenated INTO classifier head input
            if tabular_features is None:
                device = concept_logits.device
                tabular_features = torch.zeros(concept_logits.size(0), self.num_tabular_features, device=device)
            combined_logits = torch.cat([concept_logits_dropout, tabular_features], dim=-1)
            class_logits = self.classifier_head(combined_logits)
        elif self.age_sex_skip_connection:
            # Skip-connection path: image CBM logits + tabular skip head (independent channels)
            # X-ray information flows through NAM/Linear, Age/Sex is projected separately.
            class_logits = self.classifier_head(concept_logits_dropout)  # image-only logits
            if tabular_features is not None:
                class_logits = class_logits + self.tabular_skip_head(tabular_features)
        else:
            class_logits = self.classifier_head(concept_logits_dropout)  # [B, num_classes]
        
        if return_features:
            return class_logits, concept_logits, attn_weights, supervised_features, latent_features
        return class_logits, concept_logits, attn_weights

    def freeze_backbone(self):
        """Freezes the vision backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

            

def unfreeze_backbone(self):
    """Unfreezes the vision backbone parameters while maintaining 
    BatchNorm statistics in evaluation mode to prevent training instability.
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
        
        for m in self.backbone.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
                    
        print(f"{BOLD}{GREEN}[Full Unfreeze]{RESET} Activated all backbone parameters, BatchNorm layers fixed.")

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
