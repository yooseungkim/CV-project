import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import open_clip
import math
from typing import Optional, List, Tuple

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
                
    print(f"🛠️ LoRA Injector: Successfully injected LoRA (r={r}, alpha={lora_alpha}) into {len(injected_modules)} ViT attention blocks.")
    return injected_modules


class ViTBackboneWrapper(nn.Module):
    def __init__(self, vit_model):
        """Wrapper for Vision Transformer to dynamically extract patch tokens while ignoring the CLS token.
        Ensures compatibility with internal feature representations and main training pipelines.
        """
        super().__init__()
        self.vit = vit_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features (CLS token + patch tokens)
        feats = self.vit.forward_features(x)
        if isinstance(feats, tuple):
            feats = feats[0]
            
        # Assert type and shape safety to prevent silencing subtle shape bugs or non-standard timm outputs
        assert isinstance(feats, torch.Tensor), f"Expected backbone features to be a torch.Tensor, but got {type(feats)}"
        assert len(feats.shape) == 3, (
            f"Expected 3D patch token tensor [B, N_tokens, C] from ViT backbone, but got shape {feats.shape}. "
            "Ensure global pooling is disabled in timm and your model returns sequence-level features."
        )
        
        # Ignore the CLS token at index 0 and return only patch tokens: [B, N_tokens - 1, C]
        return feats[:, 1:]


class ViTCrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_concepts: int, num_heads: int = 4):
        """Concept-specific Cross-Attention Layer for ViT backbones with Cosine Attention.
        Replaces standard scaled dot-product attention with L2-normalized cosine attention
        to suppress the high-norm border-patch outliers that DINOv2 produces on tightly
        cropped images ("Vision Transformers Need Registers", ICLR 2024).

        Architecture:
          - Learnable concept queries (Q) are L2-normalized.
          - DINOv2 patch tokens projected to keys (K) are L2-normalized.
          - Scores = (Q_norm @ K_norm.T) / temperature  -> softmax -> weighted sum of values (V).
          - Concept logits = cosine(attn_out, concept_proj) / temperature + bias.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim
        
        # Learnable concept queries: [1, num_concepts, embed_dim]
        self.concept_queries = nn.Parameter(torch.randn(1, num_concepts, embed_dim))
        
        # Explicit Q / K / V projections (replacing nn.MultiheadAttention)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Concept-specific projections & biases: [num_concepts, embed_dim]
        self.concept_proj = nn.Parameter(torch.randn(num_concepts, embed_dim))
        self.concept_bias = nn.Parameter(torch.zeros(num_concepts))
        
        # Learnable temperature for cosine similarity scaling (init=1 -> no-op at start)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Surgical initialization: bias prevents Focal Loss logit explosion (RetinaNet Prior pi=0.01)
        pi = 0.01
        bias_init = -math.log((1 - pi) / pi)
        nn.init.constant_(self.concept_bias, bias_init)
        # Xavier uniform for projection layers
        for linear in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(linear.weight)

    def forward(self, features: torch.Tensor):
        # features: patch tokens [B, N_patches, embed_dim] (e.g. N=196 for 14x14 grid)
        B, N, D = features.shape
        
        # --- Project queries, keys, values ---
        queries = self.concept_queries.expand(B, -1, -1)  # [B, num_concepts, D]
        Q = self.q_proj(queries)    # [B, num_concepts, D]
        K = self.k_proj(features)   # [B, N, D]
        V = self.v_proj(features)   # [B, N, D]
        
        # --- Cosine Attention: L2-normalize Q and K along the feature dimension ---
        Q_norm = F.normalize(Q, p=2, dim=-1)  # [B, num_concepts, D]
        K_norm = F.normalize(K, p=2, dim=-1)  # [B, N, D]
        
        # Cosine similarity scores: [B, num_concepts, N]
        attn_scores = torch.bmm(Q_norm, K_norm.transpose(1, 2)) / self.temperature.clamp(min=1e-4)
        
        # Softmax over patch dimension to get normalized attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, num_concepts, N]
        
        # Weighted sum of values: [B, num_concepts, D]
        attn_out = self.out_proj(torch.bmm(attn_weights, V))  # [B, num_concepts, D]
        
        # --- Concept logits via cosine similarity with learnable concept projections ---
        concept_proj_norm = F.normalize(self.concept_proj, p=2, dim=-1)  # [num_concepts, D]
        attn_out_norm = F.normalize(attn_out, p=2, dim=-1)               # [B, num_concepts, D]
        concept_logits = (
            (attn_out_norm * concept_proj_norm.unsqueeze(0)).sum(dim=-1)
            / self.temperature.clamp(min=1e-4)
            + self.concept_bias.unsqueeze(0)
        )  # [B, num_concepts]
        
        # Reshape attention weights: [B, num_concepts, N] -> [B, num_concepts, H_attn, W_attn]
        H_attn = int(math.sqrt(N))
        W_attn = H_attn
        attn_weights_2d = attn_weights.view(B, self.num_concepts, H_attn, W_attn)
        
        return concept_logits, attn_weights_2d, attn_out


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
        concept_groups_info: Optional[List[Tuple[int, int]]] = None
    ):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.backbone_name = backbone_name.lower()
        self.num_supervised_concepts = num_supervised_concepts
        self.num_latent_concepts = num_latent_concepts
        self.num_concepts = num_supervised_concepts + num_latent_concepts
        self.num_classes = num_classes
        self.lora_active = False

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
                            
                    print("🛠️ True 14x14 Dilated Surgery: Modified layer4 strides to (1, 1) and conv dilations to (2, 2) to preserve receptive field alignment.")
                except Exception as e:
                    print(f"⚠️ Warning: Stride surgery failed on resnet backbone: {e}. Falling back to standard stride.")
            
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
            self.backbone = ViTBackboneWrapper(vit_model)
            
            # Apply LoRA 어댑터 주입 if requested
            self.lora_active = use_lora
            if use_lora:
                inject_lora_to_vit(self.backbone, r=lora_r, lora_alpha=lora_alpha)
            
            # Extract embed_dim from the Vit model
            embed_dim = vit_model.embed_dim if hasattr(vit_model, 'embed_dim') else 768
            print(f"🛠️ Dual-Backbone Factory: Configured Cross-Attention CBM for {backbone_name} (embed_dim: {embed_dim}, use_lora: {use_lora})")
            
            # Layer Construction for ViT (using Multihead Cross-Attention)
            self.supervised_attention = ViTCrossAttentionLayer(
                embed_dim=embed_dim,
                num_concepts=num_supervised_concepts
            )
            if self.num_latent_concepts > 0:
                self.latent_attention = ViTCrossAttentionLayer(
                    embed_dim=embed_dim,
                    num_concepts=num_latent_concepts
                )
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}. ResNet and ViT backbones are supported.")

        # Common CBM classification/activation layers
        if concept_groups_info is not None:
            self.concept_activation = GroupSoftmaxActivation(concept_groups_info)
            print(f"🛠️ Dual-Backbone Factory: Activated Mutual Exclusive Group Softmax over {len(concept_groups_info)} groups.")
        else:
            self.concept_activation = nn.Sigmoid()
            print("🛠️ Dual-Backbone Factory: Activated Standard Flat Sigmoid Activation.")
            
        self.dropout = nn.Dropout(p=0.2)
        self.classifier_head = nn.Linear(self.num_concepts, num_classes)

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
            
        # Attention-based concept projection
        supervised_logits, supervised_attn, supervised_features = self.supervised_attention(features)
        
        if self.num_latent_concepts > 0:
            latent_logits, latent_attn, latent_features = self.latent_attention(features)
            concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
            attn_weights = torch.cat([supervised_attn, latent_attn], dim=1)
        else:
            concept_logits = supervised_logits
            attn_weights = supervised_attn
            latent_features = None
        
        # Concept activation (Phase 1 baseline or Group Softmax)
        concept_probs = self.concept_activation(concept_logits)  # [B, num_concepts]
        
        # Apply dropout to regularize concept predictions
        concept_probs_dropout = self.dropout(concept_probs)
        
        # Final classification target output logits
        class_logits = self.classifier_head(concept_probs_dropout)  # [B, num_classes]
        
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
            print("🔓 LoRA Unfreeze: Activated only LoRA adapter parameters for training.")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("🔓 Full Unfreeze: Activated all backbone parameters for training.")

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
