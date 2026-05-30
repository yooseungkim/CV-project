import torch
import torch.nn as nn
import timm
import open_clip

class ConceptAttentionLayer(nn.Module):
    def __init__(self, feature_dim: int, num_concepts: int, num_heads: int = 4):
        super().__init__()
        self.num_concepts = num_concepts
        self.feature_dim = feature_dim
        
        # Learnable concept queries: [1, num_concepts, feature_dim]
        self.concept_queries = nn.Parameter(torch.randn(1, num_concepts, feature_dim))
        
        # Multihead Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Projection from attended features to a single scalar per concept
        self.concept_proj = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor):
        # features: [B, C, H, W]
        B, C, H, W = features.shape
        
        # Reshape to [B, H*W, C]
        features_flat = features.flatten(2).transpose(1, 2)
        
        # Expand concept queries for the batch: [B, num_concepts, C]
        queries = self.concept_queries.expand(B, -1, -1)
        
        # Cross-attention: queries=[B, num_concepts, C], keys/values=[B, H*W, C]
        # attn_output: [B, num_concepts, C]
        # attn_weights: [B, num_concepts, H*W]
        attn_output, attn_weights = self.attention(
            query=queries,
            key=features_flat,
            value=features_flat
        )
        
        # Project to concept logits: [B, num_concepts, 1] -> [B, num_concepts]
        concept_logits = self.concept_proj(attn_output).squeeze(-1)
        
        # Reshape attention weights to spatial dimensions: [B, num_concepts, H, W]
        attn_weights = attn_weights.view(B, self.num_concepts, H, W)
        
        return concept_logits, attn_weights


class UniversalFlexibleCBM(nn.Module):
    def __init__(
        self,
        backbone_type: str,
        backbone_name: str,
        num_supervised_concepts: int,
        num_classes: int,
        num_latent_concepts: int = 0,
        pretrained: bool = True
    ):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.num_supervised_concepts = num_supervised_concepts
        self.num_latent_concepts = num_latent_concepts
        self.num_concepts = num_supervised_concepts + num_latent_concepts
        self.num_classes = num_classes

        # 1. Initialize Backbone
        if self.backbone_type == 'timm':
            # Use global_pool='' to get raw 2D feature maps
            self.backbone = timm.create_model(
                backbone_name, 
                pretrained=pretrained, 
                num_classes=0, 
                global_pool=''
            )
        elif self.backbone_type == 'clip':
            raise NotImplementedError(
                "Attention-CBM with raw spatial feature maps is currently only supported "
                "for 'timm' backbones. Support for 'open_clip' requires custom spatial extraction."
            )
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}. Use 'timm' or 'clip'.")

        # 2. Dynamic Feature Dimension Inference
        feature_dim, spatial_h, spatial_w = self._infer_feature_dim()

        # 3. Layer Construction
        self.supervised_attention = ConceptAttentionLayer(
            feature_dim=feature_dim, 
            num_concepts=num_supervised_concepts
        )
        if self.num_latent_concepts > 0:
            self.latent_attention = ConceptAttentionLayer(
                feature_dim=feature_dim, 
                num_concepts=num_latent_concepts
            )
        self.concept_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        self.classifier_head = nn.Linear(self.num_concepts, num_classes)

    def _infer_feature_dim(self) -> tuple[int, int, int]:
        """Pass a dummy tensor through the backbone to dynamically infer feature and spatial dimensions."""
        dummy_tensor = torch.randn(1, 3, 224, 224)
        
        was_training = self.backbone.training
        self.backbone.eval()
        
        with torch.no_grad():
            features = self.backbone(dummy_tensor)
            if isinstance(features, tuple):
                features = features[0]
            
            # Expected shape for spatial features: [1, C, H, W] (e.g. [1, 2048, 7, 7])
            if len(features.shape) != 4:
                raise ValueError(
                    f"Expected 4D output from backbone [B, C, H, W], but got shape {features.shape}. "
                    "Ensure the backbone does not apply global average pooling."
                )
                
            _, C, H, W = features.shape
            
        if was_training:
            self.backbone.train()
            
        return C, H, W

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input tensor shape x: [B, 3, H, W]
        features = self.backbone(x)  # [B, C, H_attn, W_attn]
        if isinstance(features, tuple):
            features = features[0]
            
        # Attention-based concept projection
        supervised_logits, supervised_attn = self.supervised_attention(features)
        
        if self.num_latent_concepts > 0:
            latent_logits, latent_attn = self.latent_attention(features)
            concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
            attn_weights = torch.cat([supervised_attn, latent_attn], dim=1)
        else:
            concept_logits = supervised_logits
            attn_weights = supervised_attn
        
        # Concept activation (Phase 1 baseline)
        concept_probs = self.concept_activation(concept_logits)  # [B, num_concepts]
        
        # Apply dropout to regularize concept predictions
        concept_probs_dropout = self.dropout(concept_probs)
        
        # Final classification target output logits
        class_logits = self.classifier_head(concept_probs_dropout)  # [B, num_classes]
        
        return class_logits, concept_logits, attn_weights

    def freeze_backbone(self):
        """Freezes the vision backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes the vision backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

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
