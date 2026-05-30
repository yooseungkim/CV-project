import torch
import torch.nn as nn
import timm
import open_clip

class ConceptAttentionLayer(nn.Module):
    def __init__(self, feature_dim: int, num_concepts: int, num_heads: int = 4):
        """Concept-specific Spatial Attention Layer.
        Replaces global average pooling and multihead cross-attention. Each concept learns
        its own unique 1x1 conv mapping to spatial location and individual feature projection.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.feature_dim = feature_dim
        
        # 1x1 Conv to produce spatial attention logits for each concept
        self.attention_conv = nn.Conv2d(feature_dim, num_concepts, kernel_size=1)
        
        # Concept-specific weight projections: [num_concepts, feature_dim]
        self.concept_proj = nn.Parameter(torch.randn(num_concepts, feature_dim))
        self.concept_bias = nn.Parameter(torch.zeros(num_concepts))

    def forward(self, features: torch.Tensor):
        # features: [B, C, H, W]
        B, C, H, W = features.shape
        
        # 1. Spatial Attention Map generation
        attn_logits = self.attention_conv(features)  # [B, num_concepts, H, W]
        
        # Softmax over spatial dimensions (H, W)
        attn_weights = torch.softmax(attn_logits.view(B, self.num_concepts, -1), dim=-1).view(B, self.num_concepts, H, W)
        
        # 2. Weighted Sum of features: [B, num_concepts, H*W] x [B, H*W, C] -> [B, num_concepts, C]
        features_flat = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        attn_flat = attn_weights.view(B, self.num_concepts, -1)  # [B, num_concepts, H*W]
        
        weighted_features = torch.bmm(attn_flat, features_flat)  # [B, num_concepts, C]
        
        # 3. Concept Logits prediction
        concept_logits = (weighted_features * self.concept_proj.unsqueeze(0)).sum(dim=-1) + self.concept_bias.unsqueeze(0)  # [B, num_concepts]
        
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
            
            # Apply True 14x14 Spatial Surgery to preserve spatial detail (beaks, eyes, etc.)
            # Changing the stride of layer4's downsampling blocks from 2 to 1 keeps spatial map at 14x14
            if backbone_name.startswith('resnet') and hasattr(self.backbone, 'layer4'):
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
                            
                    # Apply Dilated Convolutions (Atrous Conv) to maintain receptive field alignment (DeepLab style)
                    # We set dilation=2 and padding=2 on all 3x3 convolutions in layer4 blocks
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
