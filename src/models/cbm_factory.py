import torch
import torch.nn as nn
import timm
import open_clip

class UniversalFlexibleCBM(nn.Module):
    def __init__(
        self,
        backbone_type: str,
        backbone_name: str,
        num_concepts: int,
        num_classes: int,
        pretrained: bool = True
    ):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.num_concepts = num_concepts
        self.num_classes = num_classes

        # 1. Initialize Backbone
        if self.backbone_type == 'timm':
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        elif self.backbone_type == 'clip':
            # For open_clip, backbone_name can be a model name or a huggingface hub id
            # e.g., "hf-hub:imageomics/bioclip" or "ViT-B-32"
            model, _, _ = open_clip.create_model_and_transforms(backbone_name)
            self.backbone = model.visual
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}. Use 'timm' or 'clip'.")

        # 2. Dynamic Feature Dimension Inference
        feature_dim = self._infer_feature_dim()

        # 3. Layer Construction
        self.projection_layer = nn.Linear(feature_dim, num_concepts)
        self.concept_activation = nn.Sigmoid()
        self.classifier_head = nn.Linear(num_concepts, num_classes)

    def _infer_feature_dim(self) -> int:
        """Pass a dummy tensor through the backbone to dynamically infer feature dimensions."""
        dummy_tensor = torch.randn(1, 3, 224, 224)
        
        was_training = self.backbone.training
        self.backbone.eval()
        
        with torch.no_grad():
            features = self.backbone(dummy_tensor)
            # Handle potential tuple outputs from different backbones
            if isinstance(features, tuple):
                features = features[0]
            feature_dim = features.shape[-1]
            
        if was_training:
            self.backbone.train()
            
        return feature_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
            
        concept_logits = self.projection_layer(features)
        concept_probs = self.concept_activation(concept_logits)
        class_logits = self.classifier_head(concept_probs)
        
        return concept_probs, class_logits

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
