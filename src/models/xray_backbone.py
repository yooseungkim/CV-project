import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv

class XRayDenseNetCBM(nn.Module):
    """
    Custom PyTorch module wrapping torchxrayvision's DenseNet121
    with a 3-channel input adapter and a 9-dimensional concept head.
    """
    def __init__(self, num_concepts: int = 9, pretrained: bool = True):
        super().__init__()
        
        # 1x1 Conv adapter to convert 3-channel (RGB) to 1-channel (grayscale)
        self.channel_adapter = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            # Initialize weight using standard RGB-to-Grayscale conversion coefficients
            self.channel_adapter.weight.copy_(torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        
        # Load pre-trained DenseNet121 chest X-ray model
        weights = "densenet121-res224-all" if pretrained else None
        self.backbone = xrv.models.DenseNet(weights=weights)
        
        # Replace the original classifier with identity to retrieve features
        self.backbone.classifier = nn.Identity()
        
        # New linear head for predicting 9 concept logits
        self.concept_head = nn.Linear(1024, num_concepts)
        
        # ImageNet normalization buffers to denormalize input pipeline tensors
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Denormalize ImageNet pipeline format back to [0, 1] range
        x = x * self.std + self.mean
        
        # Step 2: Adapt 3-channel input to 1-channel grayscale
        x = self.channel_adapter(x)
        
        # Step 3: Rescale to [-1024, 1024] as required by torchxrayvision DenseNet
        x = (2.0 * x - 1.0) * 1024.0
        
        # Step 4: Extract 1024-dimensional features
        features = self.backbone.features2(x)
        
        # Step 5: Map to 9 concept logits
        logits = self.concept_head(features)
        
        return logits


class XRayDenseNetBackboneWrapper(nn.Module):
    """
    Backbone wrapper for UniversalFlexibleCBM that maps 3-channel inputs to 1024-dimensional features.
    Can return either 1D globally pooled features or 4D spatial feature maps.
    """
    def __init__(self, pretrained: bool = True, return_spatial: bool = False):
        super().__init__()
        self.return_spatial = return_spatial
        
        # 1x1 Conv adapter to convert 3-channel (RGB) to 1-channel (grayscale)
        self.channel_adapter = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.channel_adapter.weight.copy_(torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        
        weights = "densenet121-res224-all" if pretrained else None
        self.backbone = xrv.models.DenseNet(weights=weights)
        self.backbone.classifier = nn.Identity()
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.std + self.mean
        x = self.channel_adapter(x)
        x = (2.0 * x - 1.0) * 1024.0
        
        if self.return_spatial:
            # Return 4D spatial features [B, 1024, 7, 7]
            features = self.backbone.features(x)
            features = F.relu(features, inplace=True)
        else:
            # Return 1D globally pooled features [B, 1024]
            features = self.backbone.features2(x)
            
        return features


class XRayConceptHead(nn.Module):
    """
    Simple Linear concept head adapter for 1D features from DenseNet.
    """
    def __init__(self, embed_dim: int, num_concepts: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_concepts)
        
    def forward(self, features: torch.Tensor):
        logits = self.linear(features)
        B, D = features.shape
        # Return logits, dummy 2D attention maps, and expanded concept features for compatibility
        attn_weights_2d = torch.zeros(B, self.linear.out_features, 7, 7, device=features.device)
        concept_features = features.unsqueeze(1).expand(-1, self.linear.out_features, -1)
        return logits, attn_weights_2d, concept_features
