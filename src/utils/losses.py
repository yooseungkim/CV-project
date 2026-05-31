import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_orthogonality_loss(attn_weights):
    """
    Redesigned Orthogonality Loss (Preventing Attention Collapse).
    Computes pairwise cosine similarity matrix of parent group attention maps
    and penalizes deviation from the Identity matrix using Mean Squared Error (MSE).
    
    attn_weights: Tensor of shape (B, num_groups, H, W) representing mean parent group attention maps.
    """
    B, num_groups, H, W = attn_weights.shape
    if num_groups <= 1:
        return torch.tensor(0.0, device=attn_weights.device)
        
    # Flatten spatial dimensions: (B, num_groups, H * W)
    attn_flat = attn_weights.view(B, num_groups, -1)
    
    # L2 normalize over the spatial dimension to compute stable Cosine Similarity
    attn_norm = attn_flat / (torch.norm(attn_flat, p=2, dim=-1, keepdim=True) + 1e-8)
    
    # Compute pairwise Cosine Similarity matrix: (B, num_groups, num_groups)
    sim_matrix = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
    
    # Define target Identity matrix I_G: (B, num_groups, num_groups)
    identity = torch.eye(num_groups, device=attn_weights.device).unsqueeze(0).expand(B, -1, -1)
    
    # Minimize MSE(S, I) to push off-diagonal elements to 0 while keeping diagonals at 1
    loss_ortho = torch.mean((sim_matrix - identity) ** 2)
    return loss_ortho

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """Numerically stable Sigmoid Focal Loss for multi-label binary concept predictions.
        Focuses learning on hard, misclassified samples and down-weights easy majority classes.
        alpha can be:
        - A single float (constant for all concepts)
        - A torch.Tensor of shape (num_concepts,)
        - None (no alpha weighting applied)
        """
        super().__init__()
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Calculate focal weight: (1 - p_t) ^ gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        loss = focal_weight * bce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                self.alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class GroupCrossEntropyLoss(nn.Module):
    def __init__(self, groups_info: list[tuple[int, int]], lambda_ce: float = 0.1, loss_type: str = 'bce', focal_alpha = None, focal_gamma: float = 2.0):
        """Robust Group-level Softmax Cross Entropy Loss with Loss Scale Balancing.
        Penalizes prediction errors within mutually exclusive attribute categories (Softmax),
        balanced by lambda_ce against independent multi-label concept categories (Sigmoid).
        Supports both BCE and Sigmoid Focal Loss for the Sigmoid/1D fallback categories.
        
        groups_info: list of (start_idx, num_feats)
        lambda_ce: scaling hyperparameter for mutually exclusive cross entropy loss.
        loss_type: loss type for Sigmoid fallback nodes ('bce' or 'focal').
        focal_alpha: alpha weighting factor for Focal Loss (float, torch.Tensor, or None).
        focal_gamma: focus exponent gamma for Focal Loss.
        """
        super().__init__()
        self.groups_info = groups_info
        self.lambda_ce = lambda_ce
        self.loss_type = loss_type.lower()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        active_groups = 0
        
        for start_idx, num_feats in self.groups_info:
            group_logits = logits[:, start_idx : start_idx + num_feats]
            group_targets = targets[:, start_idx : start_idx + num_feats]
            
            if num_feats > 1:
                # 1. Softmax group (Mutually Exclusive Cross Entropy Loss)
                target_sum = group_targets.sum(dim=-1, keepdim=True)
                # Secure target normalization to form a probability distribution (Soft Target)
                group_targets_normalized = group_targets / torch.clamp(target_sum, min=1e-8)
                
                # Apply PyTorch's native cross_entropy supporting Soft Targets
                group_loss = F.cross_entropy(group_logits, group_targets_normalized, reduction='none')
                
                # Mask out samples that do not have annotations for this group
                mask = (target_sum.squeeze(-1) > 0.0).float()
                if mask.sum() > 0:
                    # Apply loss scale balancing multiplier (lambda_ce) to prevent gradient starvation
                    loss += self.lambda_ce * (group_loss * mask).sum() / (mask.sum() + 1e-8)
                    active_groups += 1
            else:
                # 2. Sigmoid / BCE / 1D fallback group (independent binary node)
                if self.loss_type == 'focal':
                    # Calculate focal loss for this individual node
                    probs = torch.sigmoid(group_logits)
                    bce_loss = F.binary_cross_entropy_with_logits(group_logits, group_targets, reduction='none')
                    p_t = probs * group_targets + (1 - probs) * (1 - group_targets)
                    focal_loss = ((1 - p_t) ** self.focal_gamma) * bce_loss
                    
                    if self.focal_alpha is not None:
                        # Extract alpha value for this specific concept dimension
                        if isinstance(self.focal_alpha, torch.Tensor):
                            # self.focal_alpha has shape (num_concepts,)
                            alpha_t = self.focal_alpha[start_idx] * group_targets + (1 - self.focal_alpha[start_idx]) * (1 - group_targets)
                        else:
                            alpha_t = self.focal_alpha * group_targets + (1 - self.focal_alpha) * (1 - group_targets)
                        focal_loss = alpha_t * focal_loss
                        
                    loss += focal_loss.mean()
                    active_groups += 1
                else:
                    # Standard BCE Loss fallback
                    loss += F.binary_cross_entropy_with_logits(group_logits.squeeze(-1), group_targets.squeeze(-1))
                    active_groups += 1
                
        return loss / (active_groups + 1e-8)
