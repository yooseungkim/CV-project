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


class AsymmetricLossWithWeight(nn.Module):
    """비대칭 손실 함수 (ASL): gamma_pos와 gamma_neg를 분리하여 Focal Loss의 "Gamma Trap" 해결.
    positive 예측에 대한 gradient decay를 제거하고, easy negative의 gradient noise를 강력히 억제.
    """
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                 alpha_pos: float = 1.2, clip: float = 0.05,
                 reduction: str = 'mean', cb_pos_weight: torch.Tensor = None,
                 cb_neg_weight: torch.Tensor = None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha_pos = alpha_pos
        self.clip = clip
        self.reduction = reduction
        self.cb_pos_weight = cb_pos_weight
        self.cb_neg_weight = cb_neg_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Positive / Negative 분리
        pos_probs = probs
        neg_probs = 1.0 - probs

        # 비대칭 클리핑: negative 확률을 하한 shift하여 easy negative gradient 추가 억제
        if self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1.0)

        # 수치 안정 로그 계산
        pos_log = torch.clamp(pos_probs, min=1e-8).log()
        neg_log = torch.clamp(neg_probs, min=1e-8).log()

        # Focal modulator 적용 (gamma_pos=0 → positive decay 없음)
        pos_loss = -targets * pos_log * ((1.0 - pos_probs) ** self.gamma_pos)
        neg_loss = -(1.0 - targets) * neg_log * (probs ** self.gamma_neg)

        # Class-Balanced weighting
        if self.cb_pos_weight is not None:
            pos_loss = self.cb_pos_weight * pos_loss
        if self.cb_neg_weight is not None:
            neg_loss = self.cb_neg_weight * neg_loss

        # alpha_pos 가중치: positive에 static weight 부여
        loss = self.alpha_pos * pos_loss + neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class GroupCrossEntropyLoss(nn.Module):
    def __init__(self, groups_info: list[tuple[int, int]], lambda_ce: float = 0.1, loss_type: str = 'bce',
                 focal_alpha = None, focal_gamma: float = 2.0,
                 asl_gamma_pos: float = 0.0, asl_gamma_neg: float = 4.0,
                 asl_alpha_pos: float = 1.2, asl_clip: float = 0.05,
                 cb_pos_weight: torch.Tensor = None, cb_neg_weight: torch.Tensor = None):
        """Robust Group-level Softmax Cross Entropy Loss with Loss Scale Balancing.
        Supports BCE, Sigmoid Focal Loss, and ASL for the Sigmoid/1D fallback categories.
        """
        super().__init__()
        self.groups_info = groups_info
        self.lambda_ce = lambda_ce
        self.loss_type = loss_type.lower()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.asl_gamma_pos = asl_gamma_pos
        self.asl_gamma_neg = asl_gamma_neg
        self.asl_alpha_pos = asl_alpha_pos
        self.asl_clip = asl_clip
        self.cb_pos_weight = cb_pos_weight
        self.cb_neg_weight = cb_neg_weight
        
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
                
                # Apply Class-Balanced Loss weighting for softmax groups
                if self.cb_pos_weight is not None:
                    group_cb_pos = self.cb_pos_weight[start_idx : start_idx + num_feats]
                    sum_group_cb = group_cb_pos.sum()
                    if sum_group_cb > 0:
                        norm_cb_pos = num_feats * group_cb_pos / sum_group_cb
                    else:
                        norm_cb_pos = torch.ones_like(group_cb_pos)
                    sample_weight = (group_targets_normalized * norm_cb_pos.view(1, -1)).sum(dim=-1)
                    group_loss = sample_weight * group_loss
                
                # Mask out samples that do not have annotations for this group
                mask = (target_sum.squeeze(-1) > 0.0).float()
                if mask.sum() > 0:
                    # Apply loss scale balancing multiplier (lambda_ce) to prevent gradient starvation
                    loss += self.lambda_ce * (group_loss * mask).sum() / (mask.sum() + 1e-8)
                    active_groups += 1
            else:
                # 2. Sigmoid / BCE / 1D fallback group (independent binary node)
                if self.loss_type == 'asl':
                    # ASL for 1D fallback
                    probs = torch.sigmoid(group_logits)
                    neg_probs = 1.0 - probs
                    if self.asl_clip > 0:
                        neg_probs = (neg_probs + self.asl_clip).clamp(max=1.0)
                    pos_log = torch.clamp(probs, min=1e-8).log()
                    neg_log = torch.clamp(neg_probs, min=1e-8).log()
                    pos_loss = -group_targets * pos_log * ((1.0 - probs) ** self.asl_gamma_pos)
                    neg_loss = -(1.0 - group_targets) * neg_log * (probs ** self.asl_gamma_neg)
                    
                    if self.cb_pos_weight is not None:
                        pos_loss = self.cb_pos_weight[start_idx] * pos_loss
                    if self.cb_neg_weight is not None:
                        neg_loss = self.cb_neg_weight[start_idx] * neg_loss
                        
                    asl_loss = self.asl_alpha_pos * pos_loss + neg_loss
                    loss += asl_loss.mean()
                    active_groups += 1
                elif self.loss_type == 'focal':
                    # Calculate focal loss for this individual node
                    probs = torch.sigmoid(group_logits)
                    bce_loss = F.binary_cross_entropy_with_logits(group_logits, group_targets, reduction='none')
                    p_t = probs * group_targets + (1 - probs) * (1 - group_targets)
                    focal_loss = ((1 - p_t) ** self.focal_gamma) * bce_loss
                    
                    if self.cb_pos_weight is not None and self.cb_neg_weight is not None:
                        w_pos = self.cb_pos_weight[start_idx]
                        w_neg = self.cb_neg_weight[start_idx]
                        cb_weight = group_targets * w_pos + (1.0 - group_targets) * w_neg
                        focal_loss = cb_weight * focal_loss
                        
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
                    bce_loss_raw = F.binary_cross_entropy_with_logits(group_logits.squeeze(-1), group_targets.squeeze(-1), reduction='none')
                    if self.cb_pos_weight is not None and self.cb_neg_weight is not None:
                        w_pos = self.cb_pos_weight[start_idx]
                        w_neg = self.cb_neg_weight[start_idx]
                        cb_weight = group_targets.squeeze(-1) * w_pos + (1.0 - group_targets.squeeze(-1)) * w_neg
                        loss += (bce_loss_raw * cb_weight).mean()
                    else:
                        loss += bce_loss_raw.mean()
                    active_groups += 1
                
        return loss / (active_groups + 1e-8)
