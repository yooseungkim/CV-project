import torch

def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculates accuracy for target classes. Supports binary and multi-class."""
    if outputs.dim() == 1 or outputs.shape[-1] == 1:
        # Binary classification
        preds = (outputs > 0.0).float() # Assuming outputs are logits
        targets_flat = targets.view_as(preds)
    else:
        # Multi-class classification
        preds = torch.argmax(outputs, dim=1)
        targets_flat = targets.view_as(preds)
    
    correct = (preds == targets_flat).float().sum()
    return (correct / targets.size(0)).item()

def calculate_concept_accuracy(concept_probs: torch.Tensor, concept_targets: torch.Tensor) -> float:
    """Calculates average binary accuracy across all concepts.
    Assumes concept_probs are already in [0, 1] (e.g., after Sigmoid).
    Supports both binary targets and continuous probability/score targets by thresholding.
    """
    preds = (concept_probs > 0.5).float()
    targets_bin = (concept_targets > 0.5).float()
    correct = (preds == targets_bin).float().sum()
    total = concept_targets.numel()
    if total == 0:
        return 0.0
    return (correct / total).item()

def calculate_concept_metrics(concept_logits: torch.Tensor, concept_targets: torch.Tensor, concept_groups_info = None, threshold: float = 0.0) -> dict:
    """Calculates Balanced Accuracy, True Positive Rate (TPR), and True Negative Rate (TNR)
    across all concepts to robustly evaluate models, dynamically adapting to mutually exclusive groups.
    
    Args:
        concept_logits: Raw prediction logits from the concept predictor (pre-activation).
        concept_targets: Ground truth concept targets.
        concept_groups_info: Optional list of (start_idx, num_feats) representing attribute groups.
        threshold: The decision threshold for Sigmoid/Softmax logits (default 0.0 is equivalent to probability > 0.5).
    """
    preds_bin = torch.zeros_like(concept_logits)
    
    if concept_groups_info is not None:
        for start_idx, num_feats in concept_groups_info:
            group_logits = concept_logits[:, start_idx : start_idx + num_feats]
            if num_feats > 1:
                # Group Softmax prediction: the argmax along the group dimension is active
                group_preds = torch.zeros_like(group_logits)
                max_logits, argmax_idx = torch.max(group_logits, dim=-1)
                
                # Confidence Masking: Only predict the argmax if the model is confident (max logit > threshold)
                # This elegantly handles occlusions where the ground truth is all-zeros [0, 0, 0, ...]
                valid_mask = max_logits > threshold
                group_preds.scatter_(1, argmax_idx.unsqueeze(-1), 1.0)
                group_preds = group_preds * valid_mask.unsqueeze(-1).float()
                
                preds_bin[:, start_idx : start_idx + num_feats] = group_preds
            else:
                # Sigmoid / 1D binary fallback: threshold at logit > threshold (0.0)
                preds_bin[:, start_idx : start_idx + num_feats] = (group_logits > threshold).float()
    else:
        # Global Sigmoid fallback: threshold all at logit > threshold (0.0)
        preds_bin = (concept_logits > threshold).float()
        
    targets_bin = (concept_targets > 0.5).float()

    tp = (preds_bin * targets_bin).sum(dim=0)
    tn = ((1 - preds_bin) * (1 - targets_bin)).sum(dim=0)
    fp = (preds_bin * (1 - targets_bin)).sum(dim=0)
    fn = ((1 - preds_bin) * targets_bin).sum(dim=0)

    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)

    balanced_accs = (tpr + tnr) / 2.0
    
    metrics = {
        "mean_balanced_acc": balanced_accs.mean().item(),
        "individual_balanced_acc": balanced_accs,
        "tpr": tpr.mean().item(),
        "tnr": tnr.mean().item()
    }
    return metrics
