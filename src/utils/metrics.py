import torch

def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk: int = 3) -> float:
    """Calculates Top-K accuracy for target classes. Supports binary, multi-label, and multi-class."""
    if outputs.dim() > 1 and targets.dim() > 1 and outputs.shape[-1] == targets.shape[-1]:
        # Multi-label binary classification (e.g. CheXpert targets)
        preds = (outputs > 0.0).float()
        correct = (preds == targets).float().sum()
        return (correct / targets.numel()).item()
    elif outputs.dim() == 1 or outputs.shape[-1] == 1:
        # Binary classification
        preds = (outputs > 0.0).float() # Assuming outputs are logits
        targets_flat = targets.view_as(preds)
        correct = (preds == targets_flat).float().sum()
        return (correct / targets.size(0)).item()
    else:
        # Multi-class classification
        num_classes = outputs.shape[-1]
        k = min(topk, num_classes)
        targets_flat = targets.view(-1).long()
        _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
        pred = pred.t() # Shape: [k, B]
        correct = pred.eq(targets_flat.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return (correct_k / targets.size(0)).item()

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

def calculate_concept_metrics(concept_logits: torch.Tensor, concept_targets: torch.Tensor, concept_groups_info = None, threshold = 0.0, beta = 2.0) -> dict:
    """Calculates Balanced Accuracy, True Positive Rate (TPR), and True Negative Rate (TNR)
    across all concepts to robustly evaluate models, dynamically adapting to mutually exclusive groups.
    
    Args:
        concept_logits: Raw prediction logits from the concept predictor (pre-activation).
        concept_targets: Ground truth concept targets.
        concept_groups_info: Optional list of (start_idx, num_feats) representing attribute groups.
        threshold: The decision threshold for Sigmoid/Softmax logits. Can be a single float or a tensor of shape [num_concepts].
    """
    preds_bin = torch.zeros_like(concept_logits)
    num_concepts = concept_logits.shape[-1]
    
    # Resolve threshold to a tensor of shape [num_concepts] on the correct device
    if isinstance(threshold, (int, float)):
        thresh_tensor = torch.full((num_concepts,), float(threshold), device=concept_logits.device)
    elif isinstance(threshold, torch.Tensor):
        thresh_tensor = threshold.to(concept_logits.device)
    else:
        thresh_tensor = torch.tensor(threshold, dtype=torch.float32, device=concept_logits.device)
        
    if concept_groups_info is not None:
        for start_idx, num_feats in concept_groups_info:
            group_logits = concept_logits[:, start_idx : start_idx + num_feats]
            if num_feats > 1:
                # Group Softmax prediction: the argmax along the group dimension is active
                group_preds = torch.zeros_like(group_logits)
                max_logits, argmax_idx = torch.max(group_logits, dim=-1)
                
                # Confidence Masking: Only predict the argmax if the model is confident (max logit > threshold)
                g_threshold = thresh_tensor[start_idx : start_idx + num_feats].mean()
                valid_mask = max_logits > g_threshold
                group_preds.scatter_(1, argmax_idx.unsqueeze(-1), 1.0)
                group_preds = group_preds * valid_mask.unsqueeze(-1).float()
                
                preds_bin[:, start_idx : start_idx + num_feats] = group_preds
            else:
                # Sigmoid / 1D binary fallback: threshold at logit > threshold
                g_threshold = thresh_tensor[start_idx]
                preds_bin[:, start_idx : start_idx + num_feats] = (group_logits > g_threshold).float()
    else:
        # Global Sigmoid fallback: threshold all at logit > threshold
        preds_bin = (concept_logits > thresh_tensor.unsqueeze(0)).float()
        
    targets_bin = (concept_targets > 0.5).float()

    tp = (preds_bin * targets_bin).sum(dim=0)
    tn = ((1 - preds_bin) * (1 - targets_bin)).sum(dim=0)
    fp = (preds_bin * (1 - targets_bin)).sum(dim=0)
    fn = ((1 - preds_bin) * targets_bin).sum(dim=0)

    # Robust handling for cases where a class has no positive or negative targets in the current slice
    has_pos = (tp + fn) > 0
    has_neg = (tn + fp) > 0

    tpr = torch.where(has_pos, tp / (tp + fn + 1e-8), torch.ones_like(tp))
    tnr = torch.where(has_neg, tn / (tn + fp + 1e-8), torch.ones_like(tn))
    totals = tp + tn + fp + fn
    accs = torch.where(totals > 0, (tp + tn) / (totals + 1e-8), torch.zeros_like(tp))

    balanced_accs = torch.where(
        has_pos & has_neg,
        (tpr + tnr) / 2.0,
        torch.where(has_pos, tpr, torch.where(has_neg, tnr, torch.ones_like(tpr)))
    )
    
    # Calculate precision, F1-score and F-beta score for concepts
    precision = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
    f1_scores = torch.where(
        tp + fp + fn > 0,
        (2 * tp) / (2 * tp + fp + fn + 1e-8),
        torch.zeros_like(tp)
    )
    
    beta_sq = beta ** 2
    f_beta_scores = torch.where(
        tp + fp + fn > 0,
        (1.0 + beta_sq) * tp / ((1.0 + beta_sq) * tp + beta_sq * fn + fp + 1e-8),
        torch.zeros_like(tp)
    )
    
    metrics = {
        "mean_acc": accs.mean().item(),
        "individual_acc": accs,
        "mean_balanced_acc": balanced_accs.mean().item(),
        "individual_balanced_acc": balanced_accs,
        "tpr": tpr.mean().item(),
        "individual_tpr": tpr,
        "tnr": tnr.mean().item(),
        "individual_tnr": tnr,
        "mean_f1": f1_scores.mean().item(),
        "individual_f1": f1_scores,
        "mean_f_beta": f_beta_scores.mean().item(),
        "individual_f_beta": f_beta_scores
    }
    return metrics


def find_optimal_concept_thresholds(
    concept_logits: torch.Tensor,
    concept_targets: torch.Tensor,
    concept_groups_info = None,
    candidate_thresholds = None
) -> torch.Tensor:
    """
    Finds the optimal threshold for each concept using Youden's J statistic 
    (maximizing Balanced Accuracy) on the validation set.
    
    Args:
        concept_logits: Validation concept prediction logits.
        concept_targets: Validation concept ground truth.
        concept_groups_info: Optional list of (start_idx, num_feats) representing attribute groups.
        candidate_thresholds: List of logit thresholds to search over. 
                             Defaults to 41 points between -4.0 and 4.0.
    
    Returns:
        optimal_thresholds: Tensor of shape [num_concepts] containing the optimal logit threshold for each concept.
    """
    if candidate_thresholds is None:
        # Search in logit space: -4.0 (prob=0.018) to 4.0 (prob=0.982)
        candidate_thresholds = torch.linspace(-4.0, 4.0, 41, device=concept_logits.device)
        
    num_concepts = concept_logits.shape[1]
    optimal_thresholds = torch.zeros(num_concepts, device=concept_logits.device)
    
    # We do the search per concept/group
    if concept_groups_info is not None:
        for start_idx, num_feats in concept_groups_info:
            best_j = -1.0
            best_thresh = 0.0
            
            # For each group, we search for a shared threshold that maximizes the average group F2-score (beta=2.0)
            for thresh in candidate_thresholds:
                # Calculate metrics for this specific group under candidate threshold
                # We can construct a temp threshold tensor
                temp_thresh = torch.tensor([thresh] * num_concepts, device=concept_logits.device)
                metrics = calculate_concept_metrics(concept_logits, concept_targets, concept_groups_info, temp_thresh, beta=2.0)
                
                # Get the group average F2-score
                group_f2 = metrics["individual_f_beta"][start_idx : start_idx + num_feats].mean().item()
                
                if group_f2 > best_j:
                    best_j = group_f2
                    best_thresh = thresh.item()
                    
            for idx in range(start_idx, start_idx + num_feats):
                optimal_thresholds[idx] = best_thresh
    else:
        # Binary fallback: independent F2-score optimization per concept (beta=2.0)
        for c in range(num_concepts):
            best_j = -1.0
            best_thresh = 0.0
            
            c_logits = concept_logits[:, c]
            c_targets = (concept_targets[:, c] > 0.5).float()
            
            # If target has no positive or no negative instances in validation set, default to 0.0 logit
            if c_targets.sum() == 0 or (1 - c_targets).sum() == 0:
                optimal_thresholds[c] = 0.0
                continue
                
            for thresh in candidate_thresholds:
                preds = (c_logits > thresh).float()
                
                tp = (preds * c_targets).sum()
                tn = ((1 - preds) * (1 - c_targets)).sum()
                fp = (preds * (1 - c_targets)).sum()
                fn = ((1 - preds) * c_targets).sum()
                
                # F2-score: beta^2 = 4.0 -> F2 = (5 * TP) / (5 * TP + 4 * FN + FP)
                f2_score = (5.0 * tp) / (5.0 * tp + 4.0 * fn + fp + 1e-8)
                f2_val = f2_score.item()
                
                if f2_val > best_j:
                    best_j = f2_val
                    best_thresh = thresh.item()
                    
            optimal_thresholds[c] = best_thresh
            
    return optimal_thresholds
