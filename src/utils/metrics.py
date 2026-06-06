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


def predict_concepts_from_logits(concept_logits: torch.Tensor, concept_groups_info = None) -> torch.Tensor:
    """Convert concept logits to binary concept predictions for the main concept metrics.

    Multi-class groups are evaluated as mutually exclusive argmax one-hot predictions.
    Single binary/numerical concepts use the standard logit > 0 rule.
    """
    preds_bin = torch.zeros_like(concept_logits)
    num_concepts = concept_logits.shape[-1]

    if concept_groups_info is not None:
        for start_idx, num_feats in concept_groups_info:
            if start_idx >= num_concepts or num_feats <= 0:
                continue
            num_feats = min(num_feats, num_concepts - start_idx)
            group_logits = concept_logits[:, start_idx : start_idx + num_feats]
            if num_feats > 1:
                group_preds = torch.zeros_like(group_logits)
                argmax_idx = torch.argmax(group_logits, dim=-1)
                group_preds.scatter_(1, argmax_idx.unsqueeze(-1), 1.0)
                preds_bin[:, start_idx : start_idx + num_feats] = group_preds
            else:
                preds_bin[:, start_idx] = (group_logits[:, 0] > 0.0).float()
    else:
        preds_bin = (concept_logits > 0.0).float()

    return preds_bin


def _calculate_concept_metric_values(preds_bin: torch.Tensor, concept_targets: torch.Tensor, beta: float = 2.0) -> dict:
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


def calculate_concept_metrics(
    concept_logits: torch.Tensor,
    concept_targets: torch.Tensor,
    concept_groups_info = None,
    beta = 2.0
) -> dict:
    """Calculate main concept metrics.

    The main metric intentionally does not apply confidence thresholds to
    multi-class groups. Categorical concept groups use argmax one-hot
    predictions, while binary/numerical concepts use logit > 0.
    """
    preds_bin = predict_concepts_from_logits(concept_logits, concept_groups_info)
    return _calculate_concept_metric_values(preds_bin, concept_targets, beta=beta)
