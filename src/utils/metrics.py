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

def calculate_concept_metrics(concept_logits: torch.Tensor, concept_targets: torch.Tensor, threshold: float = 0.0) -> dict:
    """Calculates Balanced Accuracy, True Positive Rate (TPR), and True Negative Rate (TNR)
    across all concepts to robustly evaluate models on highly sparse concept annotations.
    
    Args:
        concept_logits: Raw prediction logits from the concept predictor (pre-Sigmoid).
        concept_targets: Ground truth concept targets.
        threshold: The decision threshold for logits (default 0.0 is equivalent to Sigmoid > 0.5).
    """
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
