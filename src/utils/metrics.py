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
    """
    preds = (concept_probs > 0.5).float()
    correct = (preds == concept_targets).float().sum()
    total = concept_targets.numel()
    if total == 0:
        return 0.0
    return (correct / total).item()
