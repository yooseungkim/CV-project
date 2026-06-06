from typing import Dict, Sequence

import torch


MetricDict = Dict[str, float]
ConceptGroup = Dict[str, object]


def calculate_topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk=(1, 3, 5, 10)) -> Dict[int, float]:
    """Calculate Top-K accuracy for target predictions."""
    if outputs.dim() > 1 and targets.dim() > 1 and outputs.shape[-1] == targets.shape[-1]:
        preds = (outputs > 0.0).float()
        correct = (preds == targets).float().sum().item()
        return {1: correct / targets.numel()}

    batch_size = targets.size(0)
    if outputs.dim() == 1 or outputs.shape[-1] <= 1:
        preds = (outputs > 0.0).float()
        correct = (preds == targets.view_as(preds)).float().sum().item()
        return {1: correct / batch_size}

    maxk = min(outputs.shape[-1], max(topk))
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    targets_flat = targets.view(1, -1).expand_as(pred)
    correct = pred.eq(targets_flat)

    res = {}
    for k in topk:
        if k <= outputs.shape[-1]:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[k] = (correct_k / batch_size).item()
    return res


def target_probabilities_from_logits(class_logits: torch.Tensor) -> torch.Tensor:
    """Converts target logits into class probabilities for binary or multiclass heads."""
    if class_logits.dim() > 1 and class_logits.shape[-1] > 1:
        return torch.softmax(class_logits, dim=1)
    pos_probs = torch.sigmoid(class_logits.view(-1))
    return torch.stack([1.0 - pos_probs, pos_probs], dim=1)


def classifier_logits_from_concepts(
    model: torch.nn.Module,
    concept_logits: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Runs the C -> Y head with the calibrated concept logits used by model.forward."""
    return model.classifier_head(model.apply_concept_bias(concept_logits.to(device)))


def calculate_target_macro_fbeta(outputs: torch.Tensor, targets: torch.Tensor, beta: float = 1.0) -> float:
    """Calculate macro F-beta for target predictions."""
    if outputs.dim() > 1 and targets.dim() > 1 and outputs.shape[-1] == targets.shape[-1]:
        preds = (outputs > 0.0).float()
        targets = (targets > 0.5).float()
        tp = (preds * targets).sum(dim=0)
        fp = (preds * (1.0 - targets)).sum(dim=0)
        fn = ((1.0 - preds) * targets).sum(dim=0)
        beta_sq = beta ** 2
        denom = (1.0 + beta_sq) * tp + beta_sq * fn + fp
        scores = torch.where(denom > 0, ((1.0 + beta_sq) * tp) / (denom + 1e-8), torch.zeros_like(tp))
        return scores.mean().item()

    if outputs.dim() == 1 or outputs.shape[-1] <= 1:
        preds = (outputs.view(-1) > 0.0).long()
        targets_flat = targets.view(-1).long()
        num_classes = 2
    else:
        preds = torch.argmax(outputs, dim=1).long()
        targets_flat = targets.view(-1).long()
        num_classes = int(outputs.shape[-1])

    labels = torch.unique(torch.cat([targets_flat, preds]))
    labels = labels[(labels >= 0) & (labels < num_classes)]
    if labels.numel() == 0:
        return 0.0

    scores = []
    beta_sq = beta ** 2
    for label in labels:
        pred_pos = preds == label
        target_pos = targets_flat == label
        tp = (pred_pos & target_pos).sum().float()
        fp = (pred_pos & ~target_pos).sum().float()
        fn = (~pred_pos & target_pos).sum().float()
        denom = (1.0 + beta_sq) * tp + beta_sq * fn + fp
        scores.append(torch.where(denom > 0, ((1.0 + beta_sq) * tp) / (denom + 1e-8), torch.zeros_like(tp)))
    return torch.stack(scores).mean().item()


def calculate_target_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> MetricDict:
    """Calculate Top-1 target accuracy plus macro F1/F2."""
    top1_acc = calculate_topk_accuracy(outputs, targets, topk=(1,)).get(1, 0.0)
    return {
        "acc": top1_acc,
        "macro_f1": calculate_target_macro_fbeta(outputs, targets, beta=1.0),
        "macro_f2": calculate_target_macro_fbeta(outputs, targets, beta=2.0),
    }


@torch.no_grad()
def calculate_classifier_metrics(
    model: torch.nn.Module,
    concept_logits: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> MetricDict:
    class_logits = classifier_logits_from_concepts(model, concept_logits, device).cpu()
    return calculate_target_metrics(class_logits, targets)


def translate_gt_to_logits(
    gt_concepts: torch.Tensor,
    concept_groups: Sequence[ConceptGroup],
    use_probabilistic: bool,
) -> torch.Tensor:
    """
    Convert ground-truth concept values to soft intervention logits.

    For mutually exclusive categorical groups, the positive class receives a high
    probability and the remaining probability mass is spread across alternatives.
    """
    p_pos = 0.999 if use_probabilistic else 0.95
    p_neg_default = 0.001 if use_probabilistic else 0.05

    p_custom = gt_concepts.clone()
    for group in concept_groups:
        indices = list(group["flat_indices"])
        group_size = len(indices)
        if group_size > 1:
            group_gt = gt_concepts[:, indices]
            max_vals, correct_idxs = torch.max(group_gt, dim=1)
            p_others = (1.0 - p_pos) / (group_size - 1)
            group_custom = torch.full_like(group_gt, p_others)
            group_custom.scatter_(1, correct_idxs.unsqueeze(1), p_pos)

            is_positive = (max_vals > 0.5).unsqueeze(1)
            group_fallback = torch.clamp(group_gt, min=p_neg_default, max=p_pos)
            p_custom[:, indices] = torch.where(is_positive, group_custom, group_fallback)
        else:
            c_idx = indices[0]
            p_custom[:, c_idx] = torch.clamp(gt_concepts[:, c_idx], min=p_neg_default, max=p_pos)

    p_custom = torch.clamp(p_custom, min=1e-6, max=1.0 - 1e-6)
    return torch.log(p_custom / (1.0 - p_custom))
