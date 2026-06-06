import torch
from torch.utils.data import Subset
from tqdm import tqdm


def make_train_calibration_indices(num_samples: int, ratio: float, seed: int):
    if ratio <= 0.0:
        return list(range(num_samples)), []
    if ratio >= 1.0:
        raise ValueError("calibration.ratio must be < 1.0")

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator).tolist()
    cal_size = max(1, int(num_samples * ratio))
    cal_indices = perm[:cal_size]
    train_indices = perm[cal_size:]
    if not train_indices:
        raise ValueError("calibration.ratio leaves no training samples")
    return train_indices, cal_indices


def split_for_calibration(train_dataset, calibration_source_dataset, ratio: float, seed: int):
    if len(train_dataset) != len(calibration_source_dataset):
        raise ValueError(
            "Calibration source split must have the same length as the training split "
            f"({len(calibration_source_dataset)} != {len(train_dataset)})."
        )
    train_indices, cal_indices = make_train_calibration_indices(len(train_dataset), ratio, seed)
    return Subset(train_dataset, train_indices), Subset(calibration_source_dataset, cal_indices)


def _normalize_groups(concept_groups_info, num_concepts: int):
    if concept_groups_info is None:
        return [(idx, 1) for idx in range(num_concepts)]
    groups = []
    for start, size in concept_groups_info:
        start = int(start)
        size = int(size)
        if start >= num_concepts:
            continue
        groups.append((start, min(size, num_concepts - start)))
    return groups


def _project_group_biases_(bias: torch.Tensor, groups, max_abs_bias: float):
    for start, size in groups:
        if size > 1:
            group_slice = bias[start : start + size]
            group_slice -= group_slice.mean()
            max_abs = group_slice.abs().max()
            if max_abs > max_abs_bias:
                group_slice *= max_abs_bias / (max_abs + 1e-8)
        else:
            bias[start] = bias[start].clamp(min=-max_abs_bias, max=max_abs_bias)
    return bias


def _predict_concepts(adjusted_logits: torch.Tensor, groups, num_concepts: int):
    preds = torch.zeros(adjusted_logits.size(0), num_concepts, device=adjusted_logits.device)
    for start, size in groups:
        group_logits = adjusted_logits[:, start : start + size]
        if size > 1:
            argmax_idx = torch.argmax(group_logits, dim=1)
            preds[:, start : start + size].scatter_(1, argmax_idx.unsqueeze(1), 1.0)
        else:
            preds[:, start] = (group_logits[:, 0] > 0.0).float()
    return preds


def _macro_fbeta(preds: torch.Tensor, targets: torch.Tensor, beta: float):
    targets = (targets > 0.5).float()
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1.0 - targets)).sum(dim=0)
    fn = ((1.0 - preds) * targets).sum(dim=0)

    beta_sq = beta ** 2
    denom = (1.0 + beta_sq) * tp + beta_sq * fn + fp
    scores = torch.where(
        tp + fp + fn > 0,
        (1.0 + beta_sq) * tp / (denom + 1e-8),
        torch.zeros_like(tp),
    )
    return scores.mean().item()


def _target_macro_fbeta_from_logits(class_logits: torch.Tensor, targets: torch.Tensor, beta: float):
    if class_logits.dim() > 1 and targets.dim() > 1 and class_logits.shape[-1] == targets.shape[-1]:
        preds = (class_logits > 0.0).float()
        targets = (targets > 0.5).float()
        return _macro_fbeta(preds, targets, beta=beta)

    if class_logits.shape[-1] <= 1:
        preds = (class_logits.view(-1) > 0.0).long()
        targets_flat = targets.view(-1).long()
        num_classes = 2
    else:
        preds = torch.argmax(class_logits, dim=1).long()
        targets_flat = targets.view(-1).long()
        num_classes = int(class_logits.shape[-1])

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


def _score_concept_bias(logits, targets, bias, groups, metric_name):
    adjusted_logits = logits + bias.unsqueeze(0)
    preds = _predict_concepts(adjusted_logits, groups, logits.size(1))
    beta = 2.0 if metric_name == "concept_macro_f2" else 1.0
    return _macro_fbeta(preds, targets, beta=beta)


@torch.no_grad()
def _score_target_bias(model, concept_logits, targets, bias, device, metric_name):
    adjusted_logits = concept_logits.clone()
    adjusted_logits[:, :bias.numel()] = adjusted_logits[:, :bias.numel()] + bias.unsqueeze(0)
    class_logits = model.classifier_head(adjusted_logits.to(device)).cpu()
    beta = 2.0 if metric_name == "target_macro_f2" else 1.0
    return _target_macro_fbeta_from_logits(class_logits, targets, beta=beta)


@torch.no_grad()
def collect_calibration_outputs(model, dataloader, num_concepts_supervised, device):
    model.eval()
    all_logits = []
    all_concept_targets = []
    all_target_labels = []

    for images, concepts, targets in tqdm(dataloader, desc="Collecting calibration logits", leave=False):
        images = images.to(device, non_blocking=True)
        class_logits, concept_logits, _ = model(images)
        del class_logits
        all_logits.append(concept_logits.detach().cpu())
        all_concept_targets.append(concepts[:, :num_concepts_supervised].detach().cpu())
        all_target_labels.append(targets.detach().cpu())

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_concept_targets, dim=0),
        torch.cat(all_target_labels, dim=0),
    )


def learn_concept_bias(model, calibration_loader, concept_groups_info, device, config):
    objective = config.get("objective", {})
    metric_name = objective.get("metric", "concept_macro_f1")
    if metric_name not in {"concept_macro_f1", "concept_macro_f2", "target_macro_f1", "target_macro_f2"}:
        raise ValueError(
            "learn_concept_bias.objective.metric must be one of: "
            "concept_macro_f1, concept_macro_f2, target_macro_f1, target_macro_f2"
        )

    search_method = config.get("search_method", "coordinate_grid")
    if search_method != "coordinate_grid":
        raise ValueError("learn_concept_bias.search_method currently supports only coordinate_grid")

    scope = config.get("scope", "supervised_concepts")
    if scope != "supervised_concepts":
        raise ValueError("learn_concept_bias.scope currently supports only supervised_concepts")

    max_abs_bias = float(config.get("max_abs_bias", 4.0))
    num_concepts = model.num_supervised_concepts
    groups = _normalize_groups(concept_groups_info, num_concepts)

    logits, concept_targets, target_labels = collect_calibration_outputs(model, calibration_loader, num_concepts, device)
    supervised_logits = logits[:, :num_concepts]
    bias = torch.zeros(num_concepts)
    candidate_values = torch.linspace(-max_abs_bias, max_abs_bias, steps=17)
    if metric_name in {"target_macro_f1", "target_macro_f2"}:
        score_fn = lambda candidate: _score_target_bias(model, logits, target_labels, candidate, device, metric_name)
    else:
        score_fn = lambda candidate: _score_concept_bias(supervised_logits, concept_targets, candidate, groups, metric_name)
    best_score = score_fn(bias)

    max_passes = 3
    min_improvement = 1e-5
    for pass_idx in range(max_passes):
        pass_improved = False
        for start, size in groups:
            for concept_idx in range(start, start + size):
                local_best_bias = bias
                local_best_score = best_score
                for value in candidate_values:
                    candidate = bias.clone()
                    candidate[concept_idx] = value
                    _project_group_biases_(candidate, [(start, size)], max_abs_bias)
                    score = score_fn(candidate)
                    if score > local_best_score + min_improvement:
                        local_best_score = score
                        local_best_bias = candidate

                if local_best_score > best_score + min_improvement:
                    bias = local_best_bias
                    best_score = local_best_score
                    pass_improved = True

        if not pass_improved:
            break

    _project_group_biases_(bias, groups, max_abs_bias)

    with torch.no_grad():
        model.concept_bias.copy_(bias.to(device=model.concept_bias.device, dtype=model.concept_bias.dtype))

    baseline = score_fn(torch.zeros_like(bias))
    return {
        "baseline_score": baseline,
        "calibrated_score": best_score,
        "metric": metric_name,
        "max_abs_bias": max_abs_bias,
    }
