import torch
import torch.nn.functional as F
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


def _target_labels_for_ce(targets: torch.Tensor):
    if targets.ndim > 1:
        targets = targets.squeeze(-1) if targets.size(-1) == 1 else targets.argmax(dim=1)
    return targets.long()


def _calibration_mask_for_parameterization(parameterization: str, groups, num_concepts: int):
    mask = torch.zeros(num_concepts, dtype=torch.bool)
    if parameterization == "coordinate":
        mask.fill_(True)
    elif parameterization == "singleton_only":
        for start, size in groups:
            if size == 1:
                mask[start] = True
    return mask


def _apply_candidate_calibration(logits, bias, calibration_mask, temperature=1.0):
    adjusted_logits = logits.clone()
    bias = bias.to(device=logits.device, dtype=logits.dtype)
    calibration_mask = calibration_mask.to(device=logits.device)
    if torch.any(calibration_mask):
        adjusted_logits[:, calibration_mask] = (
            adjusted_logits[:, calibration_mask] / float(temperature)
            + bias[calibration_mask].unsqueeze(0)
        )
    return adjusted_logits


def _score_concept_bias(logits, targets, bias, calibration_mask, temperature, groups, metric_name):
    adjusted_logits = _apply_candidate_calibration(logits, bias, calibration_mask, temperature)
    preds = _predict_concepts(adjusted_logits, groups, logits.size(1))
    beta = 2.0 if metric_name == "concept_macro_f2" else 1.0
    return _macro_fbeta(preds, targets, beta=beta)


@torch.no_grad()
def _score_target_bias(model, concept_logits, targets, bias, calibration_mask, device, metric_name, temperature=1.0):
    adjusted_logits = concept_logits.clone()
    adjusted_logits[:, :bias.numel()] = _apply_candidate_calibration(
        adjusted_logits[:, :bias.numel()],
        bias,
        calibration_mask,
        temperature,
    )
    class_logits = model.classifier_head(adjusted_logits.to(device)).cpu()
    if metric_name == "target_acc":
        if class_logits.shape[-1] <= 1:
            preds = (class_logits.view(-1) > 0.0).long()
            labels = _target_labels_for_ce(targets).view(-1)
        else:
            preds = torch.argmax(class_logits, dim=1).long()
            labels = _target_labels_for_ce(targets).view(-1)
        return (preds == labels).float().mean().item()
    if metric_name == "target_nll":
        return -float(F.cross_entropy(class_logits, _target_labels_for_ce(targets)).item())
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
    valid_metrics = {
        "concept_macro_f1",
        "concept_macro_f2",
        "target_macro_f1",
        "target_macro_f2",
        "target_acc",
        "target_nll",
    }
    if metric_name not in valid_metrics:
        raise ValueError(
            "learn_concept_bias.objective.metric must be one of: "
            + ", ".join(sorted(valid_metrics))
        )

    search_method = config.get("search_method", "coordinate_grid")
    if search_method != "coordinate_grid":
        raise ValueError("learn_concept_bias.search_method currently supports only coordinate_grid")

    scope = config.get("scope", "supervised_concepts")
    if scope != "supervised_concepts":
        raise ValueError("learn_concept_bias.scope currently supports only supervised_concepts")

    parameterization = config.get("parameterization", config.get("bias_parameterization", "coordinate"))
    if parameterization not in {"coordinate", "singleton_only"}:
        raise ValueError("learn_concept_bias.parameterization must be one of: coordinate, singleton_only")

    max_abs_bias = float(config.get("max_abs_bias", 4.0))
    grid_steps = int(config.get("grid_steps", 17))
    max_passes = int(config.get("max_passes", 3))
    min_improvement = float(config.get("min_improvement", 1e-5))
    l2_lambda = float(config.get("l2_lambda", 0.0))
    temperature = float(config.get("temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError("learn_concept_bias.temperature must be > 0")
    if grid_steps < 1:
        raise ValueError("learn_concept_bias.grid_steps must be >= 1")

    num_concepts = model.num_supervised_concepts
    groups = _normalize_groups(concept_groups_info, num_concepts)
    calibration_mask = _calibration_mask_for_parameterization(parameterization, groups, num_concepts)
    search_indices = torch.nonzero(calibration_mask, as_tuple=False).view(-1).tolist()

    logits, concept_targets, target_labels = collect_calibration_outputs(model, calibration_loader, num_concepts, device)
    supervised_logits = logits[:, :num_concepts]
    candidate_values = torch.linspace(-max_abs_bias, max_abs_bias, steps=grid_steps)

    if metric_name.startswith("target_"):
        score_fn = lambda candidate: _score_target_bias(
            model,
            logits,
            target_labels,
            candidate,
            calibration_mask,
            device,
            metric_name,
            temperature=temperature,
        )
    else:
        score_fn = lambda candidate: _score_concept_bias(
            supervised_logits,
            concept_targets,
            candidate,
            calibration_mask,
            temperature,
            groups,
            metric_name,
        )

    def penalized_score(candidate):
        raw_score = score_fn(candidate)
        penalty = l2_lambda * candidate.pow(2).sum().item()
        return raw_score - penalty, raw_score

    bias = torch.zeros(num_concepts)
    best_objective, best_score = penalized_score(bias)

    for _ in range(max_passes):
        pass_improved = False
        for concept_idx in search_indices:
            local_best_bias = bias.clone()
            local_best_objective = best_objective
            local_best_score = best_score
            for value in candidate_values:
                candidate = bias.clone()
                candidate[concept_idx] = value
                if parameterization == "coordinate":
                    for start, size in groups:
                        if start <= concept_idx < start + size:
                            _project_group_biases_(candidate, [(start, size)], max_abs_bias)
                            break
                objective_value, raw_score = penalized_score(candidate)
                if objective_value > local_best_objective + min_improvement:
                    local_best_objective = objective_value
                    local_best_score = raw_score
                    local_best_bias = candidate

            if local_best_objective > best_objective + min_improvement:
                bias = local_best_bias
                best_objective = local_best_objective
                best_score = local_best_score
                pass_improved = True

        if not pass_improved:
            break

    with torch.no_grad():
        model.concept_bias.copy_(bias.to(device=model.concept_bias.device, dtype=model.concept_bias.dtype))
        if hasattr(model, "concept_bias_mask"):
            model.concept_bias_mask.copy_(
                calibration_mask.to(device=model.concept_bias_mask.device, dtype=model.concept_bias_mask.dtype)
            )
        if hasattr(model, "concept_bias_temperature"):
            model.concept_bias_temperature.fill_(temperature)

    baseline = score_fn(torch.zeros_like(bias))
    summary = {
        "baseline_score": baseline,
        "calibrated_score": best_score,
        "penalized_score": best_objective,
        "metric": metric_name,
        "parameterization": parameterization,
        "max_abs_bias": max_abs_bias,
        "grid_steps": grid_steps,
        "max_passes": max_passes,
        "temperature": temperature,
        "l2_lambda": l2_lambda,
        "num_search_concepts": len(search_indices),
        "bias_l2": float(bias.pow(2).sum().sqrt().item()),
        "bias_mean_abs": float(bias.abs().mean().item()),
        "bias_max_abs": float(bias.abs().max().item()),
        "nonzero_bias": int((bias.abs() > 1e-8).sum().item()),
    }
    return summary
