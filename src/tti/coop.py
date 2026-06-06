import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from src.tti.common import (
    ConceptGroup,
    MetricDict,
    calculate_classifier_metrics,
    classifier_logits_from_concepts,
    target_probabilities_from_logits,
    translate_gt_to_logits,
)

MetricFn = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, torch.device], MetricDict]
VALID_SCORE_MODES = {"additive", "product", "power"}


@dataclass(frozen=True)
class CoopFitResult:
    alpha: float
    beta: float
    gamma: float
    score_mode: str
    metric_name: str
    metric_value: float
    budget: int
    search_results: List[Dict[str, object]]


def parse_float_grid(values: str | Sequence[float]) -> List[float]:
    if isinstance(values, str):
        parsed = [float(item.strip()) for item in values.split(",") if item.strip()]
    else:
        parsed = [float(item) for item in values]
    if not parsed:
        raise ValueError("Grid must contain at least one value.")
    return parsed


def parse_coop_costs(costs_arg: Optional[str], concept_groups: Sequence[ConceptGroup]) -> torch.Tensor:
    """Loads optional CooP group acquisition costs; falls back to unit costs."""
    if costs_arg is None:
        return torch.ones(len(concept_groups), dtype=torch.float32)

    if os.path.exists(costs_arg):
        with open(costs_arg, "r", encoding="utf-8") as f:
            loaded_costs = json.load(f)
    else:
        loaded_costs = [float(item.strip()) for item in costs_arg.split(",") if item.strip()]

    if isinstance(loaded_costs, dict):
        costs = [float(loaded_costs.get(str(group["name"]), 1.0)) for group in concept_groups]
    else:
        costs = [float(value) for value in loaded_costs]

    if len(costs) != len(concept_groups):
        raise ValueError(
            f"CooP costs must provide {len(concept_groups)} group-level values, got {len(costs)}."
        )
    if any(cost <= 0.0 for cost in costs):
        raise ValueError("CooP acquisition costs must be positive.")

    return torch.tensor(costs, dtype=torch.float32)


def _minmax_normalize_rows(values: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    row_mins = values.masked_fill(~valid_mask, float("inf")).min(dim=1).values
    row_maxs = values.masked_fill(~valid_mask, float("-inf")).max(dim=1).values
    ranges = row_maxs - row_mins
    normalized = (values - row_mins.unsqueeze(1)) / ranges.clamp_min(eps).unsqueeze(1)
    normalized = torch.where(ranges.unsqueeze(1) > eps, normalized, torch.zeros_like(values))
    return normalized.masked_fill(~valid_mask, 0.0)


def _build_group_candidate_logits(
    num_group_dims: int,
    use_probabilistic: bool,
    device: torch.device,
) -> torch.Tensor:
    p_pos = 0.999 if use_probabilistic else 0.95
    p_neg_default = 0.001 if use_probabilistic else 0.05

    if num_group_dims > 1:
        p_others = (1.0 - p_pos) / (num_group_dims - 1)
        probs = torch.full((num_group_dims, num_group_dims), p_others, device=device)
        probs.fill_diagonal_(p_pos)
    else:
        probs = torch.tensor([[p_neg_default], [p_pos]], dtype=torch.float32, device=device)

    probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
    return torch.log(probs / (1.0 - probs))


def _candidate_weights_from_probs(group_probs: torch.Tensor) -> torch.Tensor:
    if group_probs.shape[1] > 1:
        weights = torch.clamp(group_probs, min=1e-8)
        denom = weights.sum(dim=1, keepdim=True)
        uniform = torch.full_like(weights, 1.0 / group_probs.shape[1])
        return torch.where(denom > 1e-8, weights / denom.clamp_min(1e-8), uniform)

    pos = torch.clamp(group_probs[:, 0], min=1e-6, max=1.0 - 1e-6)
    return torch.stack([1.0 - pos, pos], dim=1)


def _normalized_entropy(weights: torch.Tensor) -> torch.Tensor:
    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=1)
    max_entropy = torch.log(torch.tensor(float(weights.shape[1]), device=weights.device)).clamp_min(1e-8)
    return entropy / max_entropy


def _validate_score_mode(score_mode: str) -> str:
    if score_mode not in VALID_SCORE_MODES:
        raise ValueError(f"Unsupported CooP score_mode: {score_mode}. Expected one of {sorted(VALID_SCORE_MODES)}.")
    return score_mode


def _compute_coop_scores(
    cpu_norm: torch.Tensor,
    cis_norm: torch.Tensor,
    normalized_costs: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    score_mode: str,
) -> torch.Tensor:
    score_mode = _validate_score_mode(score_mode)
    if score_mode == "power":
        scores = (
            torch.pow(cpu_norm.clamp_min(1e-8), alpha)
            * torch.pow(cis_norm.clamp_min(1e-8), beta)
            - gamma * normalized_costs.view(1, -1)
        )
    elif score_mode == "product":
        scores = alpha * cpu_norm * cis_norm - gamma * normalized_costs.view(1, -1)
    else:
        scores = alpha * cpu_norm + beta * cis_norm - gamma * normalized_costs.view(1, -1)
    return scores.masked_fill(~valid_mask, float("-inf"))


@torch.no_grad()
def run_tti_coop_group_level(
    model: torch.nn.Module,
    concept_logits: torch.Tensor,
    gt_concepts: torch.Tensor,
    gt_targets: torch.Tensor,
    concept_groups: Sequence[ConceptGroup],
    device: torch.device,
    budgets: Sequence[int],
    metric_fn: MetricFn = calculate_classifier_metrics,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    costs: Optional[torch.Tensor] = None,
    influence_mode: str = "abs_change",
    score_mode: str = "additive",
    candidate_batch_size: int = 16384,
    show_progress: bool = True,
) -> Tuple[List[Tuple[int, MetricDict]], Dict[str, torch.Tensor]]:
    """Runs CooP group-level TTI and returns metrics at the requested budgets."""
    model.eval()
    score_mode = _validate_score_mode(score_mode)
    num_samples = concept_logits.shape[0]
    num_groups = len(concept_groups)
    max_budget = max(budgets)
    budget_set = set(budgets)
    use_probabilistic = getattr(model, "use_probabilistic_cbm", False)

    if costs is None:
        costs = torch.ones(num_groups, dtype=torch.float32)
    costs = costs.float().cpu()
    if costs.numel() != num_groups:
        raise ValueError(f"Expected {num_groups} CooP costs, got {costs.numel()}.")

    cost_range = costs.max() - costs.min()
    normalized_costs = (costs - costs.min()) / cost_range if cost_range > 1e-8 else torch.zeros_like(costs)

    coop_tti_metrics = [(0, metric_fn(model, concept_logits, gt_targets, device))]
    query_counts = torch.zeros(num_groups, dtype=torch.long)
    first_query_counts = torch.zeros(num_groups, dtype=torch.long)

    logits_mutated = concept_logits.clone()
    gt_logits = translate_gt_to_logits(gt_concepts, concept_groups, use_probabilistic)
    original_concept_probs = model.concept_activation(model.apply_concept_bias(concept_logits.to(device))).cpu()

    revealed_mask = torch.zeros(num_samples, num_groups, dtype=torch.bool)
    sample_indices = torch.arange(num_samples)

    step_iter = range(1, max_budget + 1)
    if show_progress:
        step_iter = tqdm(step_iter, desc="Simulating CooP Group TTI")

    for step in step_iter:
        current_logits = classifier_logits_from_concepts(model, logits_mutated, device)
        current_probs = target_probabilities_from_logits(current_logits).cpu()
        current_classes = torch.argmax(current_probs, dim=1)
        base_scores = current_probs.gather(1, current_classes.unsqueeze(1)).squeeze(1)

        valid_mask = ~revealed_mask
        cpu_scores = torch.zeros(num_samples, num_groups, dtype=torch.float32)
        cis_scores = torch.zeros(num_samples, num_groups, dtype=torch.float32)

        for group_idx, group in enumerate(concept_groups):
            candidate_rows = valid_mask[:, group_idx]
            if not candidate_rows.any():
                continue

            indices = list(group["flat_indices"])
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            group_probs = original_concept_probs[:, indices]
            candidate_weights = _candidate_weights_from_probs(group_probs)
            cpu_scores[:, group_idx] = _normalized_entropy(candidate_weights)

            valid_indices = candidate_rows.nonzero(as_tuple=True)[0]
            candidate_logits = _build_group_candidate_logits(
                len(indices),
                use_probabilistic=use_probabilistic,
                device=logits_mutated.device,
            )
            num_candidates = candidate_logits.shape[0]
            max_rows = (
                max(1, candidate_batch_size // num_candidates)
                if candidate_batch_size is not None and candidate_batch_size > 0
                else valid_indices.numel()
            )

            for chunk_indices in torch.split(valid_indices, max_rows):
                candidate_inputs = logits_mutated[chunk_indices].repeat_interleave(num_candidates, dim=0)
                repeated_candidate_logits = candidate_logits.repeat(chunk_indices.numel(), 1)
                candidate_inputs[:, idx_tensor] = repeated_candidate_logits

                candidate_class_logits = classifier_logits_from_concepts(model, candidate_inputs, device)
                candidate_class_probs = target_probabilities_from_logits(candidate_class_logits).cpu()
                repeated_current_classes = current_classes[chunk_indices].repeat_interleave(num_candidates)
                candidate_scores = candidate_class_probs.gather(
                    1,
                    repeated_current_classes.unsqueeze(1),
                ).view(chunk_indices.numel(), num_candidates)

                base_scores_for_group = base_scores[chunk_indices].unsqueeze(1)
                if influence_mode == "confidence_drop":
                    influence = torch.clamp(base_scores_for_group - candidate_scores, min=0.0)
                elif influence_mode == "paper_delta":
                    influence = candidate_scores - base_scores_for_group
                else:
                    influence = torch.abs(candidate_scores - base_scores_for_group)

                weights_for_group = candidate_weights[chunk_indices]
                cis_scores[chunk_indices, group_idx] = (weights_for_group * influence).sum(dim=1)

        cpu_norm = _minmax_normalize_rows(cpu_scores, valid_mask)
        cis_norm = _minmax_normalize_rows(cis_scores, valid_mask)
        scores = _compute_coop_scores(
            cpu_norm=cpu_norm,
            cis_norm=cis_norm,
            normalized_costs=normalized_costs,
            valid_mask=valid_mask,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            score_mode=score_mode,
        )
        selected_groups = torch.argmax(scores, dim=1)

        if step == 1:
            first_query_counts += torch.bincount(selected_groups, minlength=num_groups)
        query_counts += torch.bincount(selected_groups, minlength=num_groups)

        for group_idx, group in enumerate(concept_groups):
            rows = (selected_groups == group_idx).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue

            indices = torch.tensor(list(group["flat_indices"]), dtype=torch.long)
            logits_mutated[rows.unsqueeze(1), indices.unsqueeze(0)] = gt_logits[
                rows.unsqueeze(1),
                indices.unsqueeze(0),
            ]

        revealed_mask[sample_indices, selected_groups] = True

        if step in budget_set:
            updated_metrics = metric_fn(model, logits_mutated, gt_targets, device)
            coop_tti_metrics.append((step, updated_metrics))
            if show_progress and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix(acc=f"{updated_metrics.get('acc', 0.0) * 100:.2f}%")

    query_stats = {"total": query_counts, "first": first_query_counts}
    return coop_tti_metrics, query_stats


def fit_coop_parameters(
    model: torch.nn.Module,
    concept_logits: torch.Tensor,
    gt_concepts: torch.Tensor,
    gt_targets: torch.Tensor,
    concept_groups: Sequence[ConceptGroup],
    device: torch.device,
    alpha_grid: Iterable[float],
    gamma_grid: Iterable[float],
    fit_budget: int,
    metric_fn: MetricFn = calculate_classifier_metrics,
    metric_name: str = "acc",
    beta: float = 1.0,
    beta_grid: Optional[Iterable[float]] = None,
    costs: Optional[torch.Tensor] = None,
    influence_mode: str = "abs_change",
    score_mode: str = "additive",
    candidate_batch_size: int = 16384,
) -> CoopFitResult:
    """Fits CooP score parameters by grid search on a held-out validation set."""
    score_mode = _validate_score_mode(score_mode)
    best: Optional[CoopFitResult] = None
    search_results: List[Dict[str, object]] = []
    budgets = [0, int(fit_budget)]
    beta_candidates = [float(value) for value in beta_grid] if beta_grid is not None else [float(beta)]

    for alpha in alpha_grid:
        for beta_candidate in beta_candidates:
            for gamma in gamma_grid:
                results, _ = run_tti_coop_group_level(
                    model=model,
                    concept_logits=concept_logits,
                    gt_concepts=gt_concepts,
                    gt_targets=gt_targets,
                    concept_groups=concept_groups,
                    device=device,
                    budgets=budgets,
                    metric_fn=metric_fn,
                    alpha=float(alpha),
                    beta=float(beta_candidate),
                    gamma=float(gamma),
                    costs=costs,
                    influence_mode=influence_mode,
                    score_mode=score_mode,
                    candidate_batch_size=candidate_batch_size,
                    show_progress=False,
                )
                metrics = dict(results)[int(fit_budget)]
                if metric_name not in metrics:
                    raise KeyError(f"Metric '{metric_name}' was not returned by metric_fn. Available: {sorted(metrics)}")
                metric_value = float(metrics[metric_name])
                row = {
                    "alpha": float(alpha),
                    "beta": float(beta_candidate),
                    "gamma": float(gamma),
                    "score_mode": score_mode,
                    metric_name: metric_value,
                    "acc": float(metrics.get("acc", 0.0)),
                    "macro_f1": float(metrics.get("macro_f1", 0.0)),
                    "macro_f2": float(metrics.get("macro_f2", 0.0)),
                }
                search_results.append(row)

                if best is None or metric_value > best.metric_value:
                    best = CoopFitResult(
                        alpha=float(alpha),
                        beta=float(beta_candidate),
                        gamma=float(gamma),
                        score_mode=score_mode,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        budget=int(fit_budget),
                        search_results=[],
                    )

    assert best is not None
    return CoopFitResult(
        alpha=best.alpha,
        beta=best.beta,
        gamma=best.gamma,
        score_mode=best.score_mode,
        metric_name=best.metric_name,
        metric_value=best.metric_value,
        budget=best.budget,
        search_results=search_results,
    )
