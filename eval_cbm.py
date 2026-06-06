import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.cub import CUB2011Dataset
from src.data.derm7pt import Derm7PtDataset
from src.data.milk10k import MILK10KDataset
from src.data.chexpert import CheXpertDataset
from src.models.cbm_factory import UniversalFlexibleCBM
from src.tti.common import (
    calculate_classifier_metrics,
    calculate_target_metrics,
    translate_gt_to_logits,
)
from src.tti.coop import (
    fit_alpha_gamma,
    parse_coop_costs,
    parse_float_grid,
    run_tti_coop_group_level,
)
from src.utils.metrics import calculate_concept_metrics

# ANSI terminal colors for highlighting
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"
TTI_K_VALUES = (1, 2, 3, 5, 10)
TTI_METRICS = (
    ("acc", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("macro_f2", "Macro-F2"),
)

def parse_args():
    parser = argparse.ArgumentParser(description="Concept Bottleneck Model Evaluation & Test-Time Intervention (TTI) Benchmark")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to saved CBM model checkpoint (.pt or .pth)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for testing")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Computation device")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loader")
    parser.add_argument('--without-tti', action='store_true', help="Skip the TTI benchmark and run only standard CBM evaluation")
    parser.add_argument('--ignore-bias', action='store_true', help="Ignore saved concept_bias by zeroing it before evaluation")
    parser.add_argument('--without-coop-tti', '--skip_coop_tti', action='store_true', dest='without_coop_tti', help="Skip CooP policy evaluation for group-level TTI")
    parser.add_argument('--without-coop-fit', action='store_true', help="Skip validation fitting and use --coop-alpha/--coop-gamma directly")
    parser.add_argument('--coop-alpha', '--coop_alpha', type=float, default=1.0, dest='coop_alpha', help="Manual CooP weight for concept prediction uncertainty (CPU), used when --without-coop-fit is set")
    parser.add_argument('--coop-beta', '--coop_beta', type=float, default=1.0, dest='coop_beta', help="CooP weight for concept importance score (CIS)")
    parser.add_argument('--coop-gamma', '--coop_gamma', type=float, default=0.0, dest='coop_gamma', help="Manual CooP weight for acquisition cost, used when --without-coop-fit is set")
    parser.add_argument('--coop-alpha-grid', type=str, default='0,0.25,0.5,1,2,4', help="Comma-separated alpha values for validation fitting")
    parser.add_argument('--coop-gamma-grid', type=str, default='0,0.25,0.5,1,2,4', help="Comma-separated gamma values for validation fitting")
    parser.add_argument('--coop-fit-k', type=int, default=5, help="Group-level TTI budget K used to select fitted alpha/gamma on validation")
    parser.add_argument('--coop-fit-metric', type=str, default='acc', choices=['acc', 'macro_f1', 'macro_f2'], help="Validation metric used to select fitted alpha/gamma")
    parser.add_argument('--coop-costs', '--coop_costs', type=str, default=None, dest='coop_costs', help="Optional comma-separated group costs or JSON file containing a list or group-name mapping")
    parser.add_argument('--coop-candidate-batch-size', '--coop_candidate_batch_size', type=int, default=16384, dest='coop_candidate_batch_size', help="Maximum number of CooP counterfactual candidate rows evaluated per classifier-head call")
    parser.add_argument(
        '--coop-influence-mode',
        '--coop_influence_mode',
        type=str,
        default='abs_change',
        dest='coop_influence_mode',
        choices=['abs_change', 'confidence_drop', 'paper_delta'],
        help="How CooP converts candidate interventions into CIS. abs_change is the default influence magnitude."
    )
    return parser.parse_args()


def resolve_backbone_train_mode(checkpoint_args, checkpoint_config, state_dict):
    """Resolve backbone mode from new metadata, old metadata, or checkpoint keys."""
    valid_modes = {"frozen", "lora", "full"}
    checkpoint_args = checkpoint_args or {}
    checkpoint_config = checkpoint_config or {}
    state_dict = state_dict or {}
    bb_cfg = checkpoint_config.get("backbone", {}) if isinstance(checkpoint_config, dict) else {}
    mode = checkpoint_args.get("backbone_train_mode") or bb_cfg.get("backbone_train_mode")
    has_lora_keys = any("lora_" in key for key in state_dict.keys())

    if has_lora_keys:
        return "lora"
    if mode is not None:
        mode = str(mode).lower()
        if mode not in valid_modes:
            raise ValueError(f"Unsupported backbone_train_mode in checkpoint: {mode}. Expected one of {sorted(valid_modes)}.")
        return mode
    if checkpoint_args.get("use_lora") or bb_cfg.get("use_lora"):
        return "lora"
    if checkpoint_args.get("freeze_backbone") or bb_cfg.get("freeze_backbone"):
        return "frozen"
    return "full"


def make_tti_budgets(max_k, requested_k_values=TTI_K_VALUES):
    """Return baseline plus requested TTI budgets, clipped to the available intervention count."""
    if max_k <= 0:
        return [0]

    budgets = [0]
    for k in requested_k_values:
        clipped_k = min(int(k), max_k)
        if clipped_k > 0 and clipped_k not in budgets:
            budgets.append(clipped_k)
    if max_k not in budgets:
        budgets.append(max_k)
    return budgets



@torch.no_grad()
def run_evaluation(model, dataloader, concept_groups_info, concept_groups, device, desc="Evaluating CBM"):
    """Runs a standard evaluation pass over the dataloader."""
    model.eval()

    all_class_logits = []
    all_concept_logits = []
    all_gt_concepts = []
    all_gt_targets = []
    
    pbar = tqdm(dataloader, desc=desc)
    for images, concepts, targets in pbar:
        images = images.to(device)
        concepts = concepts.to(device)
        targets = targets.to(device)
        
        # Forward pass
        class_logits, concept_logits, _ = model(images)
        
        all_class_logits.append(class_logits.cpu())
        all_concept_logits.append(concept_logits.cpu())
        all_gt_concepts.append(concepts.cpu())
        all_gt_targets.append(targets.cpu())
        
    # Concatenate all test outputs
    all_class_logits = torch.cat(all_class_logits, dim=0)
    all_concept_logits = torch.cat(all_concept_logits, dim=0)
    all_gt_concepts = torch.cat(all_gt_concepts, dim=0)
    all_gt_targets = torch.cat(all_gt_targets, dim=0)
    
    biased_concept_logits = model.apply_concept_bias(all_concept_logits.to(device)).cpu()
    target_metrics = calculate_target_metrics(all_class_logits, all_gt_targets)

    # Compute classification metrics from GT concepts: GT concepts -> label.
    # The classifier head is trained in concept-logit space, matching the TTI intervention path.
    gt_concept_logits = translate_gt_to_logits(
        all_gt_concepts,
        concept_groups,
        getattr(model, "use_probabilistic_cbm", False)
    )
    if model.num_latent_concepts > 0:
        latent_zeros = torch.zeros(
            gt_concept_logits.size(0),
            model.num_latent_concepts,
            dtype=gt_concept_logits.dtype
        )
        gt_classifier_inputs = torch.cat([gt_concept_logits, latent_zeros], dim=1)
    else:
        gt_classifier_inputs = gt_concept_logits

    gt_classifier_inputs = model.apply_concept_bias(gt_classifier_inputs.to(device))
    classification_logits = model.classifier_head(gt_classifier_inputs).cpu()
    classification_metrics = calculate_target_metrics(classification_logits, all_gt_targets)
    
    # Compute main concept metrics without confidence thresholding.
    concept_metrics = calculate_concept_metrics(
        biased_concept_logits[:, :model.num_supervised_concepts],
        all_gt_concepts, 
        concept_groups_info=concept_groups_info
    )
    
    # Return raw concept logits to keep logit-space interventions well defined.
    standard_metrics = {
        "concept": {
            "acc": concept_metrics.get("mean_acc", 0.0),
            "macro_f1": concept_metrics.get("mean_f1", 0.0),
            "macro_f2": concept_metrics.get("mean_f_beta", 0.0),
        },
        "classification": classification_metrics,
        "target": target_metrics,
    }
    return standard_metrics, concept_metrics, all_concept_logits, all_gt_concepts, all_gt_targets


def run_tti_group_level(model, concept_logits, gt_concepts, gt_targets, concept_groups, device, budgets):
    """Simulates group-level TTI by correcting attributes group-by-group in logit space."""
    model.eval()
    num_samples = concept_logits.shape[0]
    group_tti_metrics = [(0, calculate_classifier_metrics(model, concept_logits, gt_targets, device))]
    
    # Compute concept probs for sorting erroneous groups
    concept_probs = model.concept_activation(model.apply_concept_bias(concept_logits.to(device))).cpu()
    
    # 1. Compute per-sample prediction error for each group
    sample_group_errors = []
    for i in range(num_samples):
        group_errors = []
        for g_idx, group in enumerate(concept_groups):
            indices = group["flat_indices"]
            pred_slice = concept_probs[i, indices]
            gt_slice = gt_concepts[i, indices]
            mae = torch.mean(torch.abs(pred_slice - gt_slice)).item()
            group_errors.append((g_idx, mae))
        
        # Sort groups for this specific sample in descending order of MAE error
        sorted_groups = [g_idx for g_idx, _ in sorted(group_errors, key=lambda x: x[1], reverse=True)]
        sample_group_errors.append(sorted_groups)
        
    # Translate GT concepts to logit space with soft intervention for mutually exclusive groups
    gt_logits = translate_gt_to_logits(gt_concepts, concept_groups, getattr(model, "use_probabilistic_cbm", False))
    
    # Create a copy of the predicted concept logits that we will mutate
    logits_mutated = concept_logits.clone()
    
    last_k = 0
    pbar = tqdm(budgets[1:], desc="Simulating Group TTI")
    for K in pbar:
        # Correct the top K most erroneous groups for each sample in logit space
        for i in range(num_samples):
            for correction_rank in range(last_k, K):
                g_to_correct = sample_group_errors[i][correction_rank]
                indices = concept_groups[g_to_correct]["flat_indices"]
                logits_mutated[i, indices] = gt_logits[i, indices]
            
        # Predict class targets using the updated concept logits
        updated_metrics = calculate_classifier_metrics(model, logits_mutated, gt_targets, device)
            
        group_tti_metrics.append((K, updated_metrics))
        pbar.set_postfix(acc=f"{updated_metrics['acc'] * 100:.2f}%")
        last_k = K
        
    return group_tti_metrics


def run_tti_concept_level(model, concept_logits, gt_concepts, gt_targets, concept_groups, device, budgets):
    """Simulates individual concept-level TTI in logit space by correcting top-K most erroneous concepts."""
    model.eval()
    num_samples, num_supervised = gt_concepts.shape
    concept_tti_metrics = [(0, calculate_classifier_metrics(model, concept_logits, gt_targets, device))]
    
    # Compute concept probs for sorting erroneous concepts
    concept_probs = model.concept_activation(model.apply_concept_bias(concept_logits.to(device))).cpu()
    
    # Calculate prediction error for each individual concept per sample
    sample_concept_errors = []
    for i in range(num_samples):
        errors = torch.abs(concept_probs[i, :num_supervised] - gt_concepts[i])
        sorted_indices = torch.argsort(errors, descending=True).tolist()
        sample_concept_errors.append(sorted_indices)
        
    # Translate GT concepts to logit space with soft intervention for mutually exclusive groups
    gt_logits = translate_gt_to_logits(gt_concepts, concept_groups, getattr(model, "use_probabilistic_cbm", False))
    
    logits_mutated = concept_logits.clone()
    
    last_k = 0
    pbar = tqdm(budgets[1:], desc="Simulating Concept TTI")
    for K in pbar:
        # Intervene on the next slice of top erroneous concepts in logit space
        for i in range(num_samples):
            indices_to_correct = sample_concept_errors[i][last_k:K]
            logits_mutated[i, indices_to_correct] = gt_logits[i, indices_to_correct]
            
        updated_metrics = calculate_classifier_metrics(model, logits_mutated, gt_targets, device)
            
        concept_tti_metrics.append((K, updated_metrics))
        pbar.set_postfix(acc=f"{updated_metrics['acc'] * 100:.2f}%")
        last_k = K
        
    return concept_tti_metrics


def run_tti_uncertainty_topk(model, concept_logits, gt_concepts, gt_targets, concept_groups, device, budgets):
    """Corrects top-K most uncertain groups, where uncertainty = 1 - max(p_group)."""
    model.eval()
    num_samples = concept_logits.shape[0]

    # Translate GT concepts to logit space with soft intervention for mutually exclusive groups
    gt_logits = translate_gt_to_logits(gt_concepts, concept_groups, getattr(model, "use_probabilistic_cbm", False))

    uncertainty_tti_metrics = [(0, calculate_classifier_metrics(model, concept_logits, gt_targets, device))]

    concept_probs = model.concept_activation(model.apply_concept_bias(concept_logits.to(device))).cpu()
    sample_group_order = []
    for i in range(num_samples):
        group_scores = []
        for g_idx, group in enumerate(concept_groups):
            indices = group["flat_indices"]
            group_probs = concept_probs[i, indices]
            if len(indices) > 1:
                confidence = torch.max(group_probs).item()
            else:
                p = group_probs[0].item()
                confidence = max(p, 1.0 - p)
            uncertainty = 1.0 - confidence
            group_scores.append((g_idx, uncertainty))
        sample_group_order.append([
            g_idx for g_idx, _ in sorted(group_scores, key=lambda item: item[1], reverse=True)
        ])

    logits_mutated = concept_logits.clone()
    last_k = 0
    pbar = tqdm(budgets[1:], desc="Simulating Uncertainty TTI")
    for K in pbar:
        for i in range(num_samples):
            for correction_rank in range(last_k, K):
                g_idx = sample_group_order[i][correction_rank]
                indices = concept_groups[g_idx]["flat_indices"]
                logits_mutated[i, indices] = gt_logits[i, indices]

        updated_metrics = calculate_classifier_metrics(model, logits_mutated, gt_targets, device)
        uncertainty_tti_metrics.append((K, updated_metrics))
        pbar.set_postfix(acc=f"{updated_metrics['acc'] * 100:.2f}%")
        last_k = K

    return uncertainty_tti_metrics


def print_tti_metric_table(title, results):
    """Print a TTI result table with Top-1 target metrics."""
    border = "+------+-----------+-----------+-----------+"
    print(f"\n{title}")
    print(border)
    print("| K    | Accuracy  | Macro-F1  | Macro-F2  |")
    print(border)
    for K, metrics in results:
        print(
            f"| {K:<4} | "
            f"{metrics['acc']*100:>8.2f}% | "
            f"{metrics['macro_f1']*100:>8.2f}% | "
            f"{metrics['macro_f2']*100:>8.2f}% |"
        )
    print(border)


def get_tti_metrics_at_k(results, target_k=1):
    result_by_k = dict(results)
    if target_k in result_by_k:
        return target_k, result_by_k[target_k]
    return results[-1]


def print_tti_metric_summary(group_results, concept_results, uncertainty_results, coop_results=None):
    baseline = group_results[0][1]
    group_k, group_summary = get_tti_metrics_at_k(group_results, target_k=1)
    concept_k, concept_summary = get_tti_metrics_at_k(concept_results, target_k=1)
    uncertainty_k, uncertainty_summary = get_tti_metrics_at_k(uncertainty_results, target_k=1)

    border = "+----------+--------------+--------------+--------------+----------------+--------------+--------------+--------------+"
    print(f"\n{BOLD}{GREEN}============================================================{RESET}")
    print(f"  {BOLD}{GREEN}[Success] TTI Benchmark Evaluation Complete!{RESET}")
    print(f"{BOLD}{GREEN}============================================================{RESET}")
    print(border)
    print(
        f"| Metric   | Standard K=0 | Group K={group_k:<4} | "
        f"Concept K={concept_k:<2} | Uncertainty K={uncertainty_k:<2} | "
        "Group Delta | Concept Delta | Uncert Delta |"
    )
    print(border)
    for key, label in TTI_METRICS:
        standard = baseline[key] * 100
        group_value = group_summary[key] * 100
        concept_value = concept_summary[key] * 100
        uncertainty_value = uncertainty_summary[key] * 100
        print(
            f"| {label:<8} | "
            f"{standard:>11.2f}% | "
            f"{group_value:>11.2f}% | "
            f"{concept_value:>11.2f}% | "
            f"{uncertainty_value:>13.2f}% | "
            f"{group_value - standard:>11.2f}% | "
            f"{concept_value - standard:>12.2f}% | "
            f"{uncertainty_value - standard:>11.2f}% |"
        )
    print(border)
    if coop_results is not None:
        coop_k, coop_summary = get_tti_metrics_at_k(coop_results, target_k=5)
        print(f"\n{BOLD}{GREEN}[CooP Summary]{RESET} Group policy at K={coop_k}")
        for key, label in TTI_METRICS:
            standard = baseline[key] * 100
            coop_value = coop_summary[key] * 100
            print(f"  [TTI] CooP Group Policy {label}: {coop_value:.2f}% ({coop_value - standard:+.2f}%p)")


def run_tti_benchmark(
    model,
    concept_logits,
    gt_concepts,
    gt_targets,
    concept_groups,
    device,
    coop_alpha=1.0,
    coop_beta=1.0,
    coop_gamma=0.0,
    coop_costs=None,
    coop_influence_mode='abs_change',
    coop_candidate_batch_size=16384,
    without_coop_tti=False,
    coop_fit_result=None,
):
    """Run all TTI variants and print Top-1 target metric tables."""
    group_budgets = make_tti_budgets(len(concept_groups))
    concept_budgets = group_budgets

    tqdm.write(f"\n{BOLD}{BLUE}============================================================{RESET}")
    tqdm.write(f"  {BOLD}{BLUE}[TTI - Group Level]{RESET} Top-1 target metrics")
    tqdm.write(f"  Correcting groups by concept prediction error; K={group_budgets[1:]}")
    tqdm.write(f"{BOLD}{BLUE}============================================================{RESET}")
    group_tti_results = run_tti_group_level(
        model,
        concept_logits,
        gt_concepts,
        gt_targets,
        concept_groups,
        device,
        group_budgets
    )
    print_tti_metric_table("[TTI - Group Level]", group_tti_results)

    coop_tti_results = None
    if not without_coop_tti:
        tqdm.write(f"\n{BOLD}{BLUE}============================================================{RESET}")
        tqdm.write(f"  {BOLD}{BLUE}[TTI - CooP Group Level]{RESET} Top-1 target metrics")
        tqdm.write(
            f"  score={coop_alpha:.2f}*CPU + {coop_beta:.2f}*CIS - "
            f"{coop_gamma:.2f}*cost | influence={coop_influence_mode}; K={group_budgets[1:]}"
        )
        if coop_fit_result is not None:
            tqdm.write(
                f"  fitted on validation K={coop_fit_result.budget}: "
                f"alpha={coop_fit_result.alpha:.4g}, gamma={coop_fit_result.gamma:.4g}, "
                f"{coop_fit_result.metric_name}={coop_fit_result.metric_value*100:.2f}%"
            )
        tqdm.write(f"{BOLD}{BLUE}============================================================{RESET}")
        if isinstance(coop_costs, torch.Tensor):
            coop_cost_tensor = coop_costs
        else:
            coop_cost_tensor = parse_coop_costs(coop_costs, concept_groups)
        coop_tti_results, coop_query_stats = run_tti_coop_group_level(
            model,
            concept_logits,
            gt_concepts,
            gt_targets,
            concept_groups,
            device,
            group_budgets,
            metric_fn=calculate_classifier_metrics,
            alpha=coop_alpha,
            beta=coop_beta,
            gamma=coop_gamma,
            costs=coop_cost_tensor,
            influence_mode=coop_influence_mode,
            candidate_batch_size=coop_candidate_batch_size,
        )
        print_tti_metric_table("[TTI - CooP Group Level]", coop_tti_results)

        first_counts = coop_query_stats["first"]
        top_first = torch.topk(first_counts, k=min(5, len(concept_groups)))
        print("\n  [CooP] Most frequently selected first-query groups:")
        for count, group_idx in zip(top_first.values.tolist(), top_first.indices.tolist()):
            print(f"     - {concept_groups[group_idx]['name']}: {count} / {len(gt_targets)} samples")

    tqdm.write(f"\n{BOLD}{BLUE}============================================================{RESET}")
    tqdm.write(f"  {BOLD}{BLUE}[TTI - Concept Level]{RESET} Top-1 target metrics")
    tqdm.write(f"  Correcting individual concepts by prediction error; K={concept_budgets[1:]}")
    tqdm.write(f"{BOLD}{BLUE}============================================================{RESET}")
    concept_tti_results = run_tti_concept_level(
        model,
        concept_logits,
        gt_concepts,
        gt_targets,
        concept_groups,
        device,
        concept_budgets
    )
    print_tti_metric_table("[TTI - Concept Level]", concept_tti_results)

    print(f"\n{BOLD}{YELLOW}============================================================{RESET}")
    print(f"  {BOLD}{YELLOW}[Uncertainty Top-K TTI]{RESET} Top-1 target metrics")
    print(f"  Correcting groups by uncertainty = 1 - max(p_group); K={group_budgets[1:]}")
    print(f"{BOLD}{YELLOW}============================================================{RESET}")
    uncertainty_tti_results = run_tti_uncertainty_topk(
        model,
        concept_logits,
        gt_concepts,
        gt_targets,
        concept_groups,
        device,
        group_budgets
    )
    print_tti_metric_table("[TTI - Uncertainty Group Level]", uncertainty_tti_results)

    print_tti_metric_summary(group_tti_results, concept_tti_results, uncertainty_tti_results, coop_tti_results)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    tqdm.write(f"\n{BOLD}{CYAN}============================================================{RESET}")
    tqdm.write(f"  {BOLD}{CYAN}[Checkpoint]{RESET} Loading CBM Checkpoint: {args.checkpoint}")
    tqdm.write(f"{BOLD}{CYAN}============================================================{RESET}")
    
    # 1. Load checkpoint meta to auto-detect model configs
    loaded = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_args = loaded.get('args', {})
    checkpoint_config = loaded.get('config', {})
    state_dict = loaded.get('state_dict', loaded)
    
    # Auto-detect configs with fallbacks to config dictionaries and state_dict keys
    dataset_name = checkpoint_args.get('dataset') or checkpoint_config.get('dataset', {}).get('name') or 'cub'
    backbone_type = checkpoint_args.get('backbone_type') or checkpoint_config.get('backbone', {}).get('backbone_type') or 'timm'
    backbone_name = checkpoint_args.get('backbone_name') or checkpoint_config.get('backbone', {}).get('backbone_name') or 'vit_base_patch14_dinov2'
    backbone_train_mode = resolve_backbone_train_mode(checkpoint_args, checkpoint_config, state_dict)
    use_lora = (backbone_train_mode == "lora")
    lora_r = checkpoint_args.get('lora_r') or checkpoint_config.get('backbone', {}).get('lora_r') or 8
    lora_alpha = checkpoint_args.get('lora_alpha') or checkpoint_config.get('backbone', {}).get('lora_alpha') or 16.0
    
    use_group_broadcasting = checkpoint_args.get('use_group_broadcasting') or checkpoint_config.get('backbone', {}).get('use_group_broadcasting') or False
    if not use_group_broadcasting:
        has_group_queries = any("supervised_attention.group_queries" in k for k in state_dict.keys())
        if has_group_queries:
            use_group_broadcasting = True

    use_cosine_attention = checkpoint_args.get('use_cosine_attention') or checkpoint_config.get('backbone', {}).get('use_cosine_attention') or False
    if not use_cosine_attention:
        has_cosine_keys = any(".q_proj.weight" in k and "supervised_attention" in k for k in state_dict.keys())
        has_mha_keys    = any(".cross_attention." in k for k in state_dict.keys())
        if has_cosine_keys and not has_mha_keys:
            use_cosine_attention = True

    latent_concepts = checkpoint_args.get('latent_concepts') or checkpoint_config.get('training', {}).get('latent_concepts') or 0
    num_classes = checkpoint_args.get('num_classes') or checkpoint_config.get('training', {}).get('num_classes') or 200
    filter_rare_concepts = checkpoint_args.get('filter_rare_concepts', False)
    if not filter_rare_concepts:
        filter_rare_concepts = checkpoint_config.get('dataset', {}).get('filter_rare_concepts', False)
    use_paper_preprocessing = checkpoint_args.get('use_paper_preprocessing', False)
    if not use_paper_preprocessing:
        use_paper_preprocessing = checkpoint_config.get('dataset', {}).get('use_paper_preprocessing', False)
        
    use_nam_head = False
    nam_hidden_dim = 64

    # Check if NAM head is used based on state_dict keys
    if "classifier_head.concept_gates" in state_dict:
        use_nam_head = True
        # GatedSparseNAMHead uses conv1 grouped conv.
        if "classifier_head.conv1.weight" in state_dict:
            num_supervised_gates = state_dict["classifier_head.concept_gates"].shape[0]
            out_ch = state_dict["classifier_head.conv1.weight"].shape[0]
            nam_hidden_dim = out_ch // num_supervised_gates
            
        # Detect latent concepts in NAM: check if latent_linear layer weights exist
        if "classifier_head.latent_linear.weight" in state_dict:
            latent_concepts = state_dict["classifier_head.latent_linear.weight"].shape[1]
        else:
            latent_concepts = 0
    else:
        # Fallback to args/configs if they exist in the checkpoint
        use_nam_head = checkpoint_args.get('use_nam_head') or checkpoint_args.get('use_gated_nam') or checkpoint_config.get('training', {}).get('use_nam_head') or checkpoint_config.get('training', {}).get('use_gated_nam') or False
        nam_hidden_dim = checkpoint_args.get('nam_hidden_dim') or checkpoint_config.get('training', {}).get('nam_hidden_dim') or 64

    use_probabilistic_cbm = False
    if 'use_probabilistic_cbm' in checkpoint_args:
        use_probabilistic_cbm = checkpoint_args['use_probabilistic_cbm']
    elif 'use_probabilistic_cbm' in checkpoint_config.get('training', {}):
        use_probabilistic_cbm = checkpoint_config['training']['use_probabilistic_cbm']
    elif any(k in state_dict for k in [
        'supervised_attention.concept_weight_logvar',
        'supervised_attention.concept_bias_logvar',
        'supervised_attention.concept_proj_logvar',
        'supervised_attention.concept_bias_logvar',
        'supervised_attention.mlp_logvar.0.weight'
    ]):
        use_probabilistic_cbm = True

    use_pairwise_nam = False
    if 'use_pairwise_nam' in checkpoint_args:
        use_pairwise_nam = checkpoint_args['use_pairwise_nam']
    elif 'use_pairwise_nam' in checkpoint_config.get('training', {}):
        use_pairwise_nam = checkpoint_config['training']['use_pairwise_nam']
    elif 'classifier_head.pairwise_gates' in state_dict:
        use_pairwise_nam = True

    use_concept_attention = False
    if 'use_concept_attention' in checkpoint_args:
        use_concept_attention = checkpoint_args['use_concept_attention']
    elif 'use_concept_attention' in checkpoint_config.get('backbone', {}):
        use_concept_attention = checkpoint_config['backbone']['use_concept_attention']
    elif 'supervised_attention.concept_queries' in state_dict:
        use_concept_attention = True

    tqdm.write(f"  {BOLD}{BLUE}[Config]{RESET} Auto-detected config:")
    tqdm.write(f"     ├─ Dataset: {dataset_name.upper()}")
    tqdm.write(f"     ├─ Backbone: {backbone_name} ({backbone_type})")
    tqdm.write(f"     ├─ backbone_train_mode: {backbone_train_mode}")
    tqdm.write(f"     ├─ use_lora: {use_lora} (r={lora_r}, alpha={lora_alpha})")
    tqdm.write(f"     ├─ latent_concepts: {latent_concepts}")
    tqdm.write(f"     ├─ use_group_broadcasting: {use_group_broadcasting}")
    tqdm.write(f"     ├─ use_cosine_attention: {use_cosine_attention}")
    tqdm.write(f"     ├─ use_concept_attention: {use_concept_attention}")
    tqdm.write(f"     ├─ use_probabilistic_cbm: {use_probabilistic_cbm}")
    tqdm.write(f"     ├─ filter_rare_concepts: {filter_rare_concepts}")
    tqdm.write(f"     ├─ use_paper_preprocessing: {use_paper_preprocessing}")
    tqdm.write(f"     ├─ use_nam_head: {use_nam_head}")
    tqdm.write(f"     ├─ use_pairwise_nam: {use_pairwise_nam}")
    tqdm.write(f"     └─ nam_hidden_dim: {nam_hidden_dim}")
    
    # 2. Build Datasets and Loaders dynamically based on discovered configs
    if dataset_name == 'derm7pt':
        dataset_class = Derm7PtDataset
        csv_path = checkpoint_args.get('csv_path', 'data/derm7pt/meta/meta.csv')
        image_dir = checkpoint_args.get('image_dir', 'data/derm7pt/images')
        concept_config_path = checkpoint_args.get('concept_config_path', 'data/derm7pt/concept_config.json')
    elif dataset_name == 'milk10k':
        dataset_class = MILK10KDataset
        csv_path = checkpoint_args.get('csv_path', 'data/MILK10K/MILK10k_Training_Metadata.csv')
        image_dir = checkpoint_args.get('image_dir', 'data/MILK10K/MILK10k_Training_Input/MILK10k_Training_Input')
        concept_config_path = checkpoint_args.get('concept_config_path', 'data/MILK10K/concept_config.json')
    elif dataset_name == 'chexpert':
        dataset_class = CheXpertDataset
        csv_path = checkpoint_args.get('csv_path', 'data/CheXpert/train.csv')
        image_dir = checkpoint_args.get('image_dir', 'data/CheXpert/')
        concept_config_path = checkpoint_args.get('concept_config_path', 'data/CheXpert/concept_config.json')
    else:
        dataset_class = CUB2011Dataset
        csv_path = checkpoint_args.get('csv_path', 'data/CUB_200_2011/images.txt')
        image_dir = checkpoint_args.get('image_dir', 'data/CUB_200_2011/images')
        concept_config_path = checkpoint_args.get('concept_config_path', 'data/CUB_200_2011/concept_config.json')
        if filter_rare_concepts or use_paper_preprocessing:
            filtered_path = concept_config_path.replace(".json", "_filtered.json")
            if os.path.exists(filtered_path):
                concept_config_path = filtered_path
                tqdm.write(f"     [Config] Redirected concept_config to: {concept_config_path}")
        
    dataset_config = dataset_class.get_default_config()
    dataset_config["concept_config_path"] = concept_config_path
    dataset_config["filter_rare_concepts"] = filter_rare_concepts
    dataset_config["use_paper_preprocessing"] = use_paper_preprocessing
    should_fit_coop = not args.without_tti and not args.without_coop_tti and not args.without_coop_fit
    
    # Load test split
    test_dataset = dataset_class(
        csv_path=csv_path,
        image_dir=image_dir,
        split='test',
        config=dataset_config,
        cache_in_memory=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = None
    if should_fit_coop:
        val_dataset = dataset_class(
            csv_path=csv_path,
            image_dir=image_dir,
            split='val',
            config=dataset_config,
            cache_in_memory=True
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Resolve exact dimensions
    num_supervised_concepts = test_dataset.config["num_concepts"]
    num_concepts_total = num_supervised_concepts + latent_concepts
    num_classes = test_dataset.config["num_classes"]
    
    # 3. Parse concept groups config for TTI
    with open(concept_config_path, 'r', encoding='utf-8') as f:
        concept_json = json.load(f)
        
    concept_groups = []
    total_dims = 0
    for name, info in concept_json.items():
        ctype = info.get("type", "numerical")
        if ctype == "categorical":
            classes = info.get("classes", [])
            num_feats = len(classes)
            group = {
                "name": name,
                "flat_indices": list(range(total_dims, total_dims + num_feats))
            }
            total_dims += num_feats
        else:
            group = {
                "name": name,
                "flat_indices": [total_dims]
            }
            total_dims += 1
        concept_groups.append(group)
        
    # Build concept_groups_info representation for the metrics calculation
    concept_groups_info = []
    for g in concept_groups:
        concept_groups_info.append((g["flat_indices"][0], len(g["flat_indices"])))
        
    # Build group_mapping for GroupToConceptAttention
    group_mapping = None
    num_groups = len(concept_groups)
    if use_group_broadcasting:
        group_mapping = []
        for group_idx, g in enumerate(concept_groups):
            num_in_group = len(g["flat_indices"])
            group_mapping.extend([group_idx] * num_in_group)
        concept_groups_info_param = None
    else:
        concept_groups_info_param = concept_groups_info
        
    # 4. Instantiate Model
    model = UniversalFlexibleCBM(
        backbone_type=backbone_type,
        backbone_name=backbone_name,
        num_supervised_concepts=num_supervised_concepts,
        num_classes=num_classes,
        num_latent_concepts=latent_concepts,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        concept_groups_info=concept_groups_info_param,
        use_cosine_attention=use_cosine_attention,
        use_group_broadcasting=use_group_broadcasting,
        num_groups=num_groups,
        group_mapping=group_mapping,
        # use_dino_mask and dino_mask_threshold parameters removed
        use_nam_head=use_nam_head,
        nam_hidden_dim=nam_hidden_dim,
        use_probabilistic_cbm=use_probabilistic_cbm,
        use_concept_attention=use_concept_attention,
        use_pairwise_nam=use_pairwise_nam
    )
    
    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    if args.ignore_bias and hasattr(model, "concept_bias"):
        nonzero_bias = int((model.concept_bias.detach().abs() > 0).sum().item())
        model.concept_bias.zero_()
        tqdm.write(
            f"  {BOLD}{YELLOW}[Concept Bias]{RESET} "
            f"Ignoring checkpoint concept_bias for evaluation (zeroed {nonzero_bias} non-zero entries)."
        )
    
    tqdm.write(f"\n{BOLD}{MAGENTA}============================================================{RESET}")
    tqdm.write(f"  {BOLD}{MAGENTA}[Evaluation]{RESET} Running Test Set Evaluation...")
    tqdm.write(f"{BOLD}{MAGENTA}============================================================{RESET}")
    
    # 5. Run standard evaluation
    standard_metrics, concept_metrics, concept_logits, gt_concepts, gt_targets = run_evaluation(
        model, 
        test_loader, 
        concept_groups_info if not use_group_broadcasting else None, 
        concept_groups,
        device,
        desc="Evaluating CBM (test)"
    )
    
    tqdm.write(f"\n{BOLD}{GREEN}[Performance] Standard CBM Test Performance:{RESET}")
    perf_header = "   | Task                         | Acc       | Macro-F1  | Macro-F2  |"
    perf_border = "   +------------------------------+-----------+-----------+-----------+"
    tqdm.write(perf_border)
    tqdm.write(perf_header)
    tqdm.write(perf_border)
    for task_key, task_name in (
        ("concept", "Concept"),
        ("classification", "Classification (GT Concept)"),
        ("target", "Target"),
    ):
        metrics = standard_metrics[task_key]
        tqdm.write(
            f"   | {task_name:<28} | "
            f"{metrics['acc']*100:>8.2f}% | "
            f"{metrics['macro_f1']*100:>8.2f}% | "
            f"{metrics['macro_f2']*100:>8.2f}% |"
        )
    tqdm.write(perf_border)
    tqdm.write(f"\n{BOLD}{GREEN}[Concept Diagnostics]{RESET}")
    tqdm.write(f"   Concept Mean True Positive Rate: {concept_metrics['tpr']*100:.2f}%")
    tqdm.write(f"   Concept Mean True Negative Rate: {concept_metrics['tnr']*100:.2f}%")
    tqdm.write(f"   Concept Mean Balanced Accuracy : {concept_metrics['mean_balanced_acc']*100:.2f}%")
    
    coop_alpha = args.coop_alpha
    coop_gamma = args.coop_gamma
    coop_cost_tensor = None
    coop_fit_result = None

    if not args.without_tti and not args.without_coop_tti:
        coop_cost_tensor = parse_coop_costs(args.coop_costs, concept_groups)
        if should_fit_coop:
            tqdm.write(f"\n{BOLD}{MAGENTA}============================================================{RESET}")
            tqdm.write(f"  {BOLD}{MAGENTA}[CooP Fit]{RESET} Running Validation Set Evaluation...")
            tqdm.write(f"{BOLD}{MAGENTA}============================================================{RESET}")
            _, _, val_concept_logits, val_gt_concepts, val_gt_targets = run_evaluation(
                model,
                val_loader,
                concept_groups_info if not use_group_broadcasting else None,
                concept_groups,
                device,
                desc="Evaluating CBM (val)"
            )

            alpha_grid = parse_float_grid(args.coop_alpha_grid)
            gamma_grid = parse_float_grid(args.coop_gamma_grid)
            coop_fit_budget = min(max(1, args.coop_fit_k), len(concept_groups))
            tqdm.write(
                f"  {BOLD}{BLUE}[CooP Fit]{RESET} Grid search "
                f"alpha={alpha_grid}, gamma={gamma_grid}, K={coop_fit_budget}, "
                f"metric={args.coop_fit_metric}"
            )
            coop_fit_result = fit_alpha_gamma(
                model=model,
                concept_logits=val_concept_logits,
                gt_concepts=val_gt_concepts,
                gt_targets=val_gt_targets,
                concept_groups=concept_groups,
                device=device,
                alpha_grid=alpha_grid,
                gamma_grid=gamma_grid,
                fit_budget=coop_fit_budget,
                metric_fn=calculate_classifier_metrics,
                metric_name=args.coop_fit_metric,
                beta=args.coop_beta,
                costs=coop_cost_tensor,
                influence_mode=args.coop_influence_mode,
                candidate_batch_size=args.coop_candidate_batch_size,
            )
            coop_alpha = coop_fit_result.alpha
            coop_gamma = coop_fit_result.gamma
            tqdm.write(
                f"  {BOLD}{GREEN}[CooP Fit]{RESET} Selected "
                f"alpha={coop_alpha:.4g}, gamma={coop_gamma:.4g} "
                f"with val {coop_fit_result.metric_name}={coop_fit_result.metric_value*100:.2f}%"
            )

    if args.without_tti:
        tqdm.write(f"\n{BOLD}{YELLOW}[TTI]{RESET} Skipped because --without-tti was set.")
    else:
        run_tti_benchmark(
            model,
            concept_logits,
            gt_concepts,
            gt_targets,
            concept_groups,
            device,
            coop_alpha=coop_alpha,
            coop_beta=args.coop_beta,
            coop_gamma=coop_gamma,
            coop_costs=coop_cost_tensor,
            coop_influence_mode=args.coop_influence_mode,
            coop_candidate_batch_size=args.coop_candidate_batch_size,
            without_coop_tti=args.without_coop_tti,
            coop_fit_result=coop_fit_result,
        )

    # 6. Export Active Pairwise NAM Interactions
    if getattr(model, "use_pairwise_nam", False):
        tqdm.write(f"\n{BOLD}{CYAN}[Interaction Logging]{RESET} Analyzing pairwise concept interactions...")
        gates = model.classifier_head.pairwise_gates.detach().cpu().numpy()
        pair_indices = model.classifier_head.pair_indices
        concepts_list = test_dataset.config.get("concepts_flat", test_dataset.config.get("concepts", []))
        
        active_interactions = []
        threshold = 1e-3
        for idx, g_val in enumerate(gates):
            if abs(g_val) > threshold:
                i, j = pair_indices[idx]
                name_i = concepts_list[i] if i < len(concepts_list) else f"Concept_{i}"
                name_j = concepts_list[j] if j < len(concepts_list) else f"Concept_{j}"
                active_interactions.append({
                    "index": idx,
                    "concept_i_idx": i,
                    "concept_j_idx": j,
                    "concept_i": name_i,
                    "concept_j": name_j,
                    "gate_weight": float(g_val)
                })
        
        # Sort by absolute gate weight descending
        active_interactions = sorted(active_interactions, key=lambda x: abs(x["gate_weight"]), reverse=True)
        
        export_path = "active_interactions.json"
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(active_interactions, f, indent=4)
        tqdm.write(f"  {BOLD}{GREEN}[Export]{RESET} Exported {len(active_interactions)} active interaction pairs to {export_path}")
        
        tqdm.write(f"  Top 10 Active Concept Interaction Pairs:")
        for item in active_interactions[:10]:
            tqdm.write(f"     ├─ {item['concept_i']} <--> {item['concept_j']}: weight = {item['gate_weight']:.4f}")


if __name__ == "__main__":
    main()
