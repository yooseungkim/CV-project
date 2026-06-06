import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from src.data.cub import CUB2011Dataset
from src.data.derm7pt import Derm7PtDataset
from src.data.milk10k import MILK10KDataset
from src.data.chexpert import CheXpertDataset
from src.models.cbm_factory import UniversalFlexibleCBM
from src.utils.metrics import calculate_accuracy, calculate_concept_metrics

# ANSI terminal colors for highlighting
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

def parse_args():
    parser = argparse.ArgumentParser(description="Concept Bottleneck Model Evaluation & Test-Time Intervention (TTI) Benchmark")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to saved CBM model checkpoint (.pt or .pth)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for testing")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Computation device")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loader")
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'validation'], help="Dataset split to evaluate")
    return parser.parse_args()


def calculate_topk_accuracy(outputs, targets, topk=(1, 3, 5, 10)):
    """Helper to calculate Top-K accuracy for target classes."""
    if outputs.dim() > 1 and targets.dim() > 1 and outputs.shape[-1] == targets.shape[-1]:
        # Multi-label classification (like CheXpert)
        preds = (outputs > 0.0).float()
        correct = (preds == targets).float().sum().item()
        return {1: correct / targets.numel()}
        
    maxk = min(outputs.shape[-1], max(topk))
    batch_size = targets.size(0)
    if outputs.shape[-1] <= 1:
        preds = (outputs > 0.0).float()
        correct = (preds == targets.view_as(preds)).float().sum().item()
        return {1: correct / batch_size}
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



@torch.no_grad()
def run_evaluation(model, dataloader, concept_groups_info, device):
    """Runs a standard evaluation pass over the dataloader."""
    model.eval()
    
    all_class_logits = []
    all_concept_logits = []
    all_gt_concepts = []
    all_gt_targets = []
    all_tabular_features = []
    
    pbar = tqdm(dataloader, desc="Evaluating CBM")
    for images, concepts, targets, tabular_features in pbar:
        images = images.to(device)
        concepts = concepts.to(device)
        targets = targets.to(device)
        tabular_features = tabular_features.to(device)
        
        # Forward pass
        class_logits, concept_logits, _ = model(images, tabular_features=tabular_features)
        
        all_class_logits.append(class_logits.cpu())
        all_concept_logits.append(concept_logits.cpu())
        all_gt_concepts.append(concepts.cpu())
        all_gt_targets.append(targets.cpu())
        all_tabular_features.append(tabular_features.cpu())
        
    # Concatenate all test outputs
    all_class_logits = torch.cat(all_class_logits, dim=0)
    all_concept_logits = torch.cat(all_concept_logits, dim=0)
    all_gt_concepts = torch.cat(all_gt_concepts, dim=0)
    all_gt_targets = torch.cat(all_gt_targets, dim=0)
    all_tabular_features = torch.cat(all_tabular_features, dim=0)
    
    # Calculate concept probabilities through model's registered activation
    all_concept_probs = model.concept_activation(all_concept_logits.to(device)).cpu()
    
    # Compute Target Classification Accuracy
    topk_accs = calculate_topk_accuracy(all_class_logits, all_gt_targets)
    
    # Compute Concept Metrics (Balanced Acc, TPR, TNR) using model's optimized validation thresholds
    concept_metrics = calculate_concept_metrics(
        all_concept_logits[:, :model.num_supervised_concepts], 
        all_gt_concepts, 
        concept_groups_info=concept_groups_info,
        threshold=model.concept_thresholds.cpu()
    )
    
    # Return all_concept_logits instead of all_concept_probs to enable Inverse Sigmoid Intervention
    return topk_accs, concept_metrics, all_concept_logits, all_gt_concepts, all_gt_targets, all_tabular_features


def translate_gt_to_logits(gt_concepts, concept_groups, use_probabilistic):
    """
    Vectorized translation of ground truth probabilities to logit space with soft intervention.
    For mutually exclusive categorical groups, if one class is positive (value=1.0),
    we perform a soft intervention:
    - The positive class gets probability p_pos (0.999 for prob, 0.95 for non-prob).
    - The other classes in the group split the remaining probability (1 - p_pos)
      equally so that the group sum is 1.0, making their probabilities close to 0.
    For numerical concepts, we just use standard clipping.
    """
    p_pos = 0.999 if use_probabilistic else 0.95
    p_neg_default = 0.001 if use_probabilistic else 0.05
    
    p_custom = gt_concepts.clone()
    
    for group in concept_groups:
        indices = group["flat_indices"]
        M = len(indices)
        if M > 1:
            group_gt = gt_concepts[:, indices]  # [N, M]
            # Find the argmax and max values for each sample
            max_vals, correct_idxs = torch.max(group_gt, dim=1)  # [N], [N]
            
            # Create a tensor of shape [N, M] filled with p_others
            p_others = (1.0 - p_pos) / (M - 1)
            group_custom = torch.full_like(group_gt, p_others)
            
            # Set the correct class to p_pos
            group_custom.scatter_(1, correct_idxs.unsqueeze(1), p_pos)
            
            # For samples where max_val <= 0.5 (no positive class), fallback to clamped original GT
            is_positive = (max_vals > 0.5).unsqueeze(1)  # [N, 1]
            group_fallback = torch.clamp(group_gt, min=p_neg_default, max=p_pos)
            
            # Combine based on whether a positive class exists
            p_custom[:, indices] = torch.where(is_positive, group_custom, group_fallback)
        else:
            # Numerical concept
            c_idx = indices[0]
            p_custom[:, c_idx] = torch.clamp(gt_concepts[:, c_idx], min=p_neg_default, max=p_pos)
            
    p_custom = torch.clamp(p_custom, min=1e-6, max=1.0 - 1e-6)
    gt_logits = torch.log(p_custom / (1.0 - p_custom))
    return gt_logits


def run_tti_group_level(model, concept_logits, gt_concepts, gt_targets, concept_groups, latent_concepts, device, all_tabular_features=None):
    """Simulates group-level TTI by correcting attributes group-by-group in logit space."""
    model.eval()
    num_samples = concept_logits.shape[0]
    num_groups = len(concept_groups)
    
    # Pre-allocate array to store accuracy at each step
    group_tti_accuracies = []
    
    # Calculate initial predictions (K=0 interventions) using predicted logits directly
    if getattr(model, "use_multimodal", False) and all_tabular_features is not None:
        inputs = torch.cat([concept_logits.to(device), all_tabular_features.to(device)], dim=-1)
        init_logits = model.classifier_head(inputs)
    elif getattr(model, "age_sex_skip_connection", False) and all_tabular_features is not None:
        init_logits = model.classifier_head(concept_logits.to(device)) + model.tabular_skip_head(all_tabular_features.to(device))
    else:
        inputs = concept_logits.to(device)
        init_logits = model.classifier_head(inputs)
    init_topk = calculate_topk_accuracy(init_logits.cpu(), gt_targets)
    group_tti_accuracies.append((0, init_topk))
    
    # Compute concept probs for sorting erroneous groups
    concept_probs = model.concept_activation(concept_logits.to(device)).cpu()
    
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
    
    pbar = tqdm(range(1, num_groups + 1), desc="Simulating Group TTI")
    for K in pbar:
        # Correct the top K most erroneous groups for each sample in logit space
        for i in range(num_samples):
            g_to_correct = sample_group_errors[i][K - 1]
            indices = concept_groups[g_to_correct]["flat_indices"]
            # Overwrite with translated GT logits
            logits_mutated[i, indices] = gt_logits[i, indices]
            
        # Predict class targets using the updated concept logits
        with torch.no_grad():
            if getattr(model, "use_multimodal", False) and all_tabular_features is not None:
                inputs_mutated = torch.cat([logits_mutated.to(device), all_tabular_features.to(device)], dim=-1)
                updated_logits = model.classifier_head(inputs_mutated)
            elif getattr(model, "age_sex_skip_connection", False) and all_tabular_features is not None:
                updated_logits = model.classifier_head(logits_mutated.to(device)) + model.tabular_skip_head(all_tabular_features.to(device))
            else:
                inputs_mutated = logits_mutated.to(device)
                updated_logits = model.classifier_head(inputs_mutated)
            updated_topk = calculate_topk_accuracy(updated_logits.cpu(), gt_targets)
            
        group_tti_accuracies.append((K, updated_topk))
        pbar.set_postfix(acc=f"{updated_topk.get(1, 0.0) * 100:.2f}%")
        
    return group_tti_accuracies


def run_tti_concept_level(model, concept_logits, gt_concepts, gt_targets, concept_groups, latent_concepts, device, all_tabular_features=None):
    """Simulates individual concept-level TTI in logit space by correcting top-K most erroneous concepts."""
    model.eval()
    num_samples, num_supervised = gt_concepts.shape
    
    concept_tti_accuracies = []
    
    # Init (K=0)
    if getattr(model, "use_multimodal", False) and all_tabular_features is not None:
        inputs = torch.cat([concept_logits.to(device), all_tabular_features.to(device)], dim=-1)
        init_logits = model.classifier_head(inputs)
    elif getattr(model, "age_sex_skip_connection", False) and all_tabular_features is not None:
        init_logits = model.classifier_head(concept_logits.to(device)) + model.tabular_skip_head(all_tabular_features.to(device))
    else:
        inputs = concept_logits.to(device)
        init_logits = model.classifier_head(inputs)
    init_topk = calculate_topk_accuracy(init_logits.cpu(), gt_targets)
    concept_tti_accuracies.append((0, init_topk))
    
    # Compute concept probs for sorting erroneous concepts
    concept_probs = model.concept_activation(concept_logits.to(device)).cpu()
    
    # Calculate prediction error for each individual concept per sample
    sample_concept_errors = []
    for i in range(num_samples):
        errors = torch.abs(concept_probs[i, :num_supervised] - gt_concepts[i])
        sorted_indices = torch.argsort(errors, descending=True).tolist()
        sample_concept_errors.append(sorted_indices)
        
    # Translate GT concepts to logit space with soft intervention for mutually exclusive groups
    gt_logits = translate_gt_to_logits(gt_concepts, concept_groups, getattr(model, "use_probabilistic_cbm", False))
    
    # Evaluate at specific intervention percentages (0%, 10%, 20%, ..., 100% of concepts)
    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    logits_mutated = concept_logits.clone()
    
    last_k = 0
    for pct in percentages:
        target_k = int((pct / 100.0) * num_supervised)
        
        # Intervene on the next slice of top erroneous concepts in logit space
        for i in range(num_samples):
            indices_to_correct = sample_concept_errors[i][last_k:target_k]
            logits_mutated[i, indices_to_correct] = gt_logits[i, indices_to_correct]
            
        with torch.no_grad():
            if getattr(model, "use_multimodal", False) and all_tabular_features is not None:
                inputs_mutated = torch.cat([logits_mutated.to(device), all_tabular_features.to(device)], dim=-1)
                updated_logits = model.classifier_head(inputs_mutated)
            elif getattr(model, "age_sex_skip_connection", False) and all_tabular_features is not None:
                updated_logits = model.classifier_head(logits_mutated.to(device)) + model.tabular_skip_head(all_tabular_features.to(device))
            else:
                inputs_mutated = logits_mutated.to(device)
                updated_logits = model.classifier_head(inputs_mutated)
            updated_topk = calculate_topk_accuracy(updated_logits.cpu(), gt_targets)
            
        concept_tti_accuracies.append((pct, updated_topk))
        last_k = target_k
        
    return concept_tti_accuracies


def run_tti_unconfident_only(model, concept_logits, gt_concepts, gt_targets, concept_groups, latent_concepts, device, logit_margin=0.0, all_tabular_features=None):
    """Simulates TTI by correcting concept groups predicted as "Not Visible / Occluded" or within a logit margin of optimal dynamic thresholds."""
    model.eval()
    num_samples = concept_logits.shape[0]
    num_groups = len(concept_groups)
    
    # Get optimal validation thresholds in logit space
    thresh_tensor = model.concept_thresholds.cpu()
    
    # Translate GT concepts to logit space with soft intervention for mutually exclusive groups
    gt_logits = translate_gt_to_logits(gt_concepts, concept_groups, getattr(model, "use_probabilistic_cbm", False))
    
    # Create a copy of predicted concept logits
    logits_mutated = concept_logits.clone()
    
    corrected_counts = []
    for i in range(num_samples):
        corrected_count = 0
        for group in concept_groups:
            indices = group["flat_indices"]
            # Extract predicted logits for this group
            group_logits = concept_logits[i, indices]
            max_logit = torch.max(group_logits).item()
            
            # Group threshold (mean of optimal thresholds for this group)
            g_threshold = thresh_tensor[indices].mean().item()
            
            # Check if predicted logit is below threshold + logit margin
            if max_logit <= (g_threshold + logit_margin):
                # Correct this group
                logits_mutated[i, indices] = gt_logits[i, indices]
                corrected_count += 1
        corrected_counts.append(corrected_count)
        
    # Predict target classes using mutated logits
    with torch.no_grad():
        if getattr(model, "use_multimodal", False) and all_tabular_features is not None:
            inputs_mutated = torch.cat([logits_mutated.to(device), all_tabular_features.to(device)], dim=-1)
            updated_logits = model.classifier_head(inputs_mutated)
        elif getattr(model, "age_sex_skip_connection", False) and all_tabular_features is not None:
            updated_logits = model.classifier_head(logits_mutated.to(device)) + model.tabular_skip_head(all_tabular_features.to(device))
        else:
            inputs_mutated = logits_mutated.to(device)
            updated_logits = model.classifier_head(inputs_mutated)
        updated_topk = calculate_topk_accuracy(updated_logits.cpu(), gt_targets)
        
    avg_corrected = np.mean(corrected_counts)
    return updated_topk, avg_corrected


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
    use_lora = checkpoint_args.get('use_lora') or checkpoint_config.get('backbone', {}).get('use_lora') or False
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

    use_multimodal = False
    if 'use_multimodal' in checkpoint_args:
        use_multimodal = checkpoint_args['use_multimodal']
    elif 'use_multimodal' in checkpoint_config.get('dataset', {}):
        use_multimodal = checkpoint_config['dataset']['use_multimodal']
    elif "classifier_head.concept_gates" in state_dict:
        num_supervised_gates = state_dict["classifier_head.concept_gates"].shape[0]
        if num_supervised_gates > 9 and dataset_name == 'chexpert':
            use_multimodal = True
    elif "classifier_head.weight" in state_dict:
        checkpoint_dims = state_dict["classifier_head.weight"].shape[1]
        if checkpoint_dims > 9 and dataset_name == 'chexpert':
            use_multimodal = True

    # Auto-detect age_sex_skip_connection from checkpoint
    age_sex_skip_connection = False
    if 'age_sex_skip_connection' in checkpoint_args:
        age_sex_skip_connection = checkpoint_args['age_sex_skip_connection']
    elif 'age_sex_skip_connection' in checkpoint_config.get('dataset', {}):
        age_sex_skip_connection = checkpoint_config['dataset']['age_sex_skip_connection']
    elif 'tabular_skip_head.weight' in state_dict:
        age_sex_skip_connection = True

    tqdm.write(f"  {BOLD}{BLUE}[Config]{RESET} Auto-detected config:")
    tqdm.write(f"     ├─ Dataset: {dataset_name.upper()}")
    tqdm.write(f"     ├─ Backbone: {backbone_name} ({backbone_type})")
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
    tqdm.write(f"     ├─ use_multimodal: {use_multimodal}")
    tqdm.write(f"     ├─ age_sex_skip_connection: {age_sex_skip_connection}")
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
    
    # Load split
    split_name = 'val' if args.split in ['val', 'validation'] else 'test'
    test_dataset = dataset_class(
        csv_path=csv_path,
        image_dir=image_dir,
        split=split_name,
        config=dataset_config,
        cache_in_memory=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
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
        use_pairwise_nam=use_pairwise_nam,
        use_multimodal=use_multimodal,
        age_sex_skip_connection=age_sex_skip_connection,
    )
    
    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    
    tqdm.write(f"\n{BOLD}{MAGENTA}============================================================{RESET}")
    tqdm.write(f"  {BOLD}{MAGENTA}[Evaluation]{RESET} Running {args.split.upper()} Set Evaluation...")
    tqdm.write(f"{BOLD}{MAGENTA}============================================================{RESET}")
    
    # 5. Run standard evaluation
    topk_accs, concept_metrics, concept_logits, gt_concepts, gt_targets, all_tabular_features = run_evaluation(
        model, 
        test_loader, 
        concept_groups_info if not use_group_broadcasting else None, 
        device
    )
    
    tqdm.write(f"\n{BOLD}{GREEN}[Performance] Standard CBM {args.split.upper()} Performance:{RESET}")
    tqdm.write(f"   Target Accuracy (Top-1)  : {BOLD}{GREEN}{topk_accs.get(1, 0.0)*100:.2f}%{RESET}")
    if 3 in topk_accs: tqdm.write(f"   Target Accuracy (Top-3)  : {topk_accs[3]*100:.2f}%")
    if 5 in topk_accs: tqdm.write(f"   Target Accuracy (Top-5)  : {topk_accs[5]*100:.2f}%")
    if 10 in topk_accs: tqdm.write(f"   Target Accuracy (Top-10) : {topk_accs[10]*100:.2f}%")
    tqdm.write(f"   Concept Mean Balanced Accuracy : {concept_metrics['mean_balanced_acc']*100:.2f}%")
    tqdm.write(f"   Concept Mean True Positive Rate: {concept_metrics['tpr']*100:.2f}%")
    tqdm.write(f"   Concept Mean True Negative Rate: {concept_metrics['tnr']*100:.2f}%")
    tqdm.write(f"   Concept Mean F1-Score          : {concept_metrics.get('mean_f1', 0.0)*100:.2f}%")
    tqdm.write(f"   Concept Mean F2-Score          : {concept_metrics.get('mean_f_beta', 0.0)*100:.2f}%")
    
    # 6. Run Group-level Test-Time Intervention (TTI)
    tqdm.write(f"\n{BOLD}{BLUE}============================================================{RESET}")
    tqdm.write(f"  {BOLD}{BLUE}[TTI - Group Level]{RESET} Running Group-level Test-Time Intervention (TTI)...")
    tqdm.write(f"  (Correcting anatomical attribute groups in order of prediction error)")
    tqdm.write(f"{BOLD}{BLUE}============================================================{RESET}")
    
    group_tti_results = run_tti_group_level(
        model, 
        concept_logits, 
        gt_concepts, 
        gt_targets, 
        concept_groups, 
        latent_concepts, 
        device,
        all_tabular_features=all_tabular_features
    )
    
    # Print beautiful ASCII table for group-level TTI with Top-K metrics
    available_ks = sorted(list(group_tti_results[0][1].keys()))
    header = "| Number of Groups Corrected (TTI)   "
    for k in available_ks:
        header += f"| Top-{k:<4} "
    header += "|"
    border = "+" + "-"*36
    for k in available_ks:
        border += "+" + "-"*10
    border += "+"
    
    print("\n" + border)
    print(header)
    print(border)
    for K, accs_dict in group_tti_results:
        row = f"| {K:<34} "
        for k in available_ks:
            val = accs_dict.get(k, 0.0) * 100
            row += f"| {val:>8.2f}% "
        row += "|"
        print(row)
    print(border)
    
    # 7. Run Individual Concept-level Test-Time Intervention (TTI)
    tqdm.write(f"\n{BOLD}{BLUE}============================================================{RESET}")
    tqdm.write(f"  {BOLD}{BLUE}[TTI - Concept Level]{RESET} Running Individual Concept-level TTI...")
    tqdm.write(f"  (Correcting individual concepts by percentage)")
    tqdm.write(f"{BOLD}{BLUE}============================================================{RESET}")
    
    concept_tti_results = run_tti_concept_level(
        model, 
        concept_logits, 
        gt_concepts, 
        gt_targets, 
        concept_groups, 
        latent_concepts, 
        device,
        all_tabular_features=all_tabular_features
    )
    
    # Print beautiful ASCII table for concept-level TTI with Top-K metrics
    available_ks_concept = sorted(list(concept_tti_results[0][1].keys()))
    header_concept = "| Percentage of Concepts Corrected   "
    for k in available_ks_concept:
        header_concept += f"| Top-{k:<4} "
    header_concept += "|"
    border_concept = "+" + "-"*36
    for k in available_ks_concept:
        border_concept += "+" + "-"*10
    border_concept += "+"
    
    print("\n" + border_concept)
    print(header_concept)
    print(border_concept)
    for pct, accs_dict in concept_tti_results:
        row = f"| {f'{pct}%':<34} "
        for k in available_ks_concept:
            val = accs_dict.get(k, 0.0) * 100
            row += f"| {val:>8.2f}% "
        row += "|"
        print(row)
    print(border_concept)
    
    # 8. Run "Not Visible / Occluded" Only TTI under different logit margins
    print(f"\n{BOLD}{YELLOW}============================================================{RESET}")
    print(f"  {BOLD}{YELLOW}[Margin Search]{RESET} Searching for the optimal logit margin to achieve ~2-3 corrected groups...")
    print(f"{BOLD}{YELLOW}============================================================{RESET}")
    
    logit_margin_candidates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    print("| Logit Margin | Top-1 Accuracy | Top-3 Accuracy | Avg Groups Corrected |")
    print("| :----------: | :------------: | :------------: | :------------------: |")
    for margin in logit_margin_candidates:
        unconf_topk, avg_corrected = run_tti_unconfident_only(
            model, 
            concept_logits, 
            gt_concepts, 
            gt_targets, 
            concept_groups, 
            latent_concepts, 
            device,
            logit_margin=margin,
            all_tabular_features=all_tabular_features
        )
        print(f"| {margin:>12.2f} | {unconf_topk.get(1, 0.0)*100:>12.2f}% | {unconf_topk.get(3, 0.0)*100:>12.2f}% | {avg_corrected:>20.2f} / {num_groups} |")
    print("============================================================\n")
    
    # 9. Evaluate Chosen Sweet-Spot Logit Margin (0.30)
    print(f"{BOLD}{YELLOW}============================================================{RESET}")
    print(f"  {BOLD}{YELLOW}[Unconfident TTI]{RESET} Running Unconfident-Only TTI with Selected Logit Margin (0.30)...")
    print(f"{BOLD}{YELLOW}============================================================{RESET}")
    unconf_topk, avg_corrected = run_tti_unconfident_only(
        model, 
        concept_logits, 
        gt_concepts, 
        gt_targets, 
        concept_groups, 
        latent_concepts, 
        device,
        logit_margin=0.30,
        all_tabular_features=all_tabular_features
    )
    print(f"  Unconfident-Only (Margin 0.30) Top-1 Accuracy : {unconf_topk.get(1, 0.0)*100:.2f}%")
    if 3 in unconf_topk: print(f"  Unconfident-Only (Margin 0.30) Top-3 Accuracy : {unconf_topk[3]*100:.2f}%")
    if 5 in unconf_topk: print(f"  Unconfident-Only (Margin 0.30) Top-5 Accuracy : {unconf_topk[5]*100:.2f}%")
    if 10 in unconf_topk: print(f"  Unconfident-Only (Margin 0.30) Top-10 Accuracy: {unconf_topk[10]*100:.2f}%")
    print(f"  Avg Groups Corrected per Sample               : {avg_corrected:.2f} / {num_groups}")
    print("============================================================\n")
    
    # Summary of accomplishments with Top-K metrics
    print(f"\n{BOLD}{GREEN}============================================================{RESET}")
    print(f"  {BOLD}{GREEN}[Success] TTI Benchmark Evaluation Complete!{RESET}")
    for k in available_ks:
        val_0 = group_tti_results[0][1][k] * 100
        val_all = group_tti_results[-1][1][k] * 100
        val_unconf = unconf_topk.get(k, 0.0) * 100
        delta = val_all - val_0
        delta_unconf = val_unconf - val_0
        print(f"  [TTI] Standard (K=0) Target Top-{k} Accuracy: {val_0:.2f}%")
        print(f"  [TTI] Unconfident-Only (Margin=0.30) Top-{k} Accuracy: {val_unconf:.2f}% ({BOLD}{YELLOW}{delta_unconf:+.2f}%{RESET})")
        print(f"  [TTI] Perfect Concept (K=All) Target Top-{k} Accuracy: {val_all:.2f}%")
        print(f"  [TTI] Top-{k} Intervention headroom (TTI Delta): {BOLD}{GREEN}{delta:+.2f}%{RESET}")
        print(f"  ----------------------------------------------------------")
    print(f"  [TTI] Unconfident Avg Groups Corrected: {avg_corrected:.2f} / {num_groups}")
    print(f"{BOLD}{GREEN}============================================================{RESET}\n")
    
    # 10. Export Active Pairwise NAM Interactions
    if getattr(model, "use_pairwise_nam", False):
        tqdm.write(f"\n{BOLD}{CYAN}[Interaction Logging]{RESET} Analyzing pairwise concept interactions...")
        gates = model.classifier_head.pairwise_gates.detach().cpu().numpy()
        pair_indices = model.classifier_head.pair_indices
        concepts_list = list(test_dataset.config.get("concepts_flat", test_dataset.config.get("concepts", [])))
        if getattr(model, "use_multimodal", False) and len(concepts_list) == 9:
            concepts_list += ["Age", "Sex (Male)", "Sex (Female)"]
            
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
