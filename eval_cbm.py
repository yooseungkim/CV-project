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
from src.models.cbm_factory import UniversalFlexibleCBM
from src.utils.metrics import calculate_accuracy, calculate_concept_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Concept Bottleneck Model Evaluation & Test-Time Intervention (TTI) Benchmark")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to saved CBM model checkpoint (.pt or .pth)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for testing")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Computation device")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loader")
    return parser.parse_args()


def calculate_topk_accuracy(outputs, targets, topk=(1, 3, 5, 10)):
    """Helper to calculate Top-K accuracy for target classes."""
    maxk = max(topk)
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
    
    pbar = tqdm(dataloader, desc="Evaluating CBM")
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
    return topk_accs, concept_metrics, all_concept_logits, all_gt_concepts, all_gt_targets


def run_tti_group_level(model, concept_logits, gt_concepts, gt_targets, concept_groups, latent_concepts, device):
    """Simulates group-level TTI by correcting attributes group-by-group in logit space."""
    model.eval()
    num_samples = concept_logits.shape[0]
    num_groups = len(concept_groups)
    
    # Pre-allocate array to store accuracy at each step
    group_tti_accuracies = []
    
    # Calculate initial predictions (K=0 interventions) using predicted logits directly
    init_logits = model.classifier_head(concept_logits.to(device))
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
        
    # Translate GT concepts (probabilities in [0, 1]) to logit space (z = log(p / (1 - p)))
    # Soft Intervention: Clip p to [0.05, 0.95] (logit range [-2.94, +2.94]) to match CBM predicted logit scale and prevent linear head saturation
    p_clipped = torch.clamp(gt_concepts, min=0.05, max=0.95)
    gt_logits = torch.log(p_clipped / (1.0 - p_clipped))
    
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
            updated_logits = model.classifier_head(logits_mutated.to(device))
            updated_topk = calculate_topk_accuracy(updated_logits.cpu(), gt_targets)
            
        group_tti_accuracies.append((K, updated_topk))
        pbar.set_postfix(acc=f"{updated_topk.get(1, 0.0) * 100:.2f}%")
        
    return group_tti_accuracies


def run_tti_concept_level(model, concept_logits, gt_concepts, gt_targets, latent_concepts, device):
    """Simulates individual concept-level TTI in logit space by correcting top-K most erroneous concepts."""
    model.eval()
    num_samples, num_supervised = gt_concepts.shape
    
    concept_tti_accuracies = []
    
    # Init (K=0)
    init_logits = model.classifier_head(concept_logits.to(device))
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
        
    # Translate GT concepts (probabilities in [0, 1]) to logit space
    # Soft Intervention: Clip p to [0.05, 0.95] (logit range [-2.94, +2.94]) to match CBM predicted logit scale and prevent linear head saturation
    p_clipped = torch.clamp(gt_concepts, min=0.05, max=0.95)
    gt_logits = torch.log(p_clipped / (1.0 - p_clipped))
    
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
            updated_logits = model.classifier_head(logits_mutated.to(device))
            updated_topk = calculate_topk_accuracy(updated_logits.cpu(), gt_targets)
            
        concept_tti_accuracies.append((pct, updated_topk))
        last_k = target_k
        
    return concept_tti_accuracies


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    tqdm.write(f"\n============================================================")
    tqdm.write(f"  🔍 Loading CBM Checkpoint: {args.checkpoint}")
    tqdm.write(f"============================================================")
    
    # 1. Load checkpoint meta to auto-detect model configs
    loaded = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_args = loaded.get('args', {})
    checkpoint_config = loaded.get('config', {})
    state_dict = loaded.get('state_dict', loaded)
    
    # Auto-detect configs
    dataset_name = checkpoint_args.get('dataset', 'cub')
    backbone_type = checkpoint_args.get('backbone_type', 'timm')
    backbone_name = checkpoint_args.get('backbone_name', 'vit_base_patch14_dinov2')
    use_lora = checkpoint_args.get('use_lora', False)
    lora_r = checkpoint_args.get('lora_r', 8)
    lora_alpha = checkpoint_args.get('lora_alpha', 16.0)
    use_group_broadcasting = checkpoint_args.get('use_group_broadcasting', False)
    use_cosine_attention = checkpoint_args.get('use_cosine_attention', False)
    latent_concepts = checkpoint_args.get('latent_concepts', 0)
    num_classes = checkpoint_args.get('num_classes', 200)
    filter_rare_concepts = checkpoint_args.get('filter_rare_concepts', False)
    if not filter_rare_concepts:
        filter_rare_concepts = checkpoint_config.get('dataset', {}).get('filter_rare_concepts', False)
    
    tqdm.write(f"  📦 Auto-detected config:")
    tqdm.write(f"     ├─ Dataset: {dataset_name.upper()}")
    tqdm.write(f"     ├─ Backbone: {backbone_name} ({backbone_type})")
    tqdm.write(f"     ├─ use_lora: {use_lora} (r={lora_r}, alpha={lora_alpha})")
    tqdm.write(f"     ├─ latent_concepts: {latent_concepts}")
    tqdm.write(f"     ├─ use_group_broadcasting: {use_group_broadcasting}")
    tqdm.write(f"     └─ filter_rare_concepts: {filter_rare_concepts}")
    
    # 2. Build Datasets and Loaders dynamically based on discovered configs
    if dataset_name == 'derm7pt':
        dataset_class = Derm7PtDataset
        csv_path = checkpoint_args.get('csv_path', 'data/derm7pt/meta/meta.csv')
        image_dir = checkpoint_args.get('image_dir', 'data/derm7pt/images')
        concept_config_path = checkpoint_args.get('concept_config_path', 'data/derm7pt/concept_config.json')
    else:
        dataset_class = CUB2011Dataset
        csv_path = checkpoint_args.get('csv_path', 'data/CUB_200_2011/images.txt')
        image_dir = checkpoint_args.get('image_dir', 'data/CUB_200_2011/images')
        concept_config_path = checkpoint_args.get('concept_config_path', 'data/CUB_200_2011/concept_config.json')
        if filter_rare_concepts:
            filtered_path = concept_config_path.replace(".json", "_filtered.json")
            if os.path.exists(filtered_path):
                concept_config_path = filtered_path
                tqdm.write(f"     ℹ️ Redirected concept_config to: {concept_config_path}")
        
    dataset_config = dataset_class.get_default_config()
    dataset_config["concept_config_path"] = concept_config_path
    dataset_config["filter_rare_concepts"] = filter_rare_concepts
    
    # Load test split
    test_dataset = dataset_class(
        csv_path=csv_path,
        image_dir=image_dir,
        split='test',
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
        group_mapping=group_mapping
    )
    
    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    
    tqdm.write(f"\n============================================================")
    tqdm.write(f"  🎬 Running Test Set Evaluation...")
    tqdm.write(f"============================================================")
    
    # 5. Run standard evaluation
    topk_accs, concept_metrics, concept_logits, gt_concepts, gt_targets = run_evaluation(
        model, 
        test_loader, 
        concept_groups_info if not use_group_broadcasting else None, 
        device
    )
    
    tqdm.write(f"\n📈 Standard CBM Test Performance:")
    tqdm.write(f"   🎯 Target Accuracy (Top-1)  : {topk_accs.get(1, 0.0)*100:.2f}%")
    if 3 in topk_accs: tqdm.write(f"   🎯 Target Accuracy (Top-3)  : {topk_accs[3]*100:.2f}%")
    if 5 in topk_accs: tqdm.write(f"   🎯 Target Accuracy (Top-5)  : {topk_accs[5]*100:.2f}%")
    if 10 in topk_accs: tqdm.write(f"   🎯 Target Accuracy (Top-10) : {topk_accs[10]*100:.2f}%")
    tqdm.write(f"   🧬 Concept Mean Balanced Accuracy : {concept_metrics['mean_balanced_acc']*100:.2f}%")
    tqdm.write(f"   🧬 Concept Mean True Positive Rate: {concept_metrics['tpr']*100:.2f}%")
    tqdm.write(f"   🧬 Concept Mean True Negative Rate: {concept_metrics['tnr']*100:.2f}%")
    
    # 6. Run Group-level Test-Time Intervention (TTI)
    tqdm.write(f"\n============================================================")
    tqdm.write(f"  🧑‍⚕️ Running Group-level Test-Time Intervention (TTI)...")
    tqdm.write(f"  (Correcting anatomical attribute groups in order of prediction error)")
    tqdm.write(f"============================================================")
    
    group_tti_results = run_tti_group_level(
        model, 
        concept_logits, 
        gt_concepts, 
        gt_targets, 
        concept_groups, 
        latent_concepts, 
        device
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
    tqdm.write(f"\n============================================================")
    tqdm.write(f"  🧑‍⚕️ Running Individual Concept-level TTI...")
    tqdm.write(f"  (Correcting individual concepts by percentage)")
    tqdm.write(f"============================================================")
    
    concept_tti_results = run_tti_concept_level(
        model, 
        concept_logits, 
        gt_concepts, 
        gt_targets, 
        latent_concepts, 
        device
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
    
    # Summary of accomplishments with Top-K metrics
    print(f"\n============================================================")
    print(f"  ✅ TTI Benchmark Evaluation Complete!")
    for k in available_ks:
        val_0 = group_tti_results[0][1][k] * 100
        val_all = group_tti_results[-1][1][k] * 100
        delta = val_all - val_0
        print(f"  🌟 Standard (K=0) Target Top-{k} Accuracy: {val_0:.2f}%")
        print(f"  🌟 Perfect Concept (K=All) Target Top-{k} Accuracy: {val_all:.2f}%")
        print(f"  📈 Top-{k} Intervention headroom (TTI Delta): {delta:+.2f}%")
        print(f"  ----------------------------------------------------------")
    print(f"============================================================\n")


if __name__ == "__main__":
    main()
