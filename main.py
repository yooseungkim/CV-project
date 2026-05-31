import argparse
import os
import datetime
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.milk10k import MILK10KDataset
from src.data.derm7pt import Derm7PtDataset
from src.data.cub import CUB2011Dataset
from src.models.cbm_factory import UniversalFlexibleCBM
from src.utils.metrics import calculate_accuracy, calculate_concept_accuracy, calculate_concept_metrics
from src.utils.visualization import generate_concept_heatmaps

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str_or_float(v):
    if v is None:
        return None
    if isinstance(v, str) and v.lower() == 'dynamic':
        return 'dynamic'
    try:
        return float(v)
    except ValueError:
        return str(v)
def str_or_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v

def calculate_orthogonality_loss(attn_weights):
    """
    Redesigned Orthogonality Loss (Preventing Attention Collapse).
    Computes pairwise cosine similarity matrix of parent group attention maps
    and penalizes deviation from the Identity matrix using Mean Squared Error (MSE).
    
    attn_weights: Tensor of shape (B, num_groups, H, W) representing mean parent group attention maps.
    """
    B, num_groups, H, W = attn_weights.shape
    if num_groups <= 1:
        return torch.tensor(0.0, device=attn_weights.device)
        
    # Flatten spatial dimensions: (B, num_groups, H * W)
    attn_flat = attn_weights.view(B, num_groups, -1)
    
    # L2 normalize over the spatial dimension to compute stable Cosine Similarity
    attn_norm = attn_flat / (torch.norm(attn_flat, p=2, dim=-1, keepdim=True) + 1e-8)
    
    # Compute pairwise Cosine Similarity matrix: (B, num_groups, num_groups)
    sim_matrix = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
    
    # Define target Identity matrix I_G: (B, num_groups, num_groups)
    identity = torch.eye(num_groups, device=attn_weights.device).unsqueeze(0).expand(B, -1, -1)
    
    # Minimize MSE(S, I) to push off-diagonal elements to 0 while keeping diagonals at 1
    loss_ortho = torch.mean((sim_matrix - identity) ** 2)
    return loss_ortho

def calculate_pos_weights(dataset, num_concepts_supervised):
    """Calculates the ratio of negative to positive samples for each concept to balance BCE loss."""
    import pandas as pd
    num_samples = len(dataset)
    if num_samples == 0:
        return torch.ones(num_concepts_supervised)
        
    # Attempt to use cached concepts if available
    if getattr(dataset, "_cache_populated", False) and dataset._cache is not None:
        concepts = torch.stack([sample[1][:num_concepts_supervised] for sample in dataset._cache], dim=0)
    elif hasattr(dataset, "concept_matrix") and dataset.concept_matrix is not None:
        # CUB Dataset
        image_idxs = dataset.df['image_idx'].values
        concepts = torch.tensor(dataset.concept_matrix[image_idxs, :num_concepts_supervised], dtype=torch.float32)
    elif hasattr(dataset, "df") and not dataset.df.empty:
        # MILK10K or Derm7pt
        concepts_list = []
        for idx in range(num_samples):
            row = dataset.df.iloc[idx]
            if dataset.concept_features_info is not None:
                concept_vals = []
                for info in dataset.concept_features_info:
                    name = info["name"]
                    val = row.get(name)
                    if info["type"] == "categorical":
                        classes = info["classes"]
                        one_hot = [0.0] * len(classes)
                        if pd.notna(val):
                            try:
                                if len(classes) > 0:
                                    target_type = type(classes[0])
                                    val_typed = target_type(val)
                                    if val_typed in classes:
                                        val_idx = classes.index(val_typed)
                                        one_hot[val_idx] = 1.0
                            except (ValueError, TypeError):
                                pass
                        concept_vals.extend(one_hot)
                    else:
                        min_val = info["min"]
                        max_val = info["max"]
                        if pd.isna(val):
                            scaled_val = 0.5
                        else:
                            try:
                                val_float = float(val)
                                denom = max_val - min_val
                                if denom == 0:
                                    scaled_val = 0.0
                                else:
                                    scaled_val = (val_float - min_val) / denom
                                    scaled_val = max(0.0, min(1.0, scaled_val))
                            except (ValueError, TypeError):
                                scaled_val = 0.5
                        concept_vals.append(scaled_val)
                concepts_list.append(torch.tensor(concept_vals, dtype=torch.float32))
            else:
                concept_vals = [float(row.get(col, 0.0)) for col in dataset.concept_cols]
                concepts_list.append(torch.tensor(concept_vals, dtype=torch.float32))
        concepts = torch.stack(concepts_list, dim=0)[:, :num_concepts_supervised]
    else:
        # Fallback to dummy concepts if none of the above
        concepts = torch.zeros((num_samples, num_concepts_supervised))

    # Calculate negative/positive ratio for each concept
    positives = (concepts > 0.5).sum(dim=0).float()
    negatives = (concepts <= 0.5).sum(dim=0).float()
    
    # Avoid division by zero
    pos_weight = negatives / (positives + 1e-8)
    # Clamp pos_weight to a reasonable range [0.1, 100.0] to prevent extreme gradient scaling
    pos_weight = torch.clamp(pos_weight, min=0.1, max=100.0)
    return pos_weight

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """Numerically stable Sigmoid Focal Loss for multi-label binary concept predictions.
        Focuses learning on hard, misclassified samples and down-weights easy majority classes.
        alpha can be:
        - A single float (constant for all concepts)
        - A torch.Tensor of shape (num_concepts,)
        - None (no alpha weighting applied)
        """
        super().__init__()
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Calculate focal weight: (1 - p_t) ^ gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        loss = focal_weight * bce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                self.alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class GroupCrossEntropyLoss(nn.Module):
    def __init__(self, groups_info: list[tuple[int, int]], lambda_ce: float = 0.1, loss_type: str = 'bce', focal_alpha = None, focal_gamma: float = 2.0):
        """Robust Group-level Softmax Cross Entropy Loss with Loss Scale Balancing.
        Penalizes prediction errors within mutually exclusive attribute categories (Softmax),
        balanced by lambda_ce against independent multi-label concept categories (Sigmoid).
        Supports both BCE and Sigmoid Focal Loss for the Sigmoid/1D fallback categories.
        
        groups_info: list of (start_idx, num_feats)
        lambda_ce: scaling hyperparameter for mutually exclusive cross entropy loss.
        loss_type: loss type for Sigmoid fallback nodes ('bce' or 'focal').
        focal_alpha: alpha weighting factor for Focal Loss (float, torch.Tensor, or None).
        focal_gamma: focus exponent gamma for Focal Loss.
        """
        super().__init__()
        self.groups_info = groups_info
        self.lambda_ce = lambda_ce
        self.loss_type = loss_type.lower()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        loss = 0.0
        active_groups = 0
        
        for start_idx, num_feats in self.groups_info:
            group_logits = logits[:, start_idx : start_idx + num_feats]
            group_targets = targets[:, start_idx : start_idx + num_feats]
            
            if num_feats > 1:
                # 1. Softmax group (Mutually Exclusive Cross Entropy Loss)
                target_sum = group_targets.sum(dim=-1, keepdim=True)
                # Secure target normalization to form a probability distribution (Soft Target)
                group_targets_normalized = group_targets / torch.clamp(target_sum, min=1e-8)
                
                # Apply PyTorch's native cross_entropy supporting Soft Targets
                group_loss = F.cross_entropy(group_logits, group_targets_normalized, reduction='none')
                
                # Mask out samples that do not have annotations for this group
                mask = (target_sum.squeeze(-1) > 0.0).float()
                if mask.sum() > 0:
                    # Apply loss scale balancing multiplier (lambda_ce) to prevent gradient starvation
                    loss += self.lambda_ce * (group_loss * mask).sum() / (mask.sum() + 1e-8)
                    active_groups += 1
            else:
                # 2. Sigmoid / BCE / 1D fallback group (independent binary node)
                if self.loss_type == 'focal':
                    # Calculate focal loss for this individual node
                    probs = torch.sigmoid(group_logits)
                    bce_loss = F.binary_cross_entropy_with_logits(group_logits, group_targets, reduction='none')
                    p_t = probs * group_targets + (1 - probs) * (1 - group_targets)
                    focal_loss = ((1 - p_t) ** self.focal_gamma) * bce_loss
                    
                    if self.focal_alpha is not None:
                        # Extract alpha value for this specific concept dimension
                        if isinstance(self.focal_alpha, torch.Tensor):
                            # self.focal_alpha has shape (num_concepts,)
                            alpha_t = self.focal_alpha[start_idx] * group_targets + (1 - self.focal_alpha[start_idx]) * (1 - group_targets)
                        else:
                            alpha_t = self.focal_alpha * group_targets + (1 - self.focal_alpha) * (1 - group_targets)
                        focal_loss = alpha_t * focal_loss
                        
                    loss += focal_loss.mean()
                    active_groups += 1
                else:
                    # Standard BCE Loss fallback
                    loss += F.binary_cross_entropy_with_logits(group_logits.squeeze(-1), group_targets.squeeze(-1))
                    active_groups += 1
                
        return loss / (active_groups + 1e-8)

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, monitor: str = "val_loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
        # Decide direction based on monitor name
        if "loss" in monitor.lower():
            self.mode = "min"
        else:
            self.mode = "max"
            
    def __call__(self, val_score: float, model: nn.Module):
        score = -val_score if self.mode == "min" else val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            tqdm.write(f"  ⏳ EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model: nn.Module):
        self.best_weights = copy.deepcopy(model.state_dict())

def get_dataset_choices():
    data_dir = 'data'
    default_choices = ['milk10k', 'derm7pt', 'cub']
    if not os.path.exists(data_dir):
        return default_choices
    choices = []
    for item in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, item)):
            choices.append(item.lower())
    return sorted(list(set(choices + default_choices)))

def parse_args():
    # Stage 1: Parse only the --config_path argument
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--config_path', type=str, default=None)
    temp_args, _ = temp_parser.parse_known_args()
    
    # Load defaults from config file if provided
    config_data = {}
    if temp_args.config_path and os.path.exists(temp_args.config_path):
        ext = os.path.splitext(temp_args.config_path)[1].lower()
        if ext in ['.yaml', '.yml']:
            import yaml
            with open(temp_args.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        else:
            import json
            with open(temp_args.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        tqdm.write(f"  📄 Config loaded: {temp_args.config_path}")
        
    flat_defaults = {}
    
    # backbone
    bb_cfg = config_data.get("backbone", {})
    if "backbone_type" in bb_cfg: flat_defaults["backbone_type"] = bb_cfg["backbone_type"]
    if "backbone_name" in bb_cfg: flat_defaults["backbone_name"] = bb_cfg["backbone_name"]
    if "freeze_backbone" in bb_cfg: flat_defaults["freeze_backbone"] = bb_cfg["freeze_backbone"]
    if "freeze_head" in bb_cfg: flat_defaults["freeze_head"] = bb_cfg["freeze_head"]
    if "use_lora" in bb_cfg: flat_defaults["use_lora"] = bb_cfg["use_lora"]
    if "lora_r" in bb_cfg: flat_defaults["lora_r"] = bb_cfg["lora_r"]
    if "lora_alpha" in bb_cfg: flat_defaults["lora_alpha"] = bb_cfg["lora_alpha"]
    if "use_cosine_attention" in bb_cfg: flat_defaults["use_cosine_attention"] = bb_cfg["use_cosine_attention"]
    
    # dataset
    ds_cfg = config_data.get("dataset", {})
    if "dataset" in ds_cfg: flat_defaults["dataset"] = ds_cfg["dataset"]
    if "csv_path" in ds_cfg: flat_defaults["csv_path"] = ds_cfg["csv_path"]
    if "image_dir" in ds_cfg: flat_defaults["image_dir"] = ds_cfg["image_dir"]
    if "concept_config_path" in ds_cfg: flat_defaults["concept_config_path"] = ds_cfg["concept_config_path"]
    if "use_concept_groups" in ds_cfg: flat_defaults["use_concept_groups"] = ds_cfg["use_concept_groups"]
    
    # training
    tr_cfg = config_data.get("training", {})
    if "epochs" in tr_cfg: flat_defaults["epochs"] = tr_cfg["epochs"]
    if "batch_size" in tr_cfg: flat_defaults["batch_size"] = tr_cfg["batch_size"]
    if "lambda_c" in tr_cfg: flat_defaults["lambda_c"] = tr_cfg["lambda_c"]
    if "num_classes" in tr_cfg: flat_defaults["num_classes"] = tr_cfg["num_classes"]
    if "save_dir" in tr_cfg: flat_defaults["save_dir"] = tr_cfg["save_dir"]
    if "use_wandb" in tr_cfg: flat_defaults["use_wandb"] = tr_cfg["use_wandb"]
    if "target_pos_weight" in tr_cfg: flat_defaults["target_pos_weight"] = tr_cfg["target_pos_weight"]
    if "num_workers" in tr_cfg: flat_defaults["num_workers"] = tr_cfg["num_workers"]
    if "pin_memory" in tr_cfg: flat_defaults["pin_memory"] = tr_cfg["pin_memory"]
    if "cache_in_memory" in tr_cfg: flat_defaults["cache_in_memory"] = tr_cfg["cache_in_memory"]
    if "max_cache_size_gb" in tr_cfg: flat_defaults["max_cache_size_gb"] = tr_cfg["max_cache_size_gb"]
    if "resume_checkpoint" in tr_cfg: flat_defaults["resume_checkpoint"] = tr_cfg["resume_checkpoint"]
    if "latent_concepts" in tr_cfg: flat_defaults["latent_concepts"] = tr_cfg["latent_concepts"]
    if "run_app" in tr_cfg: flat_defaults["run_app"] = tr_cfg["run_app"]
    if "phase1_epochs" in tr_cfg: flat_defaults["phase1_epochs"] = tr_cfg["phase1_epochs"]
    if "phase2_epochs" in tr_cfg: flat_defaults["phase2_epochs"] = tr_cfg["phase2_epochs"]
    
    # optimizer basic parameter
    opt_cfg = config_data.get("optimizer", {})
    if "lr" in opt_cfg: flat_defaults["lr"] = opt_cfg["lr"]
    if "phase1_lr" in opt_cfg: flat_defaults["phase1_lr"] = opt_cfg["phase1_lr"]
    if "phase2_lr" in opt_cfg: flat_defaults["phase2_lr"] = opt_cfg["phase2_lr"]
    if "concept_loss_type" in opt_cfg: flat_defaults["concept_loss_type"] = opt_cfg["concept_loss_type"]
    if "focal_alpha" in opt_cfg: flat_defaults["focal_alpha"] = opt_cfg["focal_alpha"]
    if "focal_gamma" in opt_cfg: flat_defaults["focal_gamma"] = opt_cfg["focal_gamma"]
    if "ortho_lambda" in opt_cfg: flat_defaults["ortho_lambda"] = opt_cfg["ortho_lambda"]
    if "lambda_ce" in opt_cfg: flat_defaults["lambda_ce"] = opt_cfg["lambda_ce"]
    if "lambda_latent_ortho" in opt_cfg: flat_defaults["lambda_latent_ortho"] = opt_cfg["lambda_latent_ortho"]
    if "lambda_latent_l1" in opt_cfg: flat_defaults["lambda_latent_l1"] = opt_cfg["lambda_latent_l1"]
    
    # early stopping patience
    es_cfg = config_data.get("early_stopping", {})
    if "phase1_patience" in es_cfg: flat_defaults["phase1_patience"] = es_cfg["phase1_patience"]
    if "phase2_patience" in es_cfg: flat_defaults["phase2_patience"] = es_cfg["phase2_patience"]
    if "phase3_patience" in es_cfg: flat_defaults["phase3_patience"] = es_cfg["phase3_patience"]
    if "phase1_monitor" in es_cfg: flat_defaults["phase1_monitor"] = es_cfg["phase1_monitor"]
    if "phase2_monitor" in es_cfg: flat_defaults["phase2_monitor"] = es_cfg["phase2_monitor"]
    if "phase3_monitor" in es_cfg: flat_defaults["phase3_monitor"] = es_cfg["phase3_monitor"]
    
    # training parameters
    if "l1_lambda" in tr_cfg: flat_defaults["l1_lambda"] = tr_cfg["l1_lambda"]
    if "phase3_epochs" in tr_cfg: flat_defaults["phase3_epochs"] = tr_cfg["phase3_epochs"]
    if "phase3_lr" in opt_cfg: flat_defaults["phase3_lr"] = opt_cfg["phase3_lr"]
    
    # Stage 2: Create full parser with dynamic defaults
    parser = argparse.ArgumentParser(description="Train a Modular CBM")
    choices = get_dataset_choices()
    
    parser.add_argument('--config_path', type=str, default=None, help="Path to config JSON file")
    parser.add_argument('--dataset', type=str, default=flat_defaults.get('dataset', 'milk10k'), choices=choices)
    parser.add_argument('--csv_path', type=str, default=flat_defaults.get('csv_path', None))
    parser.add_argument('--image_dir', type=str, default=flat_defaults.get('image_dir', None))
    parser.add_argument('--backbone_type', type=str, default=flat_defaults.get('backbone_type', 'timm'), choices=['timm', 'clip'])
    parser.add_argument('--backbone_name', type=str, default=flat_defaults.get('backbone_name', 'resnet50'))
    parser.add_argument('--num_concepts', type=int, default=None)
    parser.add_argument('--concept_cols', type=str, default=None)
    parser.add_argument('--concept_config_path', type=str, default=flat_defaults.get('concept_config_path', None))
    parser.add_argument('--latent_concepts', '--latent-concepts', type=int, default=flat_defaults.get('latent_concepts', 0), help="Number of unsupervised latent concepts to append to the bottleneck")
    parser.add_argument('--num_classes', type=int, default=flat_defaults.get('num_classes', 1))
    parser.add_argument('--epochs', type=int, default=flat_defaults.get('epochs', 1))
    parser.add_argument('--phase1_epochs', type=int, default=flat_defaults.get('phase1_epochs', None), help="Number of epochs for Phase 1 (Concept Learning)")
    parser.add_argument('--phase2_epochs', type=int, default=flat_defaults.get('phase2_epochs', None), help="Number of epochs for Phase 2 (Target Learning)")
    parser.add_argument('--phase1_lr', type=float, default=flat_defaults.get('phase1_lr', None), help="Learning rate for Phase 1 concept head")
    parser.add_argument('--phase2_lr', type=float, default=flat_defaults.get('phase2_lr', None), help="Learning rate for Phase 2 latent head and classifier")
    parser.add_argument('--phase3_lr', type=float, default=flat_defaults.get('phase3_lr', 1e-5), help="Learning rate for Phase 3 joint fine-tuning")
    parser.add_argument('--phase1_patience', type=int, default=flat_defaults.get('phase1_patience', None), help="Early stopping patience for Phase 1")
    parser.add_argument('--phase2_patience', type=int, default=flat_defaults.get('phase2_patience', None), help="Early stopping patience for Phase 2")
    parser.add_argument('--phase3_patience', type=int, default=flat_defaults.get('phase3_patience', 5), help="Early stopping patience for Phase 3")
    parser.add_argument('--phase1_monitor', type=str, default=flat_defaults.get('phase1_monitor', 'val_concept_loss'), help="Early stopping monitor for Phase 1")
    parser.add_argument('--phase2_monitor', type=str, default=flat_defaults.get('phase2_monitor', 'val_target_loss'), help="Early stopping monitor for Phase 2")
    parser.add_argument('--phase3_monitor', type=str, default=flat_defaults.get('phase3_monitor', 'val_target_loss'), help="Early stopping monitor for Phase 3")
    parser.add_argument('--phase3_epochs', type=int, default=flat_defaults.get('phase3_epochs', 5), help="Number of epochs for Phase 3 (Joint Parameter Tuning)")
    parser.add_argument('--lambda_ce', type=float, default=flat_defaults.get('lambda_ce', 0.1), help="Scaling factor for Softmax Cross-Entropy loss to balance gradient scale against Focal/BCE loss")
    parser.add_argument('--concept_loss_type', type=str, default=flat_defaults.get('concept_loss_type', 'focal'), choices=['focal', 'bce'], help="Concept loss function type")
    parser.add_argument('--focal_alpha', type=str_or_float, default=flat_defaults.get('focal_alpha', 'dynamic'), help="Alpha parameter for Focal Loss (float or 'dynamic')")
    parser.add_argument('--focal_gamma', type=float, default=flat_defaults.get('focal_gamma', 2.0), help="Gamma parameter for Focal Loss")
    parser.add_argument('--ortho_lambda', type=float, default=flat_defaults.get('ortho_lambda', 0.05), help="Orthogonality regularization loss multiplier for attention map separation")
    parser.add_argument('--l1_lambda', type=float, default=flat_defaults.get('l1_lambda', 0.0), help="L1 Lasso regularization multiplier for Phase 2 classifier")
    parser.add_argument('--lambda_latent_ortho', type=float, default=flat_defaults.get('lambda_latent_ortho', 0.1), help="Orthogonal latent projection loss weight")
    parser.add_argument('--lambda_latent_l1', type=float, default=flat_defaults.get('lambda_latent_l1', 0.01), help="L1 latent activation sparsity loss weight")
    parser.add_argument('--batch_size', type=int, default=flat_defaults.get('batch_size', 16))
    parser.add_argument('--lr', type=float, default=flat_defaults.get('lr', 1e-3))
    parser.add_argument('--lambda_c', type=float, default=flat_defaults.get('lambda_c', 1.0))
    parser.add_argument('--target_pos_weight', type=float, default=flat_defaults.get('target_pos_weight', 1.0))
    parser.add_argument('--num_workers', type=int, default=flat_defaults.get('num_workers', 4))
    parser.add_argument('--pin_memory', type=str2bool, default=flat_defaults.get('pin_memory', True))
    parser.add_argument('--freeze_backbone', action='store_true', default=flat_defaults.get('freeze_backbone', False))
    parser.add_argument('--freeze_head', action='store_true', default=flat_defaults.get('freeze_head', False))
    parser.add_argument('--use_concept_groups', type=str_or_bool, default=flat_defaults.get('use_concept_groups', True), help="Toggle Group-level Softmax and GroupCrossEntropyLoss (True/False, or comma-separated list of concept names to group)")
    parser.add_argument('--use_lora', type=str2bool, default=flat_defaults.get('use_lora', False), help="Use Low-Rank Adaptation (LoRA) for ViT backbone tuning")
    parser.add_argument('--lora_r', type=int, default=flat_defaults.get('lora_r', 8), help="LoRA Rank parameter r")
    parser.add_argument('--lora_alpha', type=float, default=flat_defaults.get('lora_alpha', 16.0), help="LoRA scaling parameter alpha")
    parser.add_argument('--use_cosine_attention', type=str2bool, default=flat_defaults.get('use_cosine_attention', False), help="Use L2-normalized Cosine Attention instead of standard MultiheadAttention (suppresses DINOv2 border-patch outliers)")
    parser.add_argument('--use_group_broadcasting', type=str2bool, default=flat_defaults.get('use_group_broadcasting', False), help="Use GroupToConceptAttention: group queries → independent BCE classifiers based on concept_config (fixes TPR/TNR collapse from Group Softmax)")
    parser.add_argument('--use_wandb', type=str2bool, default=flat_defaults.get('use_wandb', True))
    parser.add_argument('--save_dir', type=str, default=flat_defaults.get('save_dir', 'checkpoints'))
    parser.add_argument('--cache_in_memory', type=str2bool, default=flat_defaults.get('cache_in_memory', False))
    parser.add_argument('--max_cache_size_gb', type=float, default=flat_defaults.get('max_cache_size_gb', 10.0))
    parser.add_argument('--resume_checkpoint', type=str, default=flat_defaults.get('resume_checkpoint', None), help="Path to checkpoint .pth to resume or fine-tune from")
    parser.add_argument('--save_filename', type=str, default=None, help="Custom filename to save the final weights")
    parser.add_argument('--run_app', type=str2bool, default=flat_defaults.get('run_app', True), help="Automatically launch Gradio app after training finishes")
    
    args = parser.parse_args()
    return args, config_data

def train_phase1(model, train_loader, val_loader, concept_criterion, device, args, config_data, run_name, num_concepts_supervised, resolved_config, concept_groups_info=None):
    tqdm.write(f"\n{'-'*60}")
    tqdm.write("  🎬 Phase 1: Concept Learning (Backbone & Concept Head)")
    tqdm.write(f"{'-'*60}")
    
    # Extract concept grouping indices from dataset for group-level orthogonality loss
    concept_groups_indices = None
    train_dataset = train_loader.dataset
    if args.use_concept_groups and hasattr(train_dataset, "concept_features_info") and train_dataset.concept_features_info is not None:
        target_groups = None
        if isinstance(args.use_concept_groups, str):
            if args.use_concept_groups.lower() == 'true':
                target_groups = None
            elif args.use_concept_groups.lower() == 'false':
                target_groups = set()
            else:
                target_groups = {name.strip() for name in args.use_concept_groups.split(',')}
        elif isinstance(args.use_concept_groups, list):
            target_groups = {str(name).strip() for name in args.use_concept_groups}
            
        concept_groups_indices = []
        for info in train_dataset.concept_features_info:
            name = info["name"]
            if target_groups is not None and name not in target_groups:
                continue
            start = info["start_idx"]
            num = info["num_feats"]
            indices = [idx for idx in range(start, start + num) if idx < num_concepts_supervised]
            if indices:
                concept_groups_indices.append(indices)
        tqdm.write(f"  📂 Group-level Orthogonality: Detected {len(concept_groups_indices)} semantic attribute groups for separation.")
    
    # classifier_head 가중치 동결
    for param in model.classifier_head.parameters():
        param.requires_grad = False
        
    if not args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = True
    model.unfreeze_supervised_attention()
    model.freeze_latent_attention()
        
    opt_cfg = config_data.get("optimizer", {})
    opt_type = opt_cfg.get("type", "adam").lower()
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    backbone_lr = opt_cfg.get("backbone_lr")
    head_lr = opt_cfg.get("head_lr")
    
    param_groups = []
    if not args.freeze_backbone:
        backbone_trainable = [p for p in model.backbone.parameters() if p.requires_grad]
        if backbone_lr is not None:
            param_groups.append({"params": backbone_trainable, "lr": backbone_lr})
        else:
            param_groups.append({"params": backbone_trainable, "lr": args.lr})
            
    # Phase 1 learning rate configuration
    phase1_lr = args.phase1_lr if args.phase1_lr is not None else opt_cfg.get("phase1_lr", opt_cfg.get("head_lr", args.lr))
    concept_trainable = [p for p in model.supervised_attention.parameters() if p.requires_grad]
    param_groups.append({"params": concept_trainable, "lr": phase1_lr})
        
    if opt_type == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    elif opt_type == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        optimizer = optim.SGD(param_groups, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
        
    sched_cfg = config_data.get("scheduler", {})
    sched_type = sched_cfg.get("type", "none").lower()
    scheduler = None
    phase1_epochs = args.phase1_epochs if args.phase1_epochs is not None else args.epochs
    
    if sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=sched_cfg.get("eta_min", 1e-6))
    elif sched_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_cfg.get("step_size", 10), gamma=sched_cfg.get("gamma", 0.1))
        
    # Phase 1은 val_concept_loss 또는 사용자가 지정한 phase1_monitor를 기반으로 early stopping 수행
    phase1_patience = args.phase1_patience if args.phase1_patience is not None else config_data.get("early_stopping", {}).get("phase1_patience", 5)
    phase1_monitor = args.phase1_monitor if getattr(args, "phase1_monitor", None) is not None else config_data.get("early_stopping", {}).get("phase1_monitor", "val_concept_loss")
    es_handler = EarlyStopping(patience=phase1_patience, min_delta=0.0, monitor=phase1_monitor)
    
    for epoch in range(phase1_epochs):
        model.train()
        total_loss_c = 0.0
        total_acc_c = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"  P1 Epoch {epoch+1}/{phase1_epochs}", bar_format="{l_bar}{bar:25}{r_bar}", leave=False)
        for images, concepts, _ in train_pbar:
            images = images.to(device)
            concepts = concepts.to(device)
            
            optimizer.zero_grad()
            _, concept_logits, attn_weights = model(images)
            
            # Use raw logits with BCEWithLogitsLoss
            loss_c = concept_criterion(concept_logits[:, :num_concepts_supervised], concepts)
            
            # Compute spatial orthogonality loss for the supervised concept attention maps
            if getattr(args, "ortho_lambda", 0.0) > 0.0:
                supervised_attn = attn_weights[:, :num_concepts_supervised]
                if concept_groups_indices is not None:
                    # Aggregate attention maps to group level (mean of concepts in each group)
                    group_attns = []
                    for indices in concept_groups_indices:
                        group_attn_agg = supervised_attn[:, indices].mean(dim=1)
                        group_attns.append(group_attn_agg)
                    attn_to_ortho = torch.stack(group_attns, dim=1)
                else:
                    attn_to_ortho = supervised_attn
                
                loss_ortho = calculate_orthogonality_loss(attn_to_ortho)
                total_loss = loss_c + args.ortho_lambda * loss_ortho
            else:
                total_loss = loss_c
                loss_ortho = torch.tensor(0.0, device=device)
                
            total_loss.backward()
            optimizer.step()
            
            total_loss_c += total_loss.item()
            
            # Calculate Balanced Accuracy for train batch reporting
            batch_metrics = calculate_concept_metrics(concept_logits[:, :num_concepts_supervised].detach(), concepts, concept_groups_info=concept_groups_info)
            total_acc_c += batch_metrics["mean_balanced_acc"]
            train_pbar.set_postfix(CL=f"{loss_c.item():.4f}", OL=f"{loss_ortho.item():.4f}", BA=f"{batch_metrics['mean_balanced_acc']:.4f}")
            
        avg_loss_c = total_loss_c / len(train_loader)
        avg_acc_c = total_acc_c / len(train_loader)
        
        model.eval()
        val_loss_c = 0.0
        val_acc_c = 0.0
        all_val_probs = []
        all_val_targets = []
        val_vis_data = None
        
        with torch.no_grad():
            for val_images, val_concepts, _ in val_loader:
                val_images = val_images.to(device)
                val_concepts = val_concepts.to(device)
                
                _, v_concept_logits, v_attn_weights = model(val_images)
                
                # BCEWithLogitsLoss with raw logits
                v_loss_c = concept_criterion(v_concept_logits[:, :num_concepts_supervised], val_concepts)
                
                if getattr(args, "ortho_lambda", 0.0) > 0.0:
                    v_supervised_attn = v_attn_weights[:, :num_concepts_supervised]
                    if concept_groups_indices is not None:
                        # Aggregate attention maps to group level
                        v_group_attns = []
                        for indices in concept_groups_indices:
                            v_group_attn_agg = v_supervised_attn[:, indices].mean(dim=1)
                            v_group_attns.append(v_group_attn_agg)
                        v_attn_to_ortho = torch.stack(v_group_attns, dim=1)
                    else:
                        v_attn_to_ortho = v_supervised_attn
                    
                    v_loss_ortho = calculate_orthogonality_loss(v_attn_to_ortho)
                    v_total_loss = v_loss_c + args.ortho_lambda * v_loss_ortho
                else:
                    v_total_loss = v_loss_c
                    
                val_loss_c += v_total_loss.item()
                
                # Append raw logits to compute final metrics over the entire epoch
                all_val_probs.append(v_concept_logits[:, :num_concepts_supervised].cpu())
                all_val_targets.append(val_concepts.cpu())
                if val_vis_data is None:
                    val_vis_data = (val_images, v_attn_weights)
                    
        avg_val_loss_c = val_loss_c / len(val_loader)
        
        # Compute Balanced Accuracy, TPR, and TNR over the full validation set
        if all_val_probs:
            val_logits_all = torch.cat(all_val_probs, dim=0)
            val_targets_all = torch.cat(all_val_targets, dim=0)
            val_metrics = calculate_concept_metrics(val_logits_all, val_targets_all, concept_groups_info=concept_groups_info)
            avg_val_acc_c = val_metrics["mean_balanced_acc"]
            val_tpr = val_metrics["tpr"]
            val_tnr = val_metrics["tnr"]
        else:
            avg_val_acc_c = 0.0
            val_tpr = 0.0
            val_tnr = 0.0
            
        # 에포크 정보 한 줄 출력 (스크롤 이력 보존)
        tqdm.write(f"[Phase 1] Epoch {epoch+1:02d}/{phase1_epochs:02d} | Train Concept Loss: {avg_loss_c:.4f} | Val Concept Loss: {avg_val_loss_c:.4f} | Val Concept Balanced Acc: {avg_val_acc_c * 100:.2f}% | TPR: {val_tpr * 100:.2f}% | TNR: {val_tnr * 100:.2f}%")
        
        concepts_list = resolved_config.get("concepts_flat", resolved_config.get("concepts", []))
        
        # struggling concepts는 마지막 epoch이거나 조기종료일 때만 출력하여 로그 노이즈 최소화
        is_last_epoch = (epoch == phase1_epochs - 1)
        es_handler(avg_val_loss_c, model)
        
        # Compute individual balanced accuracies for struggling concepts and logging
        if all_val_probs:
            val_individual_accs = {}
            for c in range(num_concepts_supervised):
                name = concepts_list[c] if c < len(concepts_list) else f"Concept_{c}"
                ind_balanced_acc = val_metrics["individual_balanced_acc"][c].item()
                val_individual_accs[f"val_concept_acc/{name}"] = ind_balanced_acc
                
            if is_last_epoch or es_handler.early_stop:
                sorted_concept_accs = sorted(
                    [(concepts_list[c] if c < len(concepts_list) else f"Concept_{c}", val_individual_accs[f"val_concept_acc/{concepts_list[c] if c < len(concepts_list) else f'Concept_{c}'}"])
                     for c in range(num_concepts_supervised)],
                    key=lambda x: x[1]
                )
                lowest_3 = ", ".join([f"{name}: {acc:.4f}" for name, acc in sorted_concept_accs[:3]])
                tqdm.write(f"  🔍 Final Struggling Concepts (Balanced Acc): {lowest_3}")
            
        if val_vis_data is not None and (is_last_epoch or es_handler.early_stop):
            vis_images, vis_attn = val_vis_data
            num_samples = min(4, vis_images.size(0))
            heatmap_images = generate_concept_heatmaps(
                image_tensor=vis_images[:num_samples],
                attn_weights=vis_attn[:num_samples, :num_concepts_supervised],
                concept_names=concepts_list
            )
            epoch_vis_dir = os.path.join("visualizations", run_name, f"phase1_epoch_{epoch + 1}")
            os.makedirs(epoch_vis_dir, exist_ok=True)
            for idx, img in enumerate(heatmap_images):
                img.save(os.path.join(epoch_vis_dir, f"sample_{idx + 1}.png"))
                
        if scheduler is not None:
            scheduler.step()
            
        if es_handler.early_stop:
            tqdm.write(f"  🛑 Early stopping Phase 1 at Epoch {epoch + 1}. Restoring best Phase 1 weights.")
            model.load_state_dict(es_handler.best_weights)
            break
            
        if args.use_wandb:
            import wandb
            log_dict = {
                "phase1_epoch": epoch + 1,
                "train/concept_loss": avg_loss_c,
                "val/concept_loss": avg_val_loss_c,
                "val/concept_accuracy": avg_val_acc_c,
                "val/concept_tpr": val_tpr,
                "val/concept_tnr": val_tnr
            }
            # wandb에는 개별 정확도를 매 에포크 기록
            if 'val_individual_accs' in locals():
                log_dict.update(val_individual_accs)
            wandb.log(log_dict)

def train_phase2(model, train_loader, val_loader, target_criterion, device, args, config_data, run_name, num_concepts_supervised, resolved_config, num_classes):
    tqdm.write(f"\n{'-'*60}")
    tqdm.write("  🎬 Phase 2: Target Learning (Classifier Head)")
    tqdm.write(f"{'-'*60}")
    
    # 백본과 컨셉 어텐션 가중치 엄격히 동결
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.freeze_supervised_attention()
    model.unfreeze_latent_attention()
    for param in model.classifier_head.parameters():
        param.requires_grad = True
        
    opt_cfg = config_data.get("optimizer", {})
    opt_type = opt_cfg.get("type", "adam").lower()
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    phase2_lr = args.phase2_lr if args.phase2_lr is not None else opt_cfg.get("phase2_lr", opt_cfg.get("head_lr", args.lr))
    
    trainable_params = list(model.classifier_head.parameters())
    if model.num_latent_concepts > 0:
        trainable_params += list(model.latent_attention.parameters())
        
    if opt_type == "adamw":
        optimizer = optim.AdamW(trainable_params, lr=phase2_lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        optimizer = optim.SGD(trainable_params, lr=phase2_lr, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optim.Adam(trainable_params, lr=phase2_lr, weight_decay=weight_decay)
        
    sched_cfg = config_data.get("scheduler", {})
    sched_type = sched_cfg.get("type", "none").lower()
    scheduler = None
    phase2_epochs = args.phase2_epochs if args.phase2_epochs is not None else args.epochs
    
    if sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=sched_cfg.get("eta_min", 1e-6))
    elif sched_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_cfg.get("step_size", 10), gamma=sched_cfg.get("gamma", 0.1))
        
    es_cfg = config_data.get("early_stopping", {})
    es_handler = None
    if es_cfg.get("enabled", False):
        phase2_monitor = args.phase2_monitor if getattr(args, "phase2_monitor", None) is not None else es_cfg.get("phase2_monitor", es_cfg.get("monitor", "val_target_loss"))
        phase2_patience = args.phase2_patience if args.phase2_patience is not None else es_cfg.get("phase2_patience", es_cfg.get("patience", 5))
        min_delta = es_cfg.get("min_delta", 0.0)
        es_handler = EarlyStopping(patience=phase2_patience, min_delta=min_delta, monitor=phase2_monitor)
        tqdm.write(f"  🛑 Phase 2 Early stopping: monitor={phase2_monitor}, patience={phase2_patience}")
        
    for epoch in range(phase2_epochs):
        model.train()
        total_loss_t = 0.0
        total_acc_t = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"  P2 Epoch {epoch+1}/{phase2_epochs}", bar_format="{l_bar}{bar:25}{r_bar}", leave=False)
        for images, _, targets in train_pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            import torch.nn.functional as F
            # 이전 단계에서 학습된 수퍼바이즈드 컨셉 예측값을 그래프 연산 분리하여 추출
            with torch.no_grad():
                features = model.backbone(images)
                supervised_logits, _, supervised_features = model.supervised_attention(features)
                
            optimizer.zero_grad()
            
            # 레이턴트 컨셉은 그래디언트를 흘려주어야 하므로 no_grad 밖에서 계산
            if model.num_latent_concepts > 0:
                latent_logits, _, latent_features = model.latent_attention(features)
                concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
            else:
                concept_logits = supervised_logits
                latent_features = None
                
            concept_probs = model.concept_activation(concept_logits)
            concept_probs_dropout = model.dropout(concept_probs)
            class_logits = model.classifier_head(concept_probs_dropout)
            
            if num_classes == 1:
                loss_t = target_criterion(class_logits, targets)
            else:
                loss_t = target_criterion(class_logits, targets.view(-1).long())
                
            # CBM Latent concept regularization terms
            loss_latent_ortho = torch.tensor(0.0, device=device)
            loss_latent_l1 = torch.tensor(0.0, device=device)
            
            if model.num_latent_concepts > 0 and latent_features is not None:
                # 1. Cosine similarity-based Orthogonal Projection Loss
                # Normalize features along dimension C: shape [B, S, C] and [B, L, C]
                explicit_norm = F.normalize(supervised_features, p=2, dim=-1)
                latent_norm = F.normalize(latent_features, p=2, dim=-1)
                
                # Compute batch cosine similarity matrix: [B, L, S]
                cos_sim = torch.bmm(latent_norm, explicit_norm.transpose(1, 2))
                loss_latent_ortho = (cos_sim ** 2).mean()
                
                # 2. L1 sparsity regularization on latent concept activations
                # Latent activations are the last self.num_latent_concepts columns of concept_probs
                latent_activations = concept_probs[:, num_concepts_supervised:]  # [B, L]
                loss_latent_l1 = latent_activations.abs().mean()
                
                # Aggregate losses
                loss_t = loss_t + (args.lambda_latent_ortho * loss_latent_ortho) + (args.lambda_latent_l1 * loss_latent_l1)
                
            # L1 Lasso Regularization on classifier_head parameters to select high-information concepts
            l1_lambda = getattr(args, "l1_lambda", 0.0)
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.classifier_head.parameters())
                loss_t = loss_t + l1_lambda * l1_norm
                
            loss_t.backward()
            optimizer.step()
            
            total_loss_t += loss_t.item()
            total_acc_t += calculate_accuracy(class_logits.detach(), targets)
            train_pbar.set_postfix(TL=f"{loss_t.item():.4f}")
            
        avg_loss_t = total_loss_t / len(train_loader)
        avg_acc_t = total_acc_t / len(train_loader)
        
        model.eval()
        val_loss_t = 0.0
        val_acc_t = 0.0
        
        with torch.no_grad():
            for val_images, _, val_targets in val_loader:
                val_images = val_images.to(device)
                val_targets = val_targets.to(device)
                
                v_class_logits, _, _ = model(val_images)
                
                if num_classes == 1:
                    v_loss_t = target_criterion(v_class_logits, val_targets)
                else:
                    v_loss_t = target_criterion(v_class_logits, val_targets.view(-1).long())
                    
                val_loss_t += v_loss_t.item()
                val_acc_t += calculate_accuracy(v_class_logits, val_targets)
                
        avg_val_loss_t = val_loss_t / len(val_loader)
        avg_val_acc_t = val_acc_t / len(val_loader)
        
        # 에포크 정보 한 줄 출력 (스크롤 이력 보존)
        tqdm.write(f"[Phase 2] Epoch {epoch+1:02d}/{phase2_epochs:02d} | Train Target Loss: {avg_loss_t:.4f} | Val Target Loss: {avg_val_loss_t:.4f} | Val Target Acc: {avg_val_acc_t * 100:.2f}%")
        
        if scheduler is not None:
            scheduler.step()
            
        if es_handler is not None:
            monitor_target = es_handler.monitor.lower()
            if monitor_target == "val_target_loss" or monitor_target == "val_loss":
                monitor_score = avg_val_loss_t
            elif monitor_target == "val_accuracy" or monitor_target == "val_acc":
                monitor_score = avg_val_acc_t
            else:
                monitor_score = avg_val_loss_t
                
            es_handler(monitor_score, model)
            if es_handler.early_stop:
                tqdm.write(f"  🛑 Early stopping Phase 2 at Epoch {epoch + 1}. Restoring best Phase 2 weights.")
                model.load_state_dict(es_handler.best_weights)
                break
                
        if args.use_wandb:
            import wandb
            log_dict = {
                "phase2_epoch": epoch + 1,
                "train/target_loss": avg_loss_t,
                "val/target_loss": avg_val_loss_t,
                "val/accuracy": avg_val_acc_t
            }
            wandb.log(log_dict)

def train_phase3(model, train_loader, val_loader, target_criterion, concept_criterion, device, args, config_data, run_name, num_concepts_supervised, resolved_config, num_classes):
    tqdm.write(f"\n{'-'*60}")
    tqdm.write("  🎬 Phase 3: Backbone & Classifier Fine-Tuning (Concept Head Frozen)")
    tqdm.write(f"{'-'*60}")
    
    # 1. Freeze Concept Head (supervised + latent attention) — preserve Phase 1 learned attention
    model.freeze_supervised_attention()
    model.freeze_latent_attention()
    tqdm.write("  🔒 Concept Head frozen: supervised & latent attention weights are fixed.")
    
    # 2. Unfreeze only LoRA backbone adapters + classifier head
    model.unfreeze_backbone()    # LoRA-active: only lora_A / lora_B params, pretrained weights stay frozen
    model.unfreeze_classifier()
    tqdm.write("  🔓 Backbone (LoRA adapters) and Classifier Head unfrozen for fine-tuning.")
    
    opt_cfg = config_data.get("optimizer", {})
    opt_type = opt_cfg.get("type", "adam").lower()
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    
    # Use very small learning rate for joint tuning
    phase3_lr = args.phase3_lr if args.phase3_lr is not None else 1e-5
    phase3_epochs = args.phase3_epochs
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_names  = [n for n, p in model.named_parameters() if p.requires_grad]
    tqdm.write(f"  🧠 Trainable parameters in Phase 3: {len(trainable_params)} tensors")
    tqdm.write(f"     └─ Modules: {', '.join(dict.fromkeys(n.split('.')[0] for n in trainable_names))}")
    
    if opt_type == "adamw":
        optimizer = optim.AdamW(trainable_params, lr=phase3_lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        optimizer = optim.SGD(trainable_params, lr=phase3_lr, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optim.Adam(trainable_params, lr=phase3_lr, weight_decay=weight_decay)
        
    sched_cfg = config_data.get("scheduler", {})
    sched_type = sched_cfg.get("type", "none").lower()
    scheduler = None
    
    if sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase3_epochs, eta_min=sched_cfg.get("eta_min", 1e-6))
    elif sched_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_cfg.get("step_size", 10), gamma=sched_cfg.get("gamma", 0.1))
        
    es_cfg = config_data.get("early_stopping", {})
    es_handler = None
    if es_cfg.get("enabled", False):
        phase3_monitor = args.phase3_monitor if getattr(args, "phase3_monitor", None) is not None else "val_target_loss"
        phase3_patience = args.phase3_patience if args.phase3_patience is not None else 5
        min_delta = es_cfg.get("min_delta", 0.0)
        es_handler = EarlyStopping(patience=phase3_patience, min_delta=min_delta, monitor=phase3_monitor)
        tqdm.write(f"  🛑 Phase 3 Early stopping: monitor={phase3_monitor}, patience={phase3_patience}")
        
    for epoch in range(phase3_epochs):
        model.train()
        total_loss_joint = 0.0
        total_loss_t = 0.0
        total_loss_c = 0.0
        total_acc_t = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"  P3 Epoch {epoch+1}/{phase3_epochs}", bar_format="{l_bar}{bar:25}{r_bar}", leave=False)
        for images, concepts, targets in train_pbar:
            images = images.to(device)
            concepts = concepts.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Joint forward pass returning features for orthogonality regularization
            import torch.nn.functional as F
            class_logits, concept_logits, _, supervised_features, latent_features = model(images, return_features=True)
            
            # 1. Target Loss
            if num_classes == 1:
                loss_t = target_criterion(class_logits, targets)
            else:
                loss_t = target_criterion(class_logits, targets.view(-1).long())
                
            # 2. Concept Loss (on supervised concept predictions)
            supervised_logits = concept_logits[:, :num_concepts_supervised]
            loss_c = concept_criterion(supervised_logits, concepts)
            
            # 3. Latent Regularization Losses
            loss_latent_ortho = torch.tensor(0.0, device=device)
            loss_latent_l1 = torch.tensor(0.0, device=device)
            
            concept_probs = model.concept_activation(concept_logits)
            
            if model.num_latent_concepts > 0 and latent_features is not None:
                # Cosine similarity-based Orthogonal Projection Loss
                explicit_norm = F.normalize(supervised_features, p=2, dim=-1)
                latent_norm = F.normalize(latent_features, p=2, dim=-1)
                cos_sim = torch.bmm(latent_norm, explicit_norm.transpose(1, 2))
                loss_latent_ortho = (cos_sim ** 2).mean()
                
                # L1 latent activation sparsity loss
                latent_activations = concept_probs[:, num_concepts_supervised:]
                loss_latent_l1 = latent_activations.abs().mean()
                
            # Joint Loss aggregation: Target Loss + Scaled Concept Loss + Latent Orthogonality + Latent L1 Sparsity
            total_loss = loss_t + (args.lambda_c * loss_c) + (args.lambda_latent_ortho * loss_latent_ortho) + (args.lambda_latent_l1 * loss_latent_l1)
            
            # L1 Lasso Regularization on classifier_head parameters
            l1_lambda = getattr(args, "l1_lambda", 0.0)
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.classifier_head.parameters())
                total_loss = total_loss + l1_lambda * l1_norm
                
            total_loss.backward()
            optimizer.step()
            
            total_loss_joint += total_loss.item()
            total_loss_t += loss_t.item()
            total_loss_c += loss_c.item()
            total_acc_t += calculate_accuracy(class_logits.detach(), targets)
            train_pbar.set_postfix(JL=f"{total_loss.item():.4f}", TL=f"{loss_t.item():.4f}", CL=f"{loss_c.item():.4f}")
            
        avg_loss_joint = total_loss_joint / len(train_loader)
        avg_loss_t = total_loss_t / len(train_loader)
        avg_loss_c = total_loss_c / len(train_loader)
        avg_acc_t = total_acc_t / len(train_loader)
        
        model.eval()
        val_loss_t = 0.0
        val_acc_t = 0.0
        
        with torch.no_grad():
            for val_images, _, val_targets in val_loader:
                val_images = val_images.to(device)
                val_targets = val_targets.to(device)
                
                v_class_logits, _, _ = model(val_images)
                
                if num_classes == 1:
                    v_loss_t = target_criterion(v_class_logits, val_targets)
                else:
                    v_loss_t = target_criterion(v_class_logits, val_targets.view(-1).long())
                    
                val_loss_t += v_loss_t.item()
                val_acc_t += calculate_accuracy(v_class_logits, val_targets)
                
        avg_val_loss_t = val_loss_t / len(val_loader)
        avg_val_acc_t = val_acc_t / len(val_loader)
        
        tqdm.write(f"[Phase 3] Epoch {epoch+1:02d}/{phase3_epochs:02d} | Train Joint Loss: {avg_loss_joint:.4f} | Val Target Loss: {avg_val_loss_t:.4f} | Val Target Acc: {avg_val_acc_t * 100:.2f}%")
        
        if scheduler is not None:
            scheduler.step()
            
        if es_handler is not None:
            monitor_target = es_handler.monitor.lower()
            if monitor_target == "val_target_loss" or monitor_target == "val_loss":
                monitor_score = avg_val_loss_t
            elif monitor_target == "val_accuracy" or monitor_target == "val_acc":
                monitor_score = avg_val_acc_t
            else:
                monitor_score = avg_val_loss_t
                
            es_handler(monitor_score, model)
            if es_handler.early_stop:
                tqdm.write(f"  🛑 Early stopping Phase 3 at Epoch {epoch + 1}. Restoring best Phase 3 weights.")
                model.load_state_dict(es_handler.best_weights)
                break
                
        if args.use_wandb:
            import wandb
            log_dict = {
                "phase3_epoch": epoch + 1,
                "train/joint_loss": avg_loss_joint,
                "train/joint_target_loss": avg_loss_t,
                "train/joint_concept_loss": avg_loss_c,
                "val/joint_target_loss": avg_val_loss_t,
                "val/joint_accuracy": avg_val_acc_t
            }
            wandb.log(log_dict)

def main():
    args, config_data = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.backbone_name}-cbm-{timestamp}"

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  🚀 Training Run: {run_name}")
    tqdm.write(f"  📦 Device: {device}")
    tqdm.write(f"{'='*60}")

    # 1. Dataset & DataLoader Factory Setup
    if args.dataset == 'milk10k':
        dataset_class = MILK10KDataset
    elif args.dataset == 'derm7pt':
        dataset_class = Derm7PtDataset
    elif args.dataset in ['cub', 'cub_200_2011', 'cvpr2016_cub']:
        dataset_class = CUB2011Dataset
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    # Generate default dataset config
    dataset_config = dataset_class.get_default_config()

    if args.concept_config_path:
        dataset_config["concept_config_path"] = args.concept_config_path

    # Apply CLI overrides if present
    if args.concept_cols:
        dataset_config["concepts"] = [c.strip() for c in args.concept_cols.split(',')]
        dataset_config["num_concepts"] = len(dataset_config["concepts"])
        
    if args.num_classes != 1:  # Only override if explicitly customized via CLI
        dataset_config["num_classes"] = args.num_classes

    # Instantiate train and validation datasets
    train_dataset = dataset_class(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        split='train',
        config=dataset_config,
        cache_in_memory=args.cache_in_memory,
        max_cache_size_gb=args.max_cache_size_gb
    )
    val_dataset = dataset_class(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        split='val',
        config=dataset_config,
        cache_in_memory=args.cache_in_memory,
        max_cache_size_gb=args.max_cache_size_gb
    )

    # Use final resolved configuration from dataset instance
    resolved_config = train_dataset.config
    num_concepts_supervised = resolved_config["num_concepts"]
    num_concepts_total = num_concepts_supervised + args.latent_concepts
    num_classes = resolved_config["num_classes"]

    tqdm.write(f"  📂 Dataset: {args.dataset} | Supervised Concepts: {num_concepts_supervised} | Latent Concepts: {args.latent_concepts} | Classes: {num_classes}")
    tqdm.write(f"  📊 Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    
    # Log target class names if available
    target_classes = resolved_config.get("target_classes", [])
    if target_classes:
        tqdm.write(f"  🏷️  Classes: {target_classes}")

    num_workers = args.num_workers
    if getattr(train_dataset, "cache_in_memory", False):
        tqdm.write("  ⚡ In-memory caching enabled: Setting num_workers = 0 to eliminate multiprocessing IPC copy overhead.")
        num_workers = 0

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=args.pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.pin_memory
    )

    # 1c. Extract Concept Grouping metadata for Mutually Exclusive Softmax
    concept_groups_info = None
    if args.use_concept_groups and hasattr(train_dataset, "concept_features_info") and train_dataset.concept_features_info is not None:
        target_groups = None
        if isinstance(args.use_concept_groups, str):
            if args.use_concept_groups.lower() == 'true':
                target_groups = None
            elif args.use_concept_groups.lower() == 'false':
                target_groups = set()
            else:
                target_groups = {name.strip() for name in args.use_concept_groups.split(',')}
        elif isinstance(args.use_concept_groups, list):
            target_groups = {str(name).strip() for name in args.use_concept_groups}
            
        concept_groups_info = []
        grouped_count = 0
        for info in train_dataset.concept_features_info:
            start = info["start_idx"]
            num = info["num_feats"]
            name = info["name"]
            if target_groups is not None and name not in target_groups:
                # Treat as individual sigmoids (group of size 1)
                for i in range(num):
                    concept_groups_info.append((start + i, 1))
            else:
                concept_groups_info.append((start, num))
                if num > 1:
                    grouped_count += 1
        tqdm.write(f"  📂 Group-level Softmax Activation: Configured {grouped_count} mutually exclusive groups out of {len(train_dataset.concept_features_info)} total categories.")
    else:
        tqdm.write("  📂 Group-level Softmax Activation: DISABLED (Sigmoid activation fallback active).")

    # 2. Model Initialization
    tqdm.write(f"  🧠 Model: {args.backbone_type}/{args.backbone_name}")
    tqdm.write(f"  🧬 Concepts - Supervised: {num_concepts_supervised} | Latent: {args.latent_concepts} | Total Bottleneck: {num_concepts_total}")

    # 2a. Build group_mapping for GroupToConceptAttention (if requested)
    group_mapping = None
    num_groups    = None
    if args.use_group_broadcasting:
        concept_info_list = None
        if hasattr(train_dataset, "concept_features_info") and train_dataset.concept_features_info is not None:
            concept_info_list = train_dataset.concept_features_info
        elif args.concept_config_path and os.path.exists(args.concept_config_path):
            try:
                import json
                with open(args.concept_config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                concept_info_list = []
                total_dims = 0
                for name, info in cfg.items():
                    ctype = info.get("type", "numerical")
                    if ctype == "categorical":
                        num_feats = len(info.get("classes", []))
                    else:
                        num_feats = 1
                    concept_info_list.append({
                        "name": name,
                        "num_feats": num_feats
                    })
            except Exception as e:
                tqdm.write(f"  ⚠️ Error loading concept config path in main.py: {e}")
        
        if concept_info_list is not None:
            group_mapping = []
            for group_idx, info in enumerate(concept_info_list):
                num_in_group = info["num_feats"]
                group_mapping.extend([group_idx] * num_in_group)
            num_groups = len(concept_info_list)
            assert len(group_mapping) == num_concepts_supervised, (
                f"group_mapping length {len(group_mapping)} != num_supervised_concepts {num_concepts_supervised}"
                f" (Please check if your concept config JSON matches the supervised concepts dataset)"
            )
            # When using group broadcasting, disable Group Softmax (it conflicts with independent BCE)
            concept_groups_info = None
            tqdm.write(f"  📡 Group Broadcasting: {num_groups} anatomical groups → {num_concepts_supervised} independent BCE classifiers (Group Softmax disabled).")
        else:
            tqdm.write("  ⚠️  use_group_broadcasting=True but concept config information could not be resolved. Falling back to standard attention.")

    model = UniversalFlexibleCBM(
        backbone_type=args.backbone_type,
        backbone_name=args.backbone_name,
        num_supervised_concepts=num_concepts_supervised,
        num_classes=num_classes,
        num_latent_concepts=args.latent_concepts,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        concept_groups_info=concept_groups_info,
        use_cosine_attention=args.use_cosine_attention,
        use_group_broadcasting=args.use_group_broadcasting,
        num_groups=num_groups,
        group_mapping=group_mapping
    )
    
    if args.freeze_backbone:
        model.freeze_backbone()
        tqdm.write("  🔒 Backbone frozen")
        
    if args.freeze_head:
        model.freeze_classifier()
        tqdm.write("  🔒 Classifier head frozen")
        
    if args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            tqdm.write(f"  🔄 Loading pre-trained weights from: {args.resume_checkpoint}")
            loaded_checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
            if isinstance(loaded_checkpoint, dict) and 'state_dict' in loaded_checkpoint:
                state_dict = loaded_checkpoint['state_dict']
            else:
                state_dict = loaded_checkpoint

            # ── State-dict migration: old MHA → new Cosine Attention keys ───────
            def _migrate_state_dict(sd: dict) -> dict:
                """Remap legacy nn.MultiheadAttention keys to new cosine-attention layout."""
                migrated = {}
                for k, v in sd.items():
                    if ".cross_attention.in_proj_weight" in k:
                        prefix = k.replace(".cross_attention.in_proj_weight", "")
                        D = v.shape[0] // 3
                        migrated[f"{prefix}.q_proj.weight"] = v[:D].clone()
                        migrated[f"{prefix}.k_proj.weight"] = v[D:2*D].clone()
                        migrated[f"{prefix}.v_proj.weight"] = v[2*D:].clone()
                    elif ".cross_attention.in_proj_bias" in k:
                        pass  # new projections have bias=False; skip
                    elif ".cross_attention.out_proj.weight" in k:
                        new_k = k.replace(".cross_attention.out_proj.weight", ".out_proj.weight")
                        migrated[new_k] = v
                    elif ".cross_attention.out_proj.bias" in k:
                        pass  # new out_proj has no bias; skip
                    else:
                        migrated[k] = v
                return migrated

            old_mha_keys = {k for k in state_dict if ".cross_attention." in k}
            if old_mha_keys:
                tqdm.write(f"  ⚠️  Detected {len(old_mha_keys)} legacy MHA key(s). Migrating to cosine-attention layout…")
                state_dict = _migrate_state_dict(state_dict)
                tqdm.write("  ✅  State-dict migration complete.")

            try:
                model.load_state_dict(state_dict, strict=True)
                tqdm.write("  ✅ Weights loaded successfully (strict match).")
            except RuntimeError as e:
                tqdm.write(f"  ⚠️ Warning: Strict loading failed. Attempting non-strict load. Error: {e}")
                model.load_state_dict(state_dict, strict=False)
                tqdm.write("  ✅ Weights loaded successfully (non-strict match).")
        else:
            tqdm.write(f"  ❌ Error: Checkpoint path '{args.resume_checkpoint}' does not exist. Starting training from scratch.")
            
    model.to(device)

    # 3. Loss & Optimizer Setup
    if getattr(args, 'use_group_broadcasting', False):
        pos_weights = calculate_pos_weights(train_dataset, num_concepts_supervised).to(device)
        tqdm.write(f"  🎯 Concept Loss: BCEWithLogitsLoss (Group Broadcasting mode) with dynamic pos_weights (first 5 shown): {[f'{w:.2f}' for w in pos_weights[:5].tolist()]}")
        concept_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    elif concept_groups_info is not None:
        # Determine alpha value for Focal Loss if chosen as fallback
        focal_alpha = None
        if args.concept_loss_type == 'focal':
            if isinstance(args.focal_alpha, str) and args.focal_alpha.lower() == 'dynamic':
                pos_weights = calculate_pos_weights(train_dataset, num_concepts_supervised)
                focal_alpha = pos_weights / (1.0 + pos_weights)
                focal_alpha = focal_alpha.to(device)
            elif args.focal_alpha is not None and args.focal_alpha != 'dynamic':
                focal_alpha = float(args.focal_alpha)
                
        tqdm.write(
            f"  🎯 Concept Loss: Mutually Exclusive GroupCrossEntropyLoss ({len(concept_groups_info)} groups, "
            f"lambda_ce={args.lambda_ce}, fallback_loss={args.concept_loss_type})"
        )
        concept_criterion = GroupCrossEntropyLoss(
            groups_info=concept_groups_info,
            lambda_ce=args.lambda_ce,
            loss_type=args.concept_loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=args.focal_gamma
        )
    elif args.concept_loss_type == 'focal':
        if isinstance(args.focal_alpha, str) and args.focal_alpha.lower() == 'dynamic':
            # Dynamically compute per-concept alpha: alpha = pos_weight / (1 + pos_weight)
            pos_weights = calculate_pos_weights(train_dataset, num_concepts_supervised)
            focal_alpha = pos_weights / (1.0 + pos_weights)
            focal_alpha = focal_alpha.to(device)
            tqdm.write(f"  🎯 Concept Loss: Sigmoid Focal Loss with DYNAMIC alpha (first 5 shown): {[f'{a:.4f}' for a in focal_alpha[:5].tolist()]}, gamma={args.focal_gamma}")
            concept_criterion = SigmoidFocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
        else:
            tqdm.write(f"  🎯 Concept Loss: Sigmoid Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
            concept_criterion = SigmoidFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        # Calculate positive weights dynamically for BCEWithLogitsLoss to handle concept sparsity
        pos_weights = calculate_pos_weights(train_dataset, num_concepts_supervised).to(device)
        tqdm.write(f"  ⚖️  Concept Loss: BCEWithLogitsLoss with dynamic pos_weights (first 5 shown): {[f'{w:.2f}' for w in pos_weights[:5].tolist()]}")
        concept_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    if num_classes == 1:
        if args.target_pos_weight != 1.0:
            pos_weight = torch.tensor([args.target_pos_weight], dtype=torch.float32, device=device)
            target_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            target_criterion = nn.BCEWithLogitsLoss()
    else:
        # Multi-class: compute inverse-frequency class weights from training data
        if hasattr(train_dataset, 'df') and not train_dataset.dummy_mode:
            target_col = resolved_config.get("target_col", "diagnosis_idx")
            target_to_idx = getattr(train_dataset, "target_to_idx", None)
            
            counts = [0] * num_classes
            for val in train_dataset.df[target_col].dropna():
                if target_to_idx is not None and val in target_to_idx:
                    idx = target_to_idx[val]
                else:
                    try:
                        idx = int(val)
                    except (ValueError, TypeError):
                        idx = 0
                if 0 <= idx < num_classes:
                    counts[idx] += 1
            
            # Ensure at least 1 count to avoid division by zero
            counts = [max(1, c) for c in counts]
            
            # Inverse-frequency weights normalized so they sum to num_classes
            weights = [1.0 / c for c in counts]
            sum_weights = sum(weights)
            weights = [w / sum_weights * num_classes for w in weights]
            
            class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
            target_criterion = nn.CrossEntropyLoss(weight=class_weight)
        else:
            target_criterion = nn.CrossEntropyLoss()
        
    # 3b. Weights & Biases Initialization
    if args.use_wandb:
        import wandb
        wandb.init(
            project="cbm-pipeline",
            name=run_name,
            config=vars(args)
        )
        tqdm.write(f"  📡 W&B run initialized")

    tqdm.write(f"{'='*60}\n")
    
    # 4. Sequential Training Phases
    # Phase 1: Concept Learning
    train_phase1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        concept_criterion=concept_criterion,
        device=device,
        args=args,
        config_data=config_data,
        run_name=run_name,
        num_concepts_supervised=num_concepts_supervised,
        resolved_config=resolved_config,
        concept_groups_info=concept_groups_info
    )
    
    # Phase 2: Target Learning
    train_phase2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        target_criterion=target_criterion,
        device=device,
        args=args,
        config_data=config_data,
        run_name=run_name,
        num_concepts_supervised=num_concepts_supervised,
        resolved_config=resolved_config,
        num_classes=num_classes
    )
    
    # Phase 3: Joint Parameter Tuning
    if getattr(args, "phase3_epochs", 5) > 0:
        train_phase3(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            target_criterion=target_criterion,
            concept_criterion=concept_criterion,
            device=device,
            args=args,
            config_data=config_data,
            run_name=run_name,
            num_concepts_supervised=num_concepts_supervised,
            resolved_config=resolved_config,
            num_classes=num_classes
        )

    # Save Model Weights
    mode = "frozen_backbone" if args.freeze_backbone else "full"
    save_subdir = os.path.join(args.save_dir, args.backbone_name)
    os.makedirs(save_subdir, exist_ok=True)
    
    if args.save_filename:
        save_filename = args.save_filename
    else:
        simple_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        save_filename = f"{args.dataset}_{args.backbone_name}_latent{args.latent_concepts}_{simple_timestamp}.pt"
    save_path = os.path.join(save_subdir, save_filename)
    
    # Pack model state_dict along with full training configs for absolute reproducibility & lineage tracking
    checkpoint = {
        'state_dict': model.state_dict(),
        'config': config_data,
        'args': vars(args)
    }
    torch.save(checkpoint, save_path)
    
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  ✅ Training complete!")
    tqdm.write(f"  💾 Weights saved: {save_path}")
    tqdm.write(f"  🖼️  Heatmaps saved: visualizations/{run_name}/")
    tqdm.write(f"{'='*60}")

    if args.use_wandb:
        wandb.finish()

    # Automatically launch Gradio app using subprocess to execute app.py
    if args.run_app:
        tqdm.write(f"\n🚀 Launching Gradio inference application automatically...")
        import subprocess
        import sys
        
        # Build command dynamically matching the trained model parameters
        cmd = [
            sys.executable, "app.py",
            "--checkpoint", save_path,
            "--concept_config_path", args.concept_config_path or "data/MILK10K/concept_config.json",
            "--backbone_type", args.backbone_type,
            "--backbone_name", args.backbone_name,
            "--num_classes", str(num_classes)
        ]
        # Include latent_concepts if greater than zero
        if args.latent_concepts > 0:
            cmd.extend(["--latent_concepts", str(args.latent_concepts)])
            
        tqdm.write(f"  Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            tqdm.write("\n👋 Gradio app stopped by user.")

if __name__ == "__main__":
    main()
