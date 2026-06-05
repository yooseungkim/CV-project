import os
import copy
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Subset
from tqdm import tqdm

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

def get_dataset_choices():
    data_dir = 'data'
    default_choices = ['milk10k', 'derm7pt', 'cub', 'chexpert']
    if not os.path.exists(data_dir):
        return default_choices
    choices = []
    for item in os.listdir(data_dir):
        path = os.path.join(data_dir, item)
        if os.path.isdir(path) and not item.startswith('.'):
            choices.append(item.lower())
    return list(set(choices + default_choices))

def unwrap_subset(dataset):
    """Return the base dataset and absolute row indices for nested torch Subsets."""
    indices = None
    current_dataset = dataset
    while isinstance(current_dataset, Subset):
        current_indices = [int(i) for i in current_dataset.indices]
        if indices is None:
            indices = current_indices
        else:
            indices = [current_indices[i] for i in indices]
        current_dataset = current_dataset.dataset
    return current_dataset, indices

def calculate_pos_weights(dataset, num_concepts_supervised):
    """Calculates the ratio of negative to positive samples for each concept to balance BCE loss."""
    import pandas as pd
    num_samples = len(dataset)
    if num_samples == 0:
        return torch.ones(num_concepts_supervised)
    base_dataset, subset_indices = unwrap_subset(dataset)
        
    # Attempt to use cached concepts if available
    if subset_indices is None and getattr(base_dataset, "_cache_populated", False) and base_dataset._cache is not None:
        concepts = torch.stack([sample[1][:num_concepts_supervised] for sample in base_dataset._cache], dim=0)
    elif hasattr(base_dataset, "concept_matrix") and base_dataset.concept_matrix is not None:
        # CUB Dataset
        df = base_dataset.df.iloc[subset_indices] if subset_indices is not None else base_dataset.df
        image_idxs = df['image_idx'].values
        concepts = torch.tensor(base_dataset.concept_matrix[image_idxs, :num_concepts_supervised], dtype=torch.float32)
    elif hasattr(base_dataset, "concepts_data"):
        indices = subset_indices if subset_indices is not None else list(range(len(base_dataset)))
        concepts = torch.tensor(base_dataset.concepts_data[indices, :num_concepts_supervised], dtype=torch.float32)
    elif hasattr(base_dataset, "df") and not base_dataset.df.empty:
        # MILK10K or Derm7pt
        df = base_dataset.df.iloc[subset_indices] if subset_indices is not None else base_dataset.df
        concepts_list = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            if base_dataset.concept_features_info is not None:
                concept_vals = []
                for info in base_dataset.concept_features_info:
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
                concept_vals = [float(row.get(col, 0.0)) for col in base_dataset.concept_cols]
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
    pos_weight = torch.clamp(pos_weight, min=0.1, max=100.0)
    return pos_weight

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


def calculate_class_balanced_weights(dataset, num_concepts_supervised, beta=0.999):
    """
    Calculates Class-Balanced loss weights for each concept based on the effective number of samples.
    Ref: Class-Balanced Loss Based on Effective Number of Samples (CVPR 2019)
    """
    import pandas as pd
    num_samples = len(dataset)
    if num_samples == 0:
        return torch.ones(num_concepts_supervised), torch.ones(num_concepts_supervised)
    base_dataset, subset_indices = unwrap_subset(dataset)
        
    if subset_indices is None and getattr(base_dataset, "_cache_populated", False) and base_dataset._cache is not None:
        concepts = torch.stack([sample[1][:num_concepts_supervised] for sample in base_dataset._cache], dim=0)
    elif hasattr(base_dataset, "concept_matrix") and base_dataset.concept_matrix is not None:
        df = base_dataset.df.iloc[subset_indices] if subset_indices is not None else base_dataset.df
        image_idxs = df['image_idx'].values
        concepts = torch.tensor(base_dataset.concept_matrix[image_idxs, :num_concepts_supervised], dtype=torch.float32)
    elif hasattr(base_dataset, "concepts_data"):
        indices = subset_indices if subset_indices is not None else list(range(len(base_dataset)))
        concepts = torch.tensor(base_dataset.concepts_data[indices, :num_concepts_supervised], dtype=torch.float32)
    elif hasattr(base_dataset, "df") and not base_dataset.df.empty:
        df = base_dataset.df.iloc[subset_indices] if subset_indices is not None else base_dataset.df
        concepts_list = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            if base_dataset.concept_features_info is not None:
                concept_vals = []
                for info in base_dataset.concept_features_info:
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
                concept_vals = [float(row.get(col, 0.0)) for col in base_dataset.concept_cols]
                concepts_list.append(torch.tensor(concept_vals, dtype=torch.float32))
        concepts = torch.stack(concepts_list, dim=0)[:, :num_concepts_supervised]
    else:
        concepts = torch.zeros((num_samples, num_concepts_supervised))

    # Calculate positives and negatives counts per concept
    pos_counts = (concepts > 0.5).sum(dim=0).float()
    neg_counts = (concepts <= 0.5).sum(dim=0).float()

    # CB-Loss Formula: (1 - beta) / (1 - beta^n)
    pos_counts_safe = torch.clamp(pos_counts, min=0.0)
    neg_counts_safe = torch.clamp(neg_counts, min=0.0)
    
    w_pos = (1.0 - beta) / (1.0 - torch.pow(beta, pos_counts_safe) + 1e-8)
    w_neg = (1.0 - beta) / (1.0 - torch.pow(beta, neg_counts_safe) + 1e-8)
    
    # If counts are 0, default to weight of 1.0
    w_pos = torch.where(pos_counts > 0, w_pos, torch.ones_like(w_pos))
    w_neg = torch.where(neg_counts > 0, w_neg, torch.ones_like(w_neg))

    # Normalize weights so that w_pos + w_neg = 2.0 for each concept to preserve loss scale
    sum_w = w_pos + w_neg
    sum_w = torch.where(sum_w > 0, sum_w, torch.ones_like(sum_w) * 2.0)
    
    cb_pos_weight = 2.0 * w_pos / sum_w
    cb_neg_weight = 2.0 * w_neg / sum_w

    return cb_pos_weight, cb_neg_weight
