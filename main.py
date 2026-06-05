import argparse
import os
import datetime
import copy
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.milk10k import MILK10KDataset
from src.data.derm7pt import Derm7PtDataset
from src.data.cub import CUB2011Dataset
from src.data.chexpert import CheXpertDataset
from src.models.cbm_factory import UniversalFlexibleCBM
from src.utils.metrics import calculate_accuracy, calculate_concept_accuracy, calculate_concept_metrics, find_optimal_concept_thresholds
from src.utils.visualization import generate_concept_heatmaps

# Modularized utility, loss, and training loop imports
from src.utils.helpers import str2bool, str_or_float, str_or_bool, calculate_pos_weights, get_dataset_choices, unwrap_subset
from src.utils.losses import SigmoidFocalLoss, AsymmetricLossWithWeight, GroupCrossEntropyLoss
from src.utils.train_loops import PHASE_MONITORS, train_phase1, train_phase2, train_phase3, validate_monitor_name
from src.utils.concept_bias import split_for_calibration, learn_concept_bias

# ANSI terminal colors for highlighting
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"


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
        tqdm.write(f"  {BOLD}{BLUE}[Config]{RESET} Loaded config file: {temp_args.config_path}")

    flat_defaults = {}

    # backbone
    bb_cfg = config_data.get("backbone", {})
    if "backbone_type" in bb_cfg: flat_defaults["backbone_type"] = bb_cfg["backbone_type"]
    if "backbone_name" in bb_cfg: flat_defaults["backbone_name"] = bb_cfg["backbone_name"]
    if "backbone_train_mode" in bb_cfg: flat_defaults["backbone_train_mode"] = bb_cfg["backbone_train_mode"]
    if "freeze_backbone" in bb_cfg: flat_defaults["freeze_backbone"] = bb_cfg["freeze_backbone"]
    if "freeze_head" in bb_cfg: flat_defaults["freeze_head"] = bb_cfg["freeze_head"]
    if "use_lora" in bb_cfg: flat_defaults["use_lora"] = bb_cfg["use_lora"]
    if "lora_r" in bb_cfg: flat_defaults["lora_r"] = bb_cfg["lora_r"]
    if "lora_alpha" in bb_cfg: flat_defaults["lora_alpha"] = bb_cfg["lora_alpha"]
    if "use_cosine_attention" in bb_cfg: flat_defaults["use_cosine_attention"] = bb_cfg["use_cosine_attention"]
    if "use_group_broadcasting" in bb_cfg: flat_defaults["use_group_broadcasting"] = bb_cfg["use_group_broadcasting"]
    # (use_dino_mask and dino_mask_threshold flat defaults removed)

    # Backward-compatible interpretation for older config files.
    if "backbone_train_mode" not in flat_defaults:
        if flat_defaults.get("freeze_backbone", False):
            flat_defaults["backbone_train_mode"] = "frozen"
        elif flat_defaults.get("use_lora", False):
            flat_defaults["backbone_train_mode"] = "lora"
        else:
            flat_defaults["backbone_train_mode"] = "full"

    # dataset
    ds_cfg = config_data.get("dataset", {})
    if "dataset" in ds_cfg: flat_defaults["dataset"] = ds_cfg["dataset"]
    if "csv_path" in ds_cfg: flat_defaults["csv_path"] = ds_cfg["csv_path"]
    if "image_dir" in ds_cfg: flat_defaults["image_dir"] = ds_cfg["image_dir"]
    if "concept_config_path" in ds_cfg: flat_defaults["concept_config_path"] = ds_cfg["concept_config_path"]
    if "use_concept_groups" in ds_cfg: flat_defaults["use_concept_groups"] = ds_cfg["use_concept_groups"]
    if "filter_rare_concepts" in ds_cfg: flat_defaults["filter_rare_concepts"] = ds_cfg["filter_rare_concepts"]
    if "use_paper_preprocessing" in ds_cfg: flat_defaults["use_paper_preprocessing"] = ds_cfg["use_paper_preprocessing"]
    if "use_multimodal" in ds_cfg: flat_defaults["use_multimodal"] = ds_cfg["use_multimodal"]
    if "policy" in ds_cfg: flat_defaults["policy"] = ds_cfg["policy"]
    if "subset_ratio" in ds_cfg: flat_defaults["subset_ratio"] = ds_cfg["subset_ratio"]

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
    if "run_eval" in tr_cfg: flat_defaults["run_eval"] = tr_cfg["run_eval"]
    if "phase1_epochs" in tr_cfg: flat_defaults["phase1_epochs"] = tr_cfg["phase1_epochs"]
    if "phase2_epochs" in tr_cfg: flat_defaults["phase2_epochs"] = tr_cfg["phase2_epochs"]
    if "phase2_dropout" in tr_cfg: flat_defaults["phase2_dropout"] = tr_cfg["phase2_dropout"]
    if "use_dynamic_threshold" in tr_cfg: flat_defaults["use_dynamic_threshold"] = tr_cfg["use_dynamic_threshold"]
    if "phase2_scheduled_sampling" in tr_cfg: flat_defaults["phase2_scheduled_sampling"] = tr_cfg["phase2_scheduled_sampling"]
    if "scheduled_sampling_prob" in tr_cfg: flat_defaults["scheduled_sampling_prob"] = tr_cfg["scheduled_sampling_prob"]
    if "scheduled_sampling_epsilon" in tr_cfg: flat_defaults["scheduled_sampling_epsilon"] = tr_cfg["scheduled_sampling_epsilon"]
    if "phase1_label_smoothing" in tr_cfg: flat_defaults["phase1_label_smoothing"] = tr_cfg["phase1_label_smoothing"]
    if "use_nam_head" in tr_cfg: flat_defaults["use_nam_head"] = tr_cfg["use_nam_head"]
    if "nam_hidden_dim" in tr_cfg: flat_defaults["nam_hidden_dim"] = tr_cfg["nam_hidden_dim"]
    if "use_gated_nam" in tr_cfg: flat_defaults["use_gated_nam"] = tr_cfg["use_gated_nam"]
    if "use_pairwise_nam" in tr_cfg: flat_defaults["use_pairwise_nam"] = tr_cfg["use_pairwise_nam"]
    if "use_probabilistic_cbm" in tr_cfg: flat_defaults["use_probabilistic_cbm"] = tr_cfg["use_probabilistic_cbm"]
    if "pcbm_beta" in tr_cfg: flat_defaults["pcbm_beta"] = tr_cfg["pcbm_beta"]
    if "pcbm_beta_warmup_epochs" in tr_cfg: flat_defaults["pcbm_beta_warmup_epochs"] = tr_cfg["pcbm_beta_warmup_epochs"]
    if "pcbm_beta_anneal_epochs" in tr_cfg: flat_defaults["pcbm_beta_anneal_epochs"] = tr_cfg["pcbm_beta_anneal_epochs"]
    if "pcbm_beta_min" in tr_cfg: flat_defaults["pcbm_beta_min"] = tr_cfg["pcbm_beta_min"]
    if "pcbm_asymmetric_kl_weight" in tr_cfg: flat_defaults["pcbm_asymmetric_kl_weight"] = tr_cfg["pcbm_asymmetric_kl_weight"]
    if "use_concept_attention" in bb_cfg: flat_defaults["use_concept_attention"] = bb_cfg["use_concept_attention"]
    if "l1_lambda_gate" in tr_cfg: flat_defaults["l1_lambda_gate"] = tr_cfg["l1_lambda_gate"]
    if "latent_penalty_scale" in tr_cfg: flat_defaults["latent_penalty_scale"] = tr_cfg["latent_penalty_scale"]
    if "intervention_prob" in tr_cfg: flat_defaults["intervention_prob"] = tr_cfg["intervention_prob"]

    if "weight_decay_nam" in tr_cfg: flat_defaults["weight_decay_nam"] = tr_cfg["weight_decay_nam"]
    if "use_cb_loss" in tr_cfg: flat_defaults["use_cb_loss"] = tr_cfg["use_cb_loss"]
    if "cb_beta" in tr_cfg: flat_defaults["cb_beta"] = tr_cfg["cb_beta"]

    # optimizer basic parameter
    opt_cfg = config_data.get("optimizer", {})
    if "lr" in opt_cfg: flat_defaults["lr"] = opt_cfg["lr"]
    if "phase1_lr" in opt_cfg: flat_defaults["phase1_lr"] = opt_cfg["phase1_lr"]
    if "phase2_lr" in opt_cfg: flat_defaults["phase2_lr"] = opt_cfg["phase2_lr"]
    if "concept_loss_type" in opt_cfg: flat_defaults["concept_loss_type"] = opt_cfg["concept_loss_type"]
    if "focal_alpha" in opt_cfg: flat_defaults["focal_alpha"] = opt_cfg["focal_alpha"]
    if "focal_gamma" in opt_cfg: flat_defaults["focal_gamma"] = opt_cfg["focal_gamma"]
    if "ortho_lambda" in opt_cfg: flat_defaults["ortho_lambda"] = opt_cfg["ortho_lambda"]
    if "asl_gamma_pos" in opt_cfg: flat_defaults["asl_gamma_pos"] = opt_cfg["asl_gamma_pos"]
    if "asl_gamma_neg" in opt_cfg: flat_defaults["asl_gamma_neg"] = opt_cfg["asl_gamma_neg"]
    if "asl_alpha_pos" in opt_cfg: flat_defaults["asl_alpha_pos"] = opt_cfg["asl_alpha_pos"]
    if "asl_clip" in opt_cfg: flat_defaults["asl_clip"] = opt_cfg["asl_clip"]
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
    if "l1_warmup_epochs" in tr_cfg: flat_defaults["l1_warmup_epochs"] = tr_cfg["l1_warmup_epochs"]
    if "phase3_epochs" in tr_cfg: flat_defaults["phase3_epochs"] = tr_cfg["phase3_epochs"]
    if "phase3_lr" in opt_cfg: flat_defaults["phase3_lr"] = opt_cfg["phase3_lr"]

    # Stage 2: Create full parser with dynamic defaults
    parser = argparse.ArgumentParser(description="Train a Modular CBM")
    choices = get_dataset_choices()

    parser.add_argument('--config_path', type=str, default=None, help="Path to config JSON file")
    parser.add_argument('--dataset', type=str, default=flat_defaults.get('dataset', 'milk10k'), choices=choices)
    parser.add_argument('--csv_path', type=str, default=flat_defaults.get('csv_path', None))
    parser.add_argument('--image_dir', type=str, default=flat_defaults.get('image_dir', None))
    parser.add_argument('--policy', type=str, default=flat_defaults.get('policy', 'u-ones'), choices=['u-ones', 'u-zeros'], help="CheXpert uncertainty policy")
    parser.add_argument('--subset_ratio', type=float, default=flat_defaults.get('subset_ratio', None), help="Fraction of dataset to load")
    parser.add_argument('--backbone_type', type=str, default=flat_defaults.get('backbone_type', 'timm'), choices=['timm', 'clip'])
    parser.add_argument('--backbone_name', type=str, default=flat_defaults.get('backbone_name', 'resnet50'))
    parser.add_argument('--backbone_train_mode', type=str, default=flat_defaults.get('backbone_train_mode', 'full'), choices=['frozen', 'lora', 'full'], help="Backbone training mode for Phase 1 and Phase 3")
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
    parser.add_argument('--phase2_dropout', type=float, default=flat_defaults.get('phase2_dropout', None), help="Dropout probability for Phase 2 (Target Learning)")
    parser.add_argument('--phase2_scheduled_sampling', type=str2bool, default=flat_defaults.get('phase2_scheduled_sampling', False), help="Enable Scheduled Sampling (Noise Injection) for Phase 2 Classifier training")
    parser.add_argument('--scheduled_sampling_prob', type=float, default=flat_defaults.get('scheduled_sampling_prob', 0.3), help="Probability to replace prediction with GT during Phase 2 scheduled sampling")
    parser.add_argument('--scheduled_sampling_epsilon', type=float, default=flat_defaults.get('scheduled_sampling_epsilon', 0.05), help="Epsilon value for soft GT probabilities in Phase 2 scheduled sampling")
    parser.add_argument('--intervention_prob', type=float, default=flat_defaults.get('intervention_prob', 0.0), help="Probability of applying scheduled sampling concept noise in Phase 3")
    parser.add_argument('--phase1_label_smoothing', type=float, default=flat_defaults.get('phase1_label_smoothing', 0.05), help="Epsilon parameter for Phase 1 concept target label smoothing (0.0 to disable)")
    parser.add_argument('--phase3_lr', type=float, default=flat_defaults.get('phase3_lr', 1e-5), help="Learning rate for Phase 3 joint fine-tuning")
    parser.add_argument('--phase1_patience', type=int, default=flat_defaults.get('phase1_patience', None), help="Early stopping patience for Phase 1")
    parser.add_argument('--phase2_patience', type=int, default=flat_defaults.get('phase2_patience', None), help="Early stopping patience for Phase 2")
    parser.add_argument('--phase3_patience', type=int, default=flat_defaults.get('phase3_patience', 5), help="Early stopping patience for Phase 3")
    parser.add_argument('--phase1_monitor', type=str, choices=PHASE_MONITORS["phase1"], default=flat_defaults.get('phase1_monitor', 'val_concept_acc'), help="Exact Phase 1 early stopping monitor name")
    parser.add_argument('--phase2_monitor', type=str, choices=PHASE_MONITORS["phase2"], default=flat_defaults.get('phase2_monitor', 'val_target_loss'), help="Exact Phase 2 early stopping monitor name")
    parser.add_argument('--phase3_monitor', type=str, choices=PHASE_MONITORS["phase3"], default=flat_defaults.get('phase3_monitor', 'val_target_loss'), help="Exact Phase 3 early stopping monitor name")
    parser.add_argument('--phase3_epochs', type=int, default=flat_defaults.get('phase3_epochs', 5), help="Number of epochs for Phase 3 (Joint Parameter Tuning)")
    parser.add_argument('--lambda_ce', type=float, default=flat_defaults.get('lambda_ce', 0.1), help="Scaling factor for Softmax Cross-Entropy loss to balance gradient scale against Focal/BCE loss")
    parser.add_argument('--concept_loss_type', type=str, default=flat_defaults.get('concept_loss_type', 'focal'), choices=['focal', 'bce', 'asl'], help="Concept loss function type")
    parser.add_argument('--focal_alpha', type=str_or_float, default=flat_defaults.get('focal_alpha', 'dynamic'), help="Alpha parameter for Focal Loss (float or 'dynamic')")
    parser.add_argument('--focal_gamma', type=float, default=flat_defaults.get('focal_gamma', 2.0), help="Gamma parameter for Focal Loss")
    parser.add_argument('--asl_gamma_pos', type=float, default=flat_defaults.get('asl_gamma_pos', 0.0), help="ASL: gamma for positive samples (0=no decay)")
    parser.add_argument('--asl_gamma_neg', type=float, default=flat_defaults.get('asl_gamma_neg', 4.0), help="ASL: gamma for negative samples")
    parser.add_argument('--asl_alpha_pos', type=float, default=flat_defaults.get('asl_alpha_pos', 1.2), help="ASL: static weight for positive samples")
    parser.add_argument('--asl_clip', type=float, default=flat_defaults.get('asl_clip', 0.05), help="ASL: asymmetric clipping threshold")
    parser.add_argument('--ortho_lambda', type=float, default=flat_defaults.get('ortho_lambda', 0.05), help="Orthogonality regularization loss multiplier for attention map separation")
    parser.add_argument('--l1_lambda', type=float, default=flat_defaults.get('l1_lambda', 0.0), help="L1 Lasso regularization multiplier for Phase 2 classifier")
    parser.add_argument('--l1_warmup_epochs', type=int, default=flat_defaults.get('l1_warmup_epochs', 5), help="Number of warmup epochs for L1 sparsity regularization in Phase 2")
    parser.add_argument('--lambda_latent_ortho', type=float, default=flat_defaults.get('lambda_latent_ortho', 0.1), help="Orthogonal latent projection loss weight")
    parser.add_argument('--lambda_latent_l1', type=float, default=flat_defaults.get('lambda_latent_l1', 0.01), help="L1 latent activation sparsity loss weight")
    parser.add_argument('--batch_size', type=int, default=flat_defaults.get('batch_size', 16))
    parser.add_argument('--lr', type=float, default=flat_defaults.get('lr', 1e-3))
    parser.add_argument('--lambda_c', type=float, default=flat_defaults.get('lambda_c', 1.0))
    parser.add_argument('--target_pos_weight', type=float, default=flat_defaults.get('target_pos_weight', 1.0))
    parser.add_argument('--num_workers', type=int, default=flat_defaults.get('num_workers', 4))
    parser.add_argument('--pin_memory', type=str2bool, default=flat_defaults.get('pin_memory', True))
    parser.add_argument('--freeze_head', action='store_true', default=flat_defaults.get('freeze_head', False))
    parser.add_argument('--use_concept_groups', type=str_or_bool, default=flat_defaults.get('use_concept_groups', True), help="Toggle Group-level Softmax and GroupCrossEntropyLoss (True/False, or comma-separated list of concept names to group)")
    parser.add_argument('--filter_rare_concepts', type=str2bool, default=flat_defaults.get('filter_rare_concepts', False), help="Filter out concepts with occurrence frequency < 1%%")
    parser.add_argument('--use_paper_preprocessing', type=str2bool, default=flat_defaults.get('use_paper_preprocessing', False), help="Align preprocessing (majority voting, sparseness filter, 80-20 train-val split) with the paper")
    parser.add_argument('--lora_r', type=int, default=flat_defaults.get('lora_r', 8), help="LoRA Rank parameter r")
    parser.add_argument('--lora_alpha', type=float, default=flat_defaults.get('lora_alpha', 16.0), help="LoRA scaling parameter alpha")
    parser.add_argument('--use_cosine_attention', type=str2bool, default=flat_defaults.get('use_cosine_attention', False), help="Use L2-normalized Cosine Attention instead of standard MultiheadAttention (suppresses DINOv2 border-patch outliers)")
    parser.add_argument('--use_group_broadcasting', type=str2bool, default=flat_defaults.get('use_group_broadcasting', False), help="Use GroupToConceptAttention: group queries → independent BCE classifiers based on concept_config (fixes TPR/TNR collapse from Group Softmax)")
    # (use_dino_mask and dino_mask_threshold parser arguments removed)
    parser.add_argument('--use_dynamic_threshold', type=str2bool, default=flat_defaults.get('use_dynamic_threshold', True), help="Optimize validation concept decision thresholds via Youden's J statistic")
    parser.add_argument('--use_wandb', type=str2bool, default=flat_defaults.get('use_wandb', True))
    parser.add_argument('--save_dir', type=str, default=flat_defaults.get('save_dir', 'checkpoints'))
    parser.add_argument('--cache_in_memory', type=str2bool, default=flat_defaults.get('cache_in_memory', False))
    parser.add_argument('--max_cache_size_gb', type=float, default=flat_defaults.get('max_cache_size_gb', 10.0))
    parser.add_argument('--resume_checkpoint', type=str, default=flat_defaults.get('resume_checkpoint', None), help="Path to checkpoint .pth to resume or fine-tune from")
    parser.add_argument('--save_filename', type=str, default=None, help="Custom filename to save the final weights")
    parser.add_argument('--run_app', type=str2bool, default=flat_defaults.get('run_app', True), help="Automatically launch Gradio app after training finishes")
    parser.add_argument('--run_eval', type=str2bool, default=flat_defaults.get('run_eval', False), help="Automatically run evaluation (eval_cbm.py) after training finishes")
    parser.add_argument('--use_nam_head', type=str2bool, default=flat_defaults.get('use_nam_head', False), help="Use GatedSparseNAMHead instead of standard nn.Linear classifier head")
    parser.add_argument('--nam_hidden_dim', type=int, default=flat_defaults.get('nam_hidden_dim', 64), help="Hidden dimension for GatedSparseNAMHead subnetworks")
    parser.add_argument('--use_gated_nam', type=str2bool, default=flat_defaults.get('use_gated_nam', False), help="Activate Gated Sparse NAM head")
    parser.add_argument('--use_pairwise_nam', type=str2bool, default=flat_defaults.get('use_pairwise_nam', False), help="Activate Pairwise Interaction NAM^2 head")
    parser.add_argument('--use_probabilistic_cbm', type=str2bool, default=flat_defaults.get('use_probabilistic_cbm', False), help="Convert Concept Extractor to Probabilistic")
    parser.add_argument('--pcbm_beta', type=float, default=flat_defaults.get('pcbm_beta', 0.001), help="PCBM target KL Divergence loss weight beta")
    parser.add_argument('--pcbm_beta_warmup_epochs', type=int, default=flat_defaults.get('pcbm_beta_warmup_epochs', 10), help="PCBM epochs for KL beta warmup (beta=0.0 during warmup)")
    parser.add_argument('--pcbm_beta_anneal_epochs', type=int, default=flat_defaults.get('pcbm_beta_anneal_epochs', 10), help="PCBM epochs for KL beta annealing/ramp-up")
    parser.add_argument('--pcbm_beta_min', type=float, default=flat_defaults.get('pcbm_beta_min', 0.0001), help="PCBM starting beta value after warmup")
    parser.add_argument('--pcbm_asymmetric_kl_weight', type=float, default=flat_defaults.get('pcbm_asymmetric_kl_weight', 0.1), help="PCBM asymmetric KL weight multiplier for positive samples (0.0 to disable positive KL penalty)")
    parser.add_argument('--use_concept_attention', type=str2bool, default=flat_defaults.get('use_concept_attention', False), help="Activate Patch token-based Cross-Attention")
    parser.add_argument('--l1_lambda_gate', type=float, default=flat_defaults.get('l1_lambda_gate', 0.01), help="L1 Regularization strength for Gating parameters")
    parser.add_argument('--latent_penalty_scale', type=float, default=flat_defaults.get('latent_penalty_scale', 1.0), help="Multiplier for L1 penalty on latent concept gates")

    parser.add_argument('--weight_decay_nam', type=float, default=flat_defaults.get('weight_decay_nam', 1e-2), help="L2 penalty for NAM subnetworks smoothing")
    parser.add_argument('--use_cb_loss', type=str2bool, default=flat_defaults.get('use_cb_loss', True), help="Use Class-Balanced Loss weighting based on CVPR 2019")
    parser.add_argument('--cb_beta', type=float, default=flat_defaults.get('cb_beta', 0.999), help="Beta parameter for Class-Balanced Loss weighting")

    args = parser.parse_args()
    for phase in ("phase1", "phase2", "phase3"):
        validate_monitor_name(phase, getattr(args, f"{phase}_monitor"))
    args.use_lora = (args.backbone_train_mode == "lora")
    args.freeze_backbone = (args.backbone_train_mode == "frozen")
    return args, config_data

def main():
    args, config_data = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.backbone_name}-cbm-{timestamp}"

    tqdm.write(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    tqdm.write(f"  {BOLD}{CYAN}[Training Run]{RESET} {run_name}")
    tqdm.write(f"  {BOLD}{CYAN}[Device]{RESET} {device}")
    tqdm.write(f"{BOLD}{CYAN}{'='*60}{RESET}")

    # 1. Dataset & DataLoader Factory Setup
    if args.dataset == 'milk10k':
        dataset_class = MILK10KDataset
    elif args.dataset == 'derm7pt':
        dataset_class = Derm7PtDataset
    elif args.dataset in ['cub', 'cub_200_2011', 'cvpr2016_cub']:
        dataset_class = CUB2011Dataset
    elif args.dataset == 'chexpert':
        dataset_class = CheXpertDataset
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    # Generate default dataset config
    dataset_config = dataset_class.get_default_config()

    if args.concept_config_path:
        dataset_config["concept_config_path"] = args.concept_config_path

    dataset_config["filter_rare_concepts"] = getattr(args, "filter_rare_concepts", False)
    dataset_config["use_paper_preprocessing"] = getattr(args, "use_paper_preprocessing", False)
    dataset_config["policy"] = getattr(args, "policy", "u-ones")
    dataset_config["subset_ratio"] = getattr(args, "subset_ratio", None)

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

    calibration_cfg = config_data.get("calibration", {}) if isinstance(config_data, dict) else {}
    calibration_for = calibration_cfg.get("for_what", [])
    if isinstance(calibration_for, str):
        calibration_for = [calibration_for]
    use_concept_bias_calibration = "learn_concept_bias" in calibration_for
    calibration_dataset = None

    if use_concept_bias_calibration:
        source_split = calibration_cfg.get("source_split", "train")
        if source_split != "train":
            raise ValueError("calibration.source_split currently supports only 'train'")
        calibration_ratio = float(calibration_cfg.get("ratio", 0.1))
        calibration_seed = int(calibration_cfg.get("seed", 42))
        calibration_source_dataset = dataset_class(
            csv_path=args.csv_path,
            image_dir=args.image_dir,
            split='train',
            config=dataset_config,
            transform=getattr(val_dataset, "transform", None),
            cache_in_memory=False,
            max_cache_size_gb=args.max_cache_size_gb
        )
        train_dataset, calibration_dataset = split_for_calibration(
            train_dataset,
            calibration_source_dataset,
            ratio=calibration_ratio,
            seed=calibration_seed
        )

    # Use final resolved configuration from dataset instance
    base_train_dataset, _ = unwrap_subset(train_dataset)
    resolved_config = base_train_dataset.config
    num_concepts_supervised = resolved_config["num_concepts"]
    num_concepts_total = num_concepts_supervised + args.latent_concepts
    num_classes = resolved_config["num_classes"]

    tqdm.write(f"  {BOLD}{BLUE}[Dataset]{RESET} {args.dataset} | Supervised Concepts: {num_concepts_supervised} | Latent Concepts: {args.latent_concepts} | Classes: {num_classes}")
    sample_msg = f"  {BOLD}{BLUE}[Samples]{RESET} Train: {len(train_dataset)} | Val: {len(val_dataset)}"
    if calibration_dataset is not None:
        sample_msg += f" | Calibration: {len(calibration_dataset)}"
    tqdm.write(sample_msg)

    # Log target class names if available
    target_classes = resolved_config.get("target_classes", [])
    if target_classes:
        tqdm.write(f"  {BOLD}{BLUE}[Classes]{RESET} {target_classes}")

    num_workers = args.num_workers
    base_for_cache, _ = unwrap_subset(train_dataset)
    if getattr(base_for_cache, "cache_in_memory", False):
        tqdm.write(f"  {BOLD}{GREEN}[Cache]{RESET} In-memory caching enabled: Setting num_workers = 0 to eliminate multiprocessing IPC copy overhead.")
        num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(num_workers > 0)
    )
    calibration_loader = None
    if calibration_dataset is not None:
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=(num_workers > 0)
        )

    # 1c. Extract Concept Grouping metadata for Mutually Exclusive Softmax
    concept_groups_info = None
    group_source_dataset, _ = unwrap_subset(train_dataset)
    if args.use_concept_groups and hasattr(group_source_dataset, "concept_features_info") and group_source_dataset.concept_features_info is not None:
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
        for info in group_source_dataset.concept_features_info:
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
        tqdm.write(f"  {BOLD}{BLUE}[Softmax Group]{RESET} Configured {grouped_count} mutually exclusive groups out of {len(group_source_dataset.concept_features_info)} total categories.")
    else:
        tqdm.write(f"  {BOLD}{BLUE}[Softmax Group]{RESET} DISABLED (Sigmoid activation fallback active).")

    # 2. Model Initialization
    tqdm.write(f"  {BOLD}{BLUE}[Model]{RESET} {args.backbone_type}/{args.backbone_name}")
    tqdm.write(f"  {BOLD}{BLUE}[Backbone Train Mode]{RESET} {args.backbone_train_mode}")
    tqdm.write(f"  {BOLD}{BLUE}[Concepts]{RESET} Supervised: {num_concepts_supervised} | Latent: {args.latent_concepts} | Total Bottleneck: {num_concepts_total}")

    # 2a. Build group_mapping for GroupToConceptAttention (if requested)
    group_mapping = None
    num_groups    = None
    if args.use_group_broadcasting:
        concept_info_list = None
        group_source_dataset, _ = unwrap_subset(train_dataset)
        if hasattr(group_source_dataset, "concept_features_info") and group_source_dataset.concept_features_info is not None:
            concept_info_list = group_source_dataset.concept_features_info
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
            tqdm.write(f"  {BOLD}{BLUE}[Broadcasting]{RESET} {num_groups} anatomical groups -> {num_concepts_supervised} independent BCE classifiers (Group Softmax disabled).")
        else:
            tqdm.write(f"  {BOLD}{YELLOW}[Warning]{RESET} use_group_broadcasting=True but concept config information could not be resolved. Falling back to standard attention.")

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
        group_mapping=group_mapping,
        # use_dino_mask and dino_mask_threshold parameters removed
        use_nam_head=args.use_nam_head or args.use_gated_nam,
        nam_hidden_dim=args.nam_hidden_dim,
        use_probabilistic_cbm=args.use_probabilistic_cbm,
        use_concept_attention=args.use_concept_attention,
        use_pairwise_nam=args.use_pairwise_nam
    )


    if args.backbone_train_mode == "frozen":
        model.freeze_backbone()
        tqdm.write(f"  {BOLD}{YELLOW}[Freeze]{RESET} Backbone frozen")

    if args.freeze_head:
        model.freeze_classifier()
        tqdm.write(f"  {BOLD}{YELLOW}[Freeze]{RESET} Classifier head frozen")

    if args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            tqdm.write(f"  {BOLD}{CYAN}[Resume]{RESET} Loading pre-trained weights from: {args.resume_checkpoint}")
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
                tqdm.write(f"  {BOLD}{YELLOW}[Warning]{RESET} Detected {len(old_mha_keys)} legacy MHA key(s). Migrating to cosine-attention layout...")
                state_dict = _migrate_state_dict(state_dict)
                tqdm.write(f"  {BOLD}{GREEN}[Migration]{RESET} State-dict migration complete.")

            try:
                model.load_state_dict(state_dict, strict=True)
                tqdm.write(f"  {BOLD}{GREEN}[Success]{RESET} Weights loaded successfully (strict match).")
            except RuntimeError as e:
                tqdm.write(f"  {BOLD}{YELLOW}[Warning]{RESET} Strict loading failed. Attempting non-strict load. Error: {e}")
                model.load_state_dict(state_dict, strict=False)
                tqdm.write(f"  {BOLD}{GREEN}[Success]{RESET} Weights loaded successfully (non-strict match).")
        else:
            tqdm.write(f"  {BOLD}{YELLOW}[Error]{RESET} Checkpoint path '{args.resume_checkpoint}' does not exist. Starting training from scratch.")

    model.to(device)
    if use_concept_bias_calibration and hasattr(model, "concept_bias"):
        model.concept_bias.zero_()
        tqdm.write(
            f"  {BOLD}{CYAN}[Calibration]{RESET} "
            "Concept bias will be learned post-hoc after Phase 3; training phases use zero bias."
        )

    # 3. Loss & Optimizer Setup
    cb_pos_weight = None
    cb_neg_weight = None
    if getattr(args, 'use_cb_loss', True):
        from src.utils.helpers import calculate_class_balanced_weights
        cb_pos, cb_neg = calculate_class_balanced_weights(train_dataset, num_concepts_supervised, beta=getattr(args, 'cb_beta', 0.999))
        cb_pos_weight = cb_pos.to(device)
        cb_neg_weight = cb_neg.to(device)
        tqdm.write(f"  {BOLD}{BLUE}[Concept Loss]{RESET} Class-Balanced Loss weighting enabled (beta={getattr(args, 'cb_beta', 0.999)})")

    if getattr(args, 'use_group_broadcasting', False):
        pos_weights = calculate_pos_weights(train_dataset, num_concepts_supervised).to(device)
        tqdm.write(f"  {BOLD}{BLUE}[Concept Loss]{RESET} BCEWithLogitsLoss (Group Broadcasting mode) with dynamic pos_weights (first 5 shown): {[f'{w:.2f}' for w in pos_weights[:5].tolist()]}")
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
            f"  {BOLD}{BLUE}[Concept Loss]{RESET} Mutually Exclusive GroupCrossEntropyLoss ({len(concept_groups_info)} groups, "
            f"lambda_ce={args.lambda_ce}, fallback_loss={args.concept_loss_type})"
        )
        concept_criterion = GroupCrossEntropyLoss(
            groups_info=concept_groups_info,
            lambda_ce=args.lambda_ce,
            loss_type=args.concept_loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=args.focal_gamma,
            asl_gamma_pos=args.asl_gamma_pos,
            asl_gamma_neg=args.asl_gamma_neg,
            asl_alpha_pos=args.asl_alpha_pos,
            asl_clip=args.asl_clip,
            cb_pos_weight=cb_pos_weight,
            cb_neg_weight=cb_neg_weight
        )
    elif args.concept_loss_type == 'asl':
        tqdm.write(f"  {BOLD}{BLUE}[Concept Loss]{RESET} Asymmetric Loss (gamma_pos={args.asl_gamma_pos}, gamma_neg={args.asl_gamma_neg}, alpha_pos={args.asl_alpha_pos}, clip={args.asl_clip})")
        concept_criterion = AsymmetricLossWithWeight(
            gamma_pos=args.asl_gamma_pos,
            gamma_neg=args.asl_gamma_neg,
            alpha_pos=args.asl_alpha_pos,
            clip=args.asl_clip,
            cb_pos_weight=cb_pos_weight,
            cb_neg_weight=cb_neg_weight
        )
    elif args.concept_loss_type == 'focal':
        if isinstance(args.focal_alpha, str) and args.focal_alpha.lower() == 'dynamic':
            # Dynamically compute per-concept alpha: alpha = pos_weight / (1 + pos_weight)
            pos_weights = calculate_pos_weights(train_dataset, num_concepts_supervised)
            focal_alpha = pos_weights / (1.0 + pos_weights)
            focal_alpha = focal_alpha.to(device)
            tqdm.write(f"  {BOLD}{BLUE}[Concept Loss]{RESET} Sigmoid Focal Loss with DYNAMIC alpha (first 5 shown): {[f'{a:.4f}' for a in focal_alpha[:5].tolist()]}, gamma={args.focal_gamma}")
            concept_criterion = SigmoidFocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
        else:
            tqdm.write(f"  {BOLD}{BLUE}[Concept Loss]{RESET} Sigmoid Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
            concept_criterion = SigmoidFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        # Calculate positive weights dynamically for BCEWithLogitsLoss to handle concept sparsity
        pos_weights = calculate_pos_weights(train_dataset, num_concepts_supervised).to(device)
        tqdm.write(f"  {BOLD}{BLUE}[Concept Loss]{RESET} BCEWithLogitsLoss with dynamic pos_weights (first 5 shown): {[f'{w:.2f}' for w in pos_weights[:5].tolist()]}")
        concept_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    if args.dataset == 'chexpert':
        target_criterion = nn.BCEWithLogitsLoss()
    elif num_classes == 1:
        if args.target_pos_weight != 1.0:
            pos_weight = torch.tensor([args.target_pos_weight], dtype=torch.float32, device=device)
            target_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            target_criterion = nn.BCEWithLogitsLoss()
    else:
        # Multi-class: compute inverse-frequency class weights from training data
        target_source_dataset, target_subset_indices = unwrap_subset(train_dataset)
        if hasattr(target_source_dataset, 'df') and not target_source_dataset.dummy_mode:
            target_col = resolved_config.get("target_col", "diagnosis_idx")
            target_to_idx = getattr(target_source_dataset, "target_to_idx", None)
            target_df = target_source_dataset.df.iloc[target_subset_indices] if target_subset_indices is not None else target_source_dataset.df

            counts = [0] * num_classes
            for val in target_df[target_col].dropna():
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
        tqdm.write(f"  {BOLD}{BLUE}[W&B]{RESET} Run initialized successfully.")

    tqdm.write(f"{'='*60}\n")

    # 4. Sequential Training Phases
    # Phase 1: Concept Learning
    phase1_epochs = args.phase1_epochs if args.phase1_epochs is not None else args.epochs
    if phase1_epochs > 0:
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
    else:
        tqdm.write(f"  {BOLD}{YELLOW}[Skip]{RESET} Skipping Phase 1: phase1_epochs is 0.")

    if phase1_epochs > 0:
        # Safety Backup: Save intermediate Phase 1 checkpoint to allow resuming Phase 2
        save_subdir = os.path.join(args.save_dir, args.backbone_name)
        os.makedirs(save_subdir, exist_ok=True)
        phase1_save_filename = args.save_filename or f"{args.dataset}_{args.backbone_name}_latent{args.latent_concepts}_phase1.pt"
        if not phase1_save_filename.endswith("_phase1.pt") and not phase1_save_filename.endswith("_phase1.pth"):
            phase1_save_filename = phase1_save_filename.replace(".pt", "_phase1.pt").replace(".pth", "_phase1.pth")
        phase1_save_path = os.path.join(save_subdir, phase1_save_filename)

        checkpoint_p1 = {
            'state_dict': model.state_dict(),
            'config': config_data,
            'args': vars(args)
        }
        torch.save(checkpoint_p1, phase1_save_path)
        tqdm.write(f"\n{BOLD}{GREEN}[Safety Backup]{RESET} Phase 1 complete! Saved intermediate checkpoint to: {phase1_save_path}\n")

    # Phase 2: Target Learning
    phase2_epochs = args.phase2_epochs if args.phase2_epochs is not None else args.epochs
    if phase2_epochs > 0:
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
    else:
        tqdm.write(f"  {BOLD}{YELLOW}[Skip]{RESET} Skipping Phase 2: phase2_epochs is 0.")

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

    if calibration_loader is not None:
        bias_cfg = config_data.get("learn_concept_bias", {})
        tqdm.write(f"\n{BOLD}{CYAN}[Calibration]{RESET} Learning fixed supervised concept bias after Phase 3...")
        bias_summary = learn_concept_bias(
            model=model,
            calibration_loader=calibration_loader,
            concept_groups_info=concept_groups_info,
            device=device,
            config=bias_cfg
        )
        tqdm.write(
            f"  {BOLD}{GREEN}[Concept Bias]{RESET} "
            f"{bias_summary['metric']}: {bias_summary['baseline_score']*100:.2f}% "
            f"-> {bias_summary['calibrated_score']*100:.2f}%"
        )
        if args.use_wandb:
            import wandb
            wandb.log({
                "calibration/concept_bias_baseline": bias_summary["baseline_score"],
                "calibration/concept_bias_score": bias_summary["calibrated_score"],
            })

    # Save Model Weights
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

    tqdm.write(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
    tqdm.write(f"  {BOLD}{GREEN}[Success] Training complete!{RESET}")
    tqdm.write(f"  {BOLD}{GREEN}[Save]{RESET} Weights saved to: {save_path}")
    tqdm.write(f"  {BOLD}{GREEN}[Save]{RESET} Heatmaps saved to: visualizations/{run_name}/")
    tqdm.write(f"{BOLD}{GREEN}{'='*60}{RESET}")

    if args.use_wandb:
        wandb.finish()

    # Automatically run evaluation benchmark using subprocess to execute eval_cbm.py
    if args.run_eval:
        tqdm.write(f"\n{BOLD}{CYAN}[Evaluation]{RESET} Launching evaluation benchmark automatically...")
        import subprocess
        import sys

        cmd = [
            sys.executable, "eval_cbm.py",
            "--checkpoint", save_path
        ]
        tqdm.write(f"  Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd)
        except Exception as e:
            tqdm.write(f"\n{BOLD}{YELLOW}[Evaluation]{RESET} Automatic evaluation failed: {e}")

    # Automatically launch Gradio app using subprocess to execute app.py
    if args.run_app:
        tqdm.write(f"\n{BOLD}{CYAN}[Gradio]{RESET} Launching inference application automatically...")
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
        if args.use_gated_nam or args.use_nam_head:
            cmd.extend(["--use_gated_nam", "true"])
        if args.use_pairwise_nam:
            cmd.extend(["--use_pairwise_nam", "true"])
        if args.use_probabilistic_cbm:
            cmd.extend(["--use_probabilistic_cbm", "true"])
        if args.use_concept_attention:
            cmd.extend(["--use_concept_attention", "true"])

        tqdm.write(f"  Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            tqdm.write(f"\n{BOLD}{YELLOW}[Gradio]{RESET} Gradio app stopped by user.")

if __name__ == "__main__":
    main()
