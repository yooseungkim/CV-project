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
from src.models.cbm_factory import UniversalFlexibleCBM
from src.utils.metrics import calculate_accuracy, calculate_concept_accuracy
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
    default_choices = ['milk10k', 'derm7pt']
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
    
    # dataset
    ds_cfg = config_data.get("dataset", {})
    if "dataset" in ds_cfg: flat_defaults["dataset"] = ds_cfg["dataset"]
    if "csv_path" in ds_cfg: flat_defaults["csv_path"] = ds_cfg["csv_path"]
    if "image_dir" in ds_cfg: flat_defaults["image_dir"] = ds_cfg["image_dir"]
    if "concept_config_path" in ds_cfg: flat_defaults["concept_config_path"] = ds_cfg["concept_config_path"]
    
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
    
    # optimizer basic parameter
    opt_cfg = config_data.get("optimizer", {})
    if "lr" in opt_cfg: flat_defaults["lr"] = opt_cfg["lr"]
    
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
    parser.add_argument('--num_classes', type=int, default=flat_defaults.get('num_classes', 1))
    parser.add_argument('--epochs', type=int, default=flat_defaults.get('epochs', 1))
    parser.add_argument('--batch_size', type=int, default=flat_defaults.get('batch_size', 16))
    parser.add_argument('--lr', type=float, default=flat_defaults.get('lr', 1e-3))
    parser.add_argument('--lambda_c', type=float, default=flat_defaults.get('lambda_c', 1.0))
    parser.add_argument('--target_pos_weight', type=float, default=flat_defaults.get('target_pos_weight', 1.0))
    parser.add_argument('--num_workers', type=int, default=flat_defaults.get('num_workers', 4))
    parser.add_argument('--pin_memory', type=str2bool, default=flat_defaults.get('pin_memory', True))
    parser.add_argument('--freeze_backbone', action='store_true', default=flat_defaults.get('freeze_backbone', False))
    parser.add_argument('--freeze_head', action='store_true', default=flat_defaults.get('freeze_head', False))
    parser.add_argument('--use_wandb', type=str2bool, default=flat_defaults.get('use_wandb', True))
    parser.add_argument('--save_dir', type=str, default=flat_defaults.get('save_dir', 'checkpoints'))
    parser.add_argument('--cache_in_memory', type=str2bool, default=flat_defaults.get('cache_in_memory', False))
    parser.add_argument('--max_cache_size_gb', type=float, default=flat_defaults.get('max_cache_size_gb', 10.0))
    parser.add_argument('--resume_checkpoint', type=str, default=flat_defaults.get('resume_checkpoint', None), help="Path to checkpoint .pth to resume or fine-tune from")
    parser.add_argument('--save_filename', type=str, default=None, help="Custom filename to save the final weights")
    
    args = parser.parse_args()
    return args, config_data

def main():
    args, config_data = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create timestamp and run_name early for consistent naming
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
    num_concepts = resolved_config["num_concepts"]
    num_classes = resolved_config["num_classes"]

    tqdm.write(f"  📂 Dataset: {args.dataset} | Concepts: {num_concepts} | Classes: {num_classes}")
    tqdm.write(f"  📊 Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    
    # Log target class names if available
    target_classes = resolved_config.get("target_classes", [])
    if target_classes:
        tqdm.write(f"  🏷️  Classes: {target_classes}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # 2. Model Initialization
    tqdm.write(f"  🧠 Model: {args.backbone_type}/{args.backbone_name}")
    model = UniversalFlexibleCBM(
        backbone_type=args.backbone_type,
        backbone_name=args.backbone_name,
        num_concepts=num_concepts,
        num_classes=num_classes
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
            state_dict = torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
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
    concept_criterion = nn.BCELoss()
    
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
            tqdm.write(f"  ⚖️  Class weights (inv-freq): {[f'{w:.2f}' for w in weights]}")
        else:
            target_criterion = nn.CrossEntropyLoss()
        
    # Configure custom optimizer based on config
    opt_cfg = config_data.get("optimizer", {})
    opt_type = opt_cfg.get("type", "adam").lower()
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    
    # Differential LR separation
    backbone_lr = opt_cfg.get("backbone_lr")
    head_lr = opt_cfg.get("head_lr")
    
    if backbone_lr is not None and head_lr is not None:
        # Separate parameters
        backbone_params = list(model.backbone.parameters())
        backbone_param_ids = set(id(p) for p in backbone_params)
        head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
        
        # Filter trainable parameters
        backbone_trainable = [p for p in backbone_params if p.requires_grad]
        head_trainable = [p for p in head_params if p.requires_grad]
        
        param_groups = [
            {"params": backbone_trainable, "lr": backbone_lr},
            {"params": head_trainable, "lr": head_lr}
        ]
    else:
        param_groups = [{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.lr}]
    
    if opt_type == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    elif opt_type == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        optimizer = optim.SGD(param_groups, weight_decay=weight_decay, momentum=momentum)
    else:  # default 'adam'
        optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
    
    # Configure custom scheduler based on config
    sched_cfg = config_data.get("scheduler", {})
    sched_type = sched_cfg.get("type", "none").lower()
    
    tqdm.write(f"  ⚙️  Optimizer: {opt_type} | Scheduler: {sched_type}")
    scheduler = None
    
    if sched_type == "cosine":
        T_max = sched_cfg.get("T_max", args.epochs)
        eta_min = sched_cfg.get("eta_min", 1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif sched_type == "step":
        step_size = sched_cfg.get("step_size", 10)
        gamma = sched_cfg.get("gamma", 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == "plateau":
        patience = sched_cfg.get("patience", 3)
        factor = sched_cfg.get("factor", 0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
        
    # Configure Early Stopping based on config
    es_cfg = config_data.get("early_stopping", {})
    es_handler = None
    if es_cfg.get("enabled", False):
        patience = es_cfg.get("patience", 5)
        min_delta = es_cfg.get("min_delta", 0.0)
        monitor = es_cfg.get("monitor", "val_loss")
        es_handler = EarlyStopping(patience=patience, min_delta=min_delta, monitor=monitor)
        tqdm.write(f"  🛑 Early stopping: monitor={monitor}, patience={patience}")

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
    # 4. Training Loop
    epoch_pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # ---- Training Phase ----
        model.train()
        total_concept_loss = 0.0
        total_target_loss = 0.0
        total_concept_acc = 0.0
        total_target_acc = 0.0
        
        train_pbar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1}/{args.epochs}",
            bar_format="{l_bar}{bar:25}{r_bar}",
            leave=False
        )
        
        for images, concepts, targets in train_pbar:
            images = images.to(device)
            concepts = concepts.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            class_logits, concept_logits, attn_weights = model(images)
            concept_probs = torch.sigmoid(concept_logits)
            
            loss_c = concept_criterion(concept_probs, concepts)
            if num_classes == 1:
                loss_t = target_criterion(class_logits, targets)
            else:
                loss_t = target_criterion(class_logits, targets.view(-1).long())
            loss = loss_t + args.lambda_c * loss_c
            
            loss.backward()
            optimizer.step()
            
            total_concept_loss += loss_c.item()
            total_target_loss += loss_t.item()
            total_concept_acc += calculate_concept_accuracy(concept_probs.detach(), concepts)
            total_target_acc += calculate_accuracy(class_logits.detach(), targets)
            
            train_pbar.set_postfix(CL=f"{loss_c.item():.4f}", TL=f"{loss_t.item():.4f}")
        
        n_train = len(train_loader)
        avg_concept_loss = total_concept_loss / n_train
        avg_target_loss = total_target_loss / n_train
        avg_concept_acc = total_concept_acc / n_train
        avg_target_acc = total_target_acc / n_train
        
        # ---- Validation Phase (no separate progress bar) ----
        model.eval()
        val_concept_loss = 0.0
        val_target_loss = 0.0
        val_concept_acc = 0.0
        val_target_acc = 0.0
        val_vis_data = None  # store first batch data for visualization
        all_val_concept_probs = []
        all_val_concept_targets = []
        
        with torch.no_grad():
            for val_images, val_concepts, val_targets in val_loader:
                val_images = val_images.to(device)
                val_concepts = val_concepts.to(device)
                val_targets = val_targets.to(device)
                
                v_class_logits, v_concept_logits, v_attn_weights = model(val_images)
                v_concept_probs = torch.sigmoid(v_concept_logits)
                
                v_loss_c = concept_criterion(v_concept_probs, val_concepts)
                if num_classes == 1:
                    v_loss_t = target_criterion(v_class_logits, val_targets)
                else:
                    v_loss_t = target_criterion(v_class_logits, val_targets.view(-1).long())
                
                val_concept_loss += v_loss_c.item()
                val_target_loss += v_loss_t.item()
                val_concept_acc += calculate_concept_accuracy(v_concept_probs, val_concepts)
                val_target_acc += calculate_accuracy(v_class_logits, val_targets)
                
                all_val_concept_probs.append(v_concept_probs.cpu())
                all_val_concept_targets.append(val_concepts.cpu())
                
                # Capture first batch for visualization
                if val_vis_data is None:
                    val_vis_data = (val_images, v_attn_weights)
        
        n_val = len(val_loader)
        avg_val_concept_loss = val_concept_loss / n_val
        avg_val_target_loss = val_target_loss / n_val
        avg_val_concept_acc = val_concept_acc / n_val
        avg_val_target_acc = val_target_acc / n_val
        avg_val_loss = avg_val_concept_loss + avg_val_target_loss
        
        # Calculate individual concept accuracies for detailed validation logging
        val_individual_concept_accs = {}
        if all_val_concept_probs:
            val_concept_probs_all = torch.cat(all_val_concept_probs, dim=0)
            val_concept_targets_all = torch.cat(all_val_concept_targets, dim=0)
            concepts_list = resolved_config.get("concepts_flat", resolved_config.get("concepts", []))
            for c in range(num_concepts):
                name = concepts_list[c] if c < len(concepts_list) else f"Concept_{c}"
                preds_c = (val_concept_probs_all[:, c] > 0.5).float()
                targets_c = (val_concept_targets_all[:, c] > 0.5).float()
                acc_c = (preds_c == targets_c).float().mean().item()
                val_individual_concept_accs[f"val_concept_acc/{name}"] = acc_c
                
            # Log struggling concepts to console
            sorted_concept_accs = sorted(
                [(concepts_list[c] if c < len(concepts_list) else f"Concept_{c}", val_individual_concept_accs[f"val_concept_acc/{concepts_list[c] if c < len(concepts_list) else f'Concept_{c}'}"])
                 for c in range(num_concepts)],
                key=lambda x: x[1]
            )
            lowest_3_str = ", ".join([f"{name}: {acc:.4f}" for name, acc in sorted_concept_accs[:3]])
            tqdm.write(f"  🔍 Struggling Concepts: {lowest_3_str}")
        
        # ---- Visualization: save heatmaps to visualizations/{run_name}/{epoch}/ ----
        if val_vis_data is not None:
            vis_images, vis_attn = val_vis_data
            num_samples = min(4, vis_images.size(0))
            
            heatmap_images = generate_concept_heatmaps(
                image_tensor=vis_images[:num_samples],
                attn_weights=vis_attn[:num_samples],
                concept_names=resolved_config.get("concepts_flat", resolved_config["concepts"])
            )
            
            # Save locally: visualizations/{run_name}/epoch_{N}/sample_{i}.png
            epoch_vis_dir = os.path.join("visualizations", run_name, f"epoch_{epoch + 1}")
            os.makedirs(epoch_vis_dir, exist_ok=True)
            for idx, img in enumerate(heatmap_images):
                img.save(os.path.join(epoch_vis_dir, f"sample_{idx + 1}.png"))
            
            # Log to W&B if enabled
            if args.use_wandb:
                import wandb
                wandb.log({
                    "val/concept_heatmaps": [
                        wandb.Image(img, caption=f"Sample {idx + 1}")
                        for idx, img in enumerate(heatmap_images)
                    ]
                }, commit=False)
        
        # ---- Update epoch progress bar with summary ----
        epoch_pbar.set_postfix(
            TrCL=f"{avg_concept_loss:.4f}",
            TrTL=f"{avg_target_loss:.4f}",
            VaCL=f"{avg_val_concept_loss:.4f}",
            VaTL=f"{avg_val_target_loss:.4f}",
            VAcc=f"{avg_val_target_acc:.4f}"
        )
              
        # 1. Step Learning Rate Scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
                
        # 2. Evaluate Early Stopping
        if es_handler is not None:
            # Map monitor target to metric value
            monitor_target = es_handler.monitor.lower()
            if monitor_target == "val_loss":
                monitor_score = avg_val_loss
            elif monitor_target == "val_target_loss":
                monitor_score = avg_val_target_loss
            elif monitor_target == "val_concept_loss":
                monitor_score = avg_val_concept_loss
            elif monitor_target == "val_acc" or monitor_target == "val_t_acc":
                monitor_score = avg_val_target_acc
            elif monitor_target == "val_concept_acc" or monitor_target == "val_c_acc":
                monitor_score = avg_val_concept_acc
            else:
                monitor_score = avg_val_loss
                
            es_handler(monitor_score, model)
            
            if es_handler.early_stop:
                tqdm.write(f"\n  🛑 Early stopping at Epoch {epoch + 1}. Restoring best weights.")
                model.load_state_dict(es_handler.best_weights)
                break
              
        if args.use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train/total_loss": avg_concept_loss + avg_target_loss,
                "train/concept_loss": avg_concept_loss,
                "train/target_loss": avg_target_loss,
                "val/total_loss": avg_val_loss,
                "val/concept_loss": avg_val_concept_loss,
                "val/target_loss": avg_val_target_loss,
                "val/accuracy": avg_val_target_acc,
                "val/concept_accuracy": avg_val_concept_acc
            }
            log_dict.update(val_individual_concept_accs)
            wandb.log(log_dict)
              
    # Save Model Weights
    mode = "frozen_backbone" if args.freeze_backbone else "full"
    save_subdir = os.path.join(args.save_dir, args.backbone_name)
    os.makedirs(save_subdir, exist_ok=True)
    
    if args.save_filename:
        save_filename = args.save_filename
    else:
        save_filename = f"{timestamp}_cbm_{mode}.pth"
    save_path = os.path.join(save_subdir, save_filename)
    torch.save(model.state_dict(), save_path)
    
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"  ✅ Training complete!")
    tqdm.write(f"  💾 Weights saved: {save_path}")
    tqdm.write(f"  🖼️  Heatmaps saved: visualizations/{run_name}/")
    tqdm.write(f"{'='*60}")

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
