import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.utils.metrics import calculate_accuracy, calculate_concept_metrics
from src.utils.visualization import generate_concept_heatmaps
from src.utils.losses import calculate_orthogonality_loss
from src.utils.helpers import EarlyStopping, unwrap_subset

# ANSI terminal colors for highlighting
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

PHASE_MONITORS = {
    "phase1": (
        "val_concept_loss",
        "val_concept_acc",
        "val_concept_f1",
    ),
    "phase2": (
        "val_target_loss",
        "val_target_acc",
    ),
    "phase3": (
        "val_target_loss",
        "val_joint_loss",
        "val_concept_loss",
        "val_target_acc",
    ),
}

def validate_monitor_name(phase, monitor):
    valid_monitors = PHASE_MONITORS[phase]
    if monitor not in valid_monitors:
        valid_names = ", ".join(sorted(valid_monitors))
        raise ValueError(
            f"{phase}_monitor must exactly match one of [{valid_names}], got {monitor!r}."
        )
    return monitor

def get_monitor_score(phase, monitor, metrics):
    validate_monitor_name(phase, monitor)
    return metrics[monitor]

def get_phase2_sparsity_warmup_epochs(args):
    warmup_epochs = max(0, int(getattr(args, "l1_warmup_epochs", 0) or 0))
    if warmup_epochs == 0:
        return 0
    if getattr(args, "use_gated_nam", False) or getattr(args, "use_nam_head", False):
        return warmup_epochs if getattr(args, "l1_lambda_gate", 0.01) > 0 else 0
    return warmup_epochs if getattr(args, "l1_lambda", 0.0) > 0 else 0

def inject_concept_noise(pred_logits, gt_labels, replace_prob=0.3, epsilon=0.05):
    """
    pred_logits: Phase 1 output logits (Batch, Num_Concepts)
    gt_labels: Ground Truth labels (Batch, Num_Concepts)
    replace_prob: Probability to replace prediction with GT
    """
    batch_size, num_concepts = pred_logits.shape
    
    # 1. Convert GT labels to Soft Logits
    # gt_labels >= 0.5 checks both float and integer 0/1 formats safely
    soft_gt_prob = torch.where(gt_labels >= 0.5, 1.0 - epsilon, epsilon)
    gt_logits = torch.log(soft_gt_prob / (1.0 - soft_gt_prob))
    
    # 2. Generate random mask for Scheduled Sampling
    mask = torch.rand((batch_size, num_concepts), device=pred_logits.device) < replace_prob
    
    # 3. Mix predicted logits and GT logits
    mixed_logits = torch.where(mask, gt_logits, pred_logits)
    
    return mixed_logits

def apply_label_smoothing(hard_labels, epsilon=0.05):
    """
    hard_labels: [Batch, Num_Concepts] containing 0 or 1
    epsilon: smoothing strength (e.g. 0.05 maps 0 to 0.05 and 1 to 0.95)
    """
    if epsilon <= 0.0:
        return hard_labels
    return torch.where(hard_labels >= 0.5, 1.0 - epsilon, epsilon)

def train_phase1(model, train_loader, val_loader, concept_criterion, device, args, config_data, run_name, num_concepts_supervised, resolved_config, concept_groups_info=None):
    tqdm.write(f"\n{BOLD}{MAGENTA}{'-'*60}{RESET}")
    tqdm.write(f"  {BOLD}{MAGENTA}[Phase 1] Concept Learning (Backbone & Concept Head){RESET}")
    tqdm.write(f"{BOLD}{MAGENTA}{'-'*60}{RESET}")
    
    # Extract concept grouping indices from dataset for group-level orthogonality loss
    concept_groups_indices = None
    train_dataset, _ = unwrap_subset(train_loader.dataset)
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
        tqdm.write(f"  {BOLD}{BLUE}[Orthogonality]{RESET} Detected {len(concept_groups_indices)} semantic attribute groups for separation.")
    
    # classifier_head 가중치 동결
    for param in model.classifier_head.parameters():
        param.requires_grad = False
        
    backbone_train_mode = getattr(args, "backbone_train_mode", "full")
    model.set_backbone_train_mode(backbone_train_mode)
    model.unfreeze_supervised_attention()
    model.freeze_latent_attention()
        
    opt_cfg = config_data.get("optimizer", {})
    opt_type = opt_cfg.get("type", "adam").lower()
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    backbone_lr = opt_cfg.get("backbone_lr")
    
    param_groups = []
    if backbone_train_mode != "frozen":
        backbone_trainable = [p for p in model.backbone.parameters() if p.requires_grad]
        if backbone_trainable:
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
        
    es_cfg = config_data.get("early_stopping", {})
    es_handler = None
    if es_cfg.get("enabled", False):
        phase1_patience = args.phase1_patience if args.phase1_patience is not None else es_cfg.get("phase1_patience", es_cfg.get("patience", 5))
        phase1_monitor = args.phase1_monitor if getattr(args, "phase1_monitor", None) is not None else es_cfg.get("phase1_monitor", "val_concept_loss")
        validate_monitor_name("phase1", phase1_monitor)
        min_delta = es_cfg.get("min_delta", 0.0)
        es_handler = EarlyStopping(patience=phase1_patience, min_delta=min_delta, monitor=phase1_monitor)
        tqdm.write(f"  {BOLD}{YELLOW}[Early Stop]{RESET} Phase 1 Early stopping: monitor={phase1_monitor}, patience={phase1_patience}, min_delta={min_delta}")
    
    for epoch in range(phase1_epochs):
        # Calculate current beta for PCBM KL Divergence
        warmup_epochs = getattr(args, "pcbm_beta_warmup_epochs", 10)
        anneal_epochs = getattr(args, "pcbm_beta_anneal_epochs", 10)
        target_beta = getattr(args, "pcbm_beta", 0.001)
        beta_min = getattr(args, "pcbm_beta_min", 0.0001)
        
        if epoch < warmup_epochs:
            current_beta = 0.0
        else:
            if epoch >= warmup_epochs + anneal_epochs:
                current_beta = target_beta
            else:
                ratio = (epoch - warmup_epochs) / anneal_epochs
                current_beta = beta_min + (target_beta - beta_min) * ratio
                
        if getattr(model, "use_probabilistic_cbm", False):
            tqdm.write(f"  {BOLD}{CYAN}[PCBM Beta]{RESET} Epoch {epoch+1} KL Weight Beta: {current_beta:.6f}")

        model.train()
        total_loss_c = 0.0
        total_acc_c = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"  P1 Epoch {epoch+1}/{phase1_epochs}", bar_format="{l_bar}{bar:25}{r_bar}", leave=False)
        for images, concepts, _ in train_pbar:
            images = images.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            _, concept_logits, attn_weights = model(images)
            
            # Apply label smoothing if configured
            smooth_epsilon = getattr(args, "phase1_label_smoothing", 0.05)
            smoothed_concepts = apply_label_smoothing(concepts, epsilon=smooth_epsilon)
            
            # Use raw logits with BCEWithLogitsLoss
            loss_c = concept_criterion(concept_logits[:, :num_concepts_supervised], smoothed_concepts)
            
            if getattr(model, "use_probabilistic_cbm", False):
                mean = model.last_mean
                logvar = model.last_logvar
                kl_loss_raw = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
                
                # Filter out ignored/missing concepts (< 0.0, e.g. -1)
                valid_mask = (concepts >= 0.0)
                pos_mask = (concepts > 0.5) & valid_mask
                neg_mask = (concepts <= 0.5) & valid_mask
                
                # Inverse frequency weighting to balance positive/negative components and prevent negative bias dominance
                pos_sum = (kl_loss_raw * pos_mask.float()).sum(dim=0)
                neg_sum = (kl_loss_raw * neg_mask.float()).sum(dim=0)
                
                P_c = pos_mask.sum(dim=0).float()
                N_c = neg_mask.sum(dim=0).float()
                
                # Prevent division by zero to avoid NaN generation in torch.where
                P_c_safe = torch.where(P_c > 0, P_c, torch.ones_like(P_c))
                N_c_safe = torch.where(N_c > 0, N_c, torch.ones_like(N_c))
                
                avg_kl_pos = torch.where(P_c > 0, pos_sum / P_c_safe, torch.zeros_like(pos_sum))
                avg_kl_neg = torch.where(N_c > 0, neg_sum / N_c_safe, torch.zeros_like(neg_sum))
                
                asym_weight = getattr(args, "pcbm_asymmetric_kl_weight", 0.1)
                kl_loss_per_concept = avg_kl_neg + asym_weight * avg_kl_pos
                kl_loss = kl_loss_per_concept.mean()
                
                loss_c = loss_c + current_beta * kl_loss
            
            # Compute spatial orthogonality loss for the supervised concept attention maps
            if getattr(args, "ortho_lambda", 0.0) > 0.0:
                if getattr(model, "use_group_broadcasting", False):
                    # Under group broadcasting, attention maps are already at the group level
                    attn_to_ortho = attn_weights
                else:
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
                val_images = val_images.to(device, non_blocking=True)
                val_concepts = val_concepts.to(device, non_blocking=True)
                
                _, v_concept_logits, v_attn_weights = model(val_images)
                
                # Apply label smoothing if configured
                smooth_epsilon = getattr(args, "phase1_label_smoothing", 0.05)
                v_smoothed_concepts = apply_label_smoothing(val_concepts, epsilon=smooth_epsilon)
                
                # BCEWithLogitsLoss with raw logits
                v_loss_c = concept_criterion(v_concept_logits[:, :num_concepts_supervised], v_smoothed_concepts)
                
                if getattr(model, "use_probabilistic_cbm", False):
                    mean = model.last_mean
                    logvar = model.last_logvar
                    kl_loss_raw = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
                    
                    # Filter out ignored/missing concepts (< 0.0, e.g. -1)
                    valid_mask = (val_concepts >= 0.0)
                    pos_mask = (val_concepts > 0.5) & valid_mask
                    neg_mask = (val_concepts <= 0.5) & valid_mask
                    
                    # Inverse frequency weighting to balance positive/negative components and prevent negative bias dominance
                    pos_sum = (kl_loss_raw * pos_mask.float()).sum(dim=0)
                    neg_sum = (kl_loss_raw * neg_mask.float()).sum(dim=0)
                    
                    P_c = pos_mask.sum(dim=0).float()
                    N_c = neg_mask.sum(dim=0).float()
                    
                    # Prevent division by zero to avoid NaN generation in torch.where
                    P_c_safe = torch.where(P_c > 0, P_c, torch.ones_like(P_c))
                    N_c_safe = torch.where(N_c > 0, N_c, torch.ones_like(N_c))
                    
                    avg_kl_pos = torch.where(P_c > 0, pos_sum / P_c_safe, torch.zeros_like(pos_sum))
                    avg_kl_neg = torch.where(N_c > 0, neg_sum / N_c_safe, torch.zeros_like(neg_sum))
                    
                    asym_weight = getattr(args, "pcbm_asymmetric_kl_weight", 0.1)
                    kl_loss_per_concept = avg_kl_neg + asym_weight * avg_kl_pos
                    kl_loss = kl_loss_per_concept.mean()
                    
                    v_loss_c = v_loss_c + current_beta * kl_loss
                
                if getattr(args, "ortho_lambda", 0.0) > 0.0:
                    if getattr(model, "use_group_broadcasting", False):
                        # Under group broadcasting, attention maps are already at the group level
                        v_attn_to_ortho = v_attn_weights
                    else:
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
            val_f1 = val_metrics["mean_f1"]
        else:
            avg_val_acc_c = 0.0
            val_tpr = 0.0
            val_tnr = 0.0
            val_f1 = 0.0
            
        # 에포크 정보 한 줄 출력 (스크롤 이력 보존)
        tqdm.write(f"[Phase 1] Epoch {epoch+1:02d}/{phase1_epochs:02d} | Train Concept Loss: {avg_loss_c:.4f} | Val Concept Loss: {avg_val_loss_c:.4f} | Val Concept Balanced Acc: {avg_val_acc_c * 100:.2f}% | TPR: {val_tpr * 100:.2f}% | TNR: {val_tnr * 100:.2f}% | F1-Score: {val_f1 * 100:.2f}%")
        
        concepts_list = resolved_config.get("concepts_flat", resolved_config.get("concepts", []))
        
        # Determine separate concepts list for group-level heatmap visualization under group broadcasting
        if getattr(model, "use_group_broadcasting", False):
            if hasattr(train_dataset, "concept_features_info") and train_dataset.concept_features_info is not None:
                heatmap_concepts_list = [info["name"] for info in train_dataset.concept_features_info]
            else:
                heatmap_concepts_list = resolved_config.get("concepts", [])
        else:
            heatmap_concepts_list = concepts_list
            
        # struggling concepts는 마지막 epoch이거나 조기종료일 때만 출력하여 로그 노이즈 최소화
        is_last_epoch = (epoch == phase1_epochs - 1)
        
        early_stop_triggered = False
        if es_handler is not None:
            phase1_metrics = {
                "val_concept_loss": avg_val_loss_c,
                "val_concept_acc": avg_val_acc_c,
                "val_concept_f1": val_f1,
            }
            monitor_score = get_monitor_score("phase1", es_handler.monitor, phase1_metrics)
            es_handler(monitor_score, model)
            early_stop_triggered = es_handler.early_stop
        
        # Compute individual balanced accuracies for struggling concepts and logging
        if all_val_probs:
            val_individual_accs = {}
            for c in range(num_concepts_supervised):
                name = concepts_list[c] if c < len(concepts_list) else f"Concept_{c}"
                ind_balanced_acc = val_metrics["individual_balanced_acc"][c].item()
                val_individual_accs[f"val_concept_acc/{name}"] = ind_balanced_acc
                
            if is_last_epoch or early_stop_triggered:
                sorted_concept_accs = sorted(
                    [(concepts_list[c] if c < len(concepts_list) else f"Concept_{c}", val_individual_accs[f"val_concept_acc/{concepts_list[c] if c < len(concepts_list) else f'Concept_{c}'}"])
                     for c in range(num_concepts_supervised)],
                    key=lambda x: x[1]
                )
                lowest_3 = ", ".join([f"{name}: {acc:.4f}" for name, acc in sorted_concept_accs[:3]])
                tqdm.write(f"  {BOLD}{YELLOW}[Struggling Concepts]{RESET} Final Struggling Concepts (Balanced Acc): {lowest_3}")
            
        if val_vis_data is not None and (is_last_epoch or early_stop_triggered):
            vis_images, vis_attn = val_vis_data
            num_samples = min(4, vis_images.size(0))
            heatmap_images = generate_concept_heatmaps(
                image_tensor=vis_images[:num_samples],
                attn_weights=vis_attn[:num_samples, :len(heatmap_concepts_list)],
                concept_names=heatmap_concepts_list
            )
            epoch_vis_dir = os.path.join("visualizations", run_name, f"phase1_epoch_{epoch + 1}")
            os.makedirs(epoch_vis_dir, exist_ok=True)
            for idx, img in enumerate(heatmap_images):
                img.save(os.path.join(epoch_vis_dir, f"sample_{idx + 1}.png"))
                
        if scheduler is not None:
            scheduler.step()
            
        if args.use_wandb:
            import wandb
            log_dict = {
                "phase1_epoch": epoch + 1,
                "train/concept_loss": avg_loss_c,
                "val/concept_loss": avg_val_loss_c,
                "val/concept_accuracy": avg_val_acc_c,
                "val/concept_tpr": val_tpr,
                "val/concept_tnr": val_tnr,
                "val/concept_f1": val_f1
            }
            if 'val_individual_accs' in locals():
                log_dict.update(val_individual_accs)
            if es_handler is not None:
                log_dict.update({
                    "early_stop/phase1_counter": es_handler.counter,
                    "early_stop/phase1_triggered": early_stop_triggered,
                })
            wandb.log(log_dict)

        if early_stop_triggered:
            tqdm.write(f"  {BOLD}{YELLOW}[Early Stop]{RESET} Early stopping Phase 1 at Epoch {epoch + 1}. Restoring best Phase 1 weights.")
            model.load_state_dict(es_handler.best_weights)
            break

    # Phase 1 루프 완료 후 (조기 종료되지 않고 완주한 경우에도) 베스트 가중치 복원
    if es_handler is not None and not es_handler.early_stop and es_handler.best_weights is not None:
        tqdm.write(f"  {BOLD}{YELLOW}[Restore]{RESET} Phase 1 completed. Restoring best Phase 1 weights.")
        model.load_state_dict(es_handler.best_weights)

    # ── Final Validation Concept Metric Summary ──────────────────────────
    model.eval()
    all_val_logits = []
    all_val_targets = []
    
    with torch.no_grad():
        for val_images, val_concepts, _ in val_loader:
            val_images = val_images.to(device, non_blocking=True)
            val_concepts = val_concepts.to(device, non_blocking=True)
            
            _, val_concept_logits, _ = model(val_images)
            
            all_val_logits.append(val_concept_logits[:, :num_concepts_supervised].cpu())
            all_val_targets.append(val_concepts.cpu())
            
    all_val_logits = torch.cat(all_val_logits, dim=0)
    all_val_targets = torch.cat(all_val_targets, dim=0)
    
    # Compute final main concept metrics without multi-class group thresholding.
    if concept_groups_info is None and not getattr(args, "use_group_broadcasting", False):
        concept_config_path = getattr(args, "concept_config_path", None) or (resolved_config.get("concept_config_path") if isinstance(resolved_config, dict) else None)
        if concept_config_path and os.path.exists(concept_config_path):
            try:
                import json
                with open(concept_config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                concept_groups_indices = []
                total_dims = 0
                for name, info in cfg.items():
                    ctype = info.get("type", "numerical")
                    if ctype == "categorical":
                        classes = info.get("classes", [])
                        num_feats = len(classes)
                        concept_groups_indices.append(list(range(total_dims, total_dims + num_feats)))
                        total_dims += num_feats
                    else:
                        concept_groups_indices.append([total_dims])
                        total_dims += 1
                concept_groups_info = [(indices[0], len(indices)) for indices in concept_groups_indices]
            except Exception as e:
                tqdm.write(f"  ⚠️ Error loading concept config in train_loops.py: {e}")
        
    std_metrics = calculate_concept_metrics(
        all_val_logits,
        all_val_targets,
        concept_groups_info=concept_groups_info
    )

    tqdm.write(f"\n{BOLD}{CYAN}[Concept Metrics]{RESET} Phase 1 final validation metrics use argmax for multi-class groups:")
    tqdm.write(f"   ├─ Concept Mean Balanced Accuracy : {std_metrics['mean_balanced_acc']*100:.2f}%")
    tqdm.write(f"   ├─ Concept Mean True Positive Rate: {std_metrics['tpr']*100:.2f}%")
    tqdm.write(f"   ├─ Concept Mean True Negative Rate: {std_metrics['tnr']*100:.2f}%")
    tqdm.write(f"   ├─ Concept Mean F1-Score          : {std_metrics['mean_f1']*100:.2f}%")
    tqdm.write(f"   └─ Concept Mean F2-Score          : {std_metrics['mean_f_beta']*100:.2f}%")
    tqdm.write(f"{BOLD}{CYAN}============================================================{RESET}\n")

def train_phase2(model, train_loader, val_loader, target_criterion, device, args, config_data, run_name, num_concepts_supervised, resolved_config, num_classes):
    tqdm.write(f"\n{BOLD}{MAGENTA}{'-'*60}{RESET}")
    tqdm.write(f"  {BOLD}{MAGENTA}[Phase 2] Target Learning (Classifier Head){RESET}")
    tqdm.write(f"{BOLD}{MAGENTA}{'-'*60}{RESET}")
    
    # 백본과 컨셉 어텐션 가중치 엄격히 동결
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.freeze_supervised_attention()
    model.unfreeze_latent_attention()
    for param in model.classifier_head.parameters():
        param.requires_grad = True
        
    # 💧 Phase 2용 드롭아웃 설정 (사용자 요청: dropout 약하게 적용)
    original_dropout_p = getattr(model.dropout, 'p', 0.2)
    phase2_dropout_p = getattr(args, "phase2_dropout", None)
    if phase2_dropout_p is None:
        phase2_dropout_p = config_data.get("training", {}).get("phase2_dropout", 0.05)

    if hasattr(model, 'dropout'):
        model.dropout.p = phase2_dropout_p
        tqdm.write(f"  {BOLD}{YELLOW}[Dropout]{RESET} Adjusted dropout probability for Phase 2: {original_dropout_p} -> {model.dropout.p}")
        
    opt_cfg = config_data.get("optimizer", {})
    opt_type = opt_cfg.get("type", "adam").lower()
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    phase2_lr = args.phase2_lr if args.phase2_lr is not None else opt_cfg.get("phase2_lr", opt_cfg.get("head_lr", args.lr))
    
    if getattr(args, "use_gated_nam", False) or getattr(args, "use_nam_head", False):
        gates_params = []
        subnet_params = []
        for name, param in model.classifier_head.named_parameters():
            if param.requires_grad:
                if "gate" in name:
                    gates_params.append(param)
                else:
                    subnet_params.append(param)
        if model.num_latent_concepts > 0:
            for param in model.latent_attention.parameters():
                if param.requires_grad:
                    subnet_params.append(param)
                    
        param_groups = [
            {"params": subnet_params, "weight_decay": getattr(args, "weight_decay_nam", 1e-2)},
            {"params": gates_params, "weight_decay": 0.0}
        ]
        tqdm.write(f"  {BOLD}{BLUE}[Optimizer Split]{RESET} Separate group: {len(subnet_params)} subnetwork tensors (WD={getattr(args, 'weight_decay_nam', 1e-2):.4f}), {len(gates_params)} gate tensors (WD=0.0)")
    else:
        trainable_params = list(model.classifier_head.parameters())
        if model.num_latent_concepts > 0:
            trainable_params += list(model.latent_attention.parameters())
        param_groups = [{"params": trainable_params, "weight_decay": weight_decay}]
        
    if opt_type == "adamw":
        optimizer = optim.AdamW(param_groups, lr=phase2_lr)
    elif opt_type == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        optimizer = optim.SGD(param_groups, lr=phase2_lr, momentum=momentum)
    else:
        optimizer = optim.Adam(param_groups, lr=phase2_lr)
        
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
    phase2_early_stopping_start_epoch = 0
    if es_cfg.get("enabled", False):
        phase2_monitor = args.phase2_monitor if getattr(args, "phase2_monitor", None) is not None else es_cfg.get("phase2_monitor", es_cfg.get("monitor", "val_target_loss"))
        validate_monitor_name("phase2", phase2_monitor)
        phase2_patience = args.phase2_patience if args.phase2_patience is not None else es_cfg.get("phase2_patience", es_cfg.get("patience", 5))
        min_delta = es_cfg.get("min_delta", 0.0)
        es_handler = EarlyStopping(patience=phase2_patience, min_delta=min_delta, monitor=phase2_monitor)
        phase2_early_stopping_start_epoch = get_phase2_sparsity_warmup_epochs(args)
        tqdm.write(f"  {BOLD}{YELLOW}[Early Stop]{RESET} Phase 2 Early stopping: monitor={phase2_monitor}, patience={phase2_patience}, min_delta={min_delta}, starts_after_warmup_epochs={phase2_early_stopping_start_epoch}")
        
    for epoch in range(phase2_epochs):
        model.train()
        total_loss_t = 0.0
        total_acc_t = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"  P2 Epoch {epoch+1}/{phase2_epochs}", bar_format="{l_bar}{bar:25}{r_bar}", leave=False)
        for images, concepts, targets in train_pbar:
            images = images.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # 이전 단계에서 학습된 수퍼바이즈드 컨셉 예측값을 그래프 연산 분리하여 추출
            with torch.no_grad():
                features = model.backbone(images)
                if hasattr(model.supervised_attention, 'mlp'):
                    if getattr(model, "use_probabilistic_cbm", False):
                        supervised_mean, supervised_logvar, supervised_topk_indices, supervised_weights = model.supervised_attention(features, return_weights=True)
                        if model.training:
                            std = torch.exp(0.5 * supervised_logvar)
                            eps = torch.randn_like(std)
                            supervised_logits = supervised_mean + std * eps
                        else:
                            supervised_logits = supervised_mean
                    else:
                        supervised_logits, supervised_topk_indices, supervised_weights = model.supervised_attention(features, return_weights=True)
                    
                    k_val = supervised_topk_indices.size(1)
                    indices_transposed = supervised_topk_indices.permute(0, 2, 1) # [B, num_supervised_concepts, k]
                    weights_transposed = supervised_weights.permute(0, 2, 1) # [B, num_supervised_concepts, k]
                    B_size = features.size(0)
                    D_dim = features.size(-1)
                    num_c = indices_transposed.size(1)
                    
                    flat_indices = indices_transposed.reshape(B_size, num_c * k_val)
                    gathered_flat = torch.gather(
                        features,
                        dim=1,
                        index=flat_indices.unsqueeze(-1).expand(-1, -1, D_dim)
                    )
                    gathered_features = gathered_flat.view(B_size, num_c, k_val, D_dim)
                    supervised_features = torch.sum(gathered_features * weights_transposed.unsqueeze(-1), dim=2)
                else:
                    if getattr(model, "use_probabilistic_cbm", False):
                        supervised_mean, supervised_logvar, _, supervised_features = model.supervised_attention(features)
                        if model.training:
                            std = torch.exp(0.5 * supervised_logvar)
                            eps = torch.randn_like(std)
                            supervised_logits = supervised_mean + std * eps
                        else:
                            supervised_logits = supervised_mean
                    else:
                        supervised_logits, _, supervised_features = model.supervised_attention(features)
                    
            # Apply scheduled sampling (concept noise injection) if enabled
            if getattr(args, "phase2_scheduled_sampling", False):
                supervised_logits = inject_concept_noise(
                    pred_logits=supervised_logits,
                    gt_labels=concepts[:, :num_concepts_supervised],
                    replace_prob=getattr(args, "scheduled_sampling_prob", 0.3),
                    epsilon=getattr(args, "scheduled_sampling_epsilon", 0.05)
                )
                
            optimizer.zero_grad()
            
            # 레이턴트 컨셉은 그래디언트를 흘려주어야 하므로 no_grad 밖에서 계산
            if model.num_latent_concepts > 0:
                if hasattr(model.latent_attention, 'mlp'):
                    if getattr(model, "use_probabilistic_cbm", False):
                        latent_mean, latent_logvar, latent_topk_indices, latent_weights = model.latent_attention(features, return_weights=True)
                        if model.training:
                            std_l = torch.exp(0.5 * latent_logvar)
                            eps_l = torch.randn_like(std_l)
                            latent_logits = latent_mean + std_l * eps_l
                        else:
                            latent_logits = latent_mean
                    else:
                        latent_logits, latent_topk_indices, latent_weights = model.latent_attention(features, return_weights=True)
                    
                    k_val = latent_topk_indices.size(1)
                    indices_transposed = latent_topk_indices.permute(0, 2, 1) # [B, num_latent_concepts, k]
                    weights_transposed = latent_weights.permute(0, 2, 1) # [B, num_latent_concepts, k]
                    B_size = features.size(0)
                    D_dim = features.size(-1)
                    num_c = indices_transposed.size(1)
                    
                    flat_indices = indices_transposed.reshape(B_size, num_c * k_val)
                    gathered_flat = torch.gather(
                        features,
                        dim=1,
                        index=flat_indices.unsqueeze(-1).expand(-1, -1, D_dim)
                    )
                    gathered_features = gathered_flat.view(B_size, num_c, k_val, D_dim)
                    latent_features = torch.sum(gathered_features * weights_transposed.unsqueeze(-1), dim=2)
                else:
                    if getattr(model, "use_probabilistic_cbm", False):
                        latent_mean, latent_logvar, _, latent_features = model.latent_attention(features)
                        if model.training:
                            std_l = torch.exp(0.5 * latent_logvar)
                            eps_l = torch.randn_like(std_l)
                            latent_logits = latent_mean + std_l * eps_l
                        else:
                            latent_logits = latent_mean
                    else:
                        latent_logits, _, latent_features = model.latent_attention(features)
                concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
            else:
                concept_logits = supervised_logits
                latent_features = None
                
            concept_logits_for_classifier = model.apply_concept_bias(concept_logits)
            concept_probs = model.concept_activation(concept_logits_for_classifier)
            concept_logits_dropout = model.dropout(concept_logits_for_classifier)
            class_logits = model.classifier_head(concept_logits_dropout)
            
            if isinstance(target_criterion, nn.BCEWithLogitsLoss):
                loss_t = target_criterion(class_logits, targets)
            elif num_classes == 1:
                loss_t = target_criterion(class_logits, targets)
            else:
                loss_t = target_criterion(class_logits, targets.view(-1).long())
                
            # CBM Latent concept regularization terms
            loss_latent_ortho = torch.tensor(0.0, device=device)
            loss_latent_l1 = torch.tensor(0.0, device=device)
            
            if model.num_latent_concepts > 0 and latent_features is not None:
                # 1. Cosine similarity-based Orthogonal Projection Loss
                explicit_norm = F.normalize(supervised_features, p=2, dim=-1)
                latent_norm = F.normalize(latent_features, p=2, dim=-1)
                
                # Compute batch cosine similarity matrix: [B, L, S]
                cos_sim = torch.bmm(latent_norm, explicit_norm.transpose(1, 2))
                loss_latent_ortho = (cos_sim ** 2).mean()
                
                # 2. L1 sparsity regularization on latent concept activations
                latent_activations = concept_probs[:, num_concepts_supervised:]  # [B, L]
                loss_latent_l1 = latent_activations.abs().mean()
                
                # Aggregate losses
                loss_t = loss_t + (args.lambda_latent_ortho * loss_latent_ortho) + (args.lambda_latent_l1 * loss_latent_l1)
                
            # L1 Lasso Regularization on classifier_head parameters to select high-information concepts
            # Apply linear warm-up scheduler to prevent early gating collapse
            if getattr(args, "use_gated_nam", False) or getattr(args, "use_nam_head", False):
                target_l1_gate = getattr(args, "l1_lambda_gate", 0.01)
                warmup_epochs = getattr(args, "l1_warmup_epochs", 5)
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    current_l1_gate = target_l1_gate * (epoch / warmup_epochs)
                else:
                    current_l1_gate = target_l1_gate
                
                latent_penalty_scale = getattr(args, "latent_penalty_scale", 1.0)
                loss_t = loss_t + current_l1_gate * model.classifier_head.get_sparsity_loss(latent_penalty_scale=latent_penalty_scale)
            else:
                target_l1_lambda = getattr(args, "l1_lambda", 0.0)
                warmup_epochs = getattr(args, "l1_warmup_epochs", 5)
                if target_l1_lambda > 0:
                    if warmup_epochs > 0 and epoch < warmup_epochs:
                        current_l1_lambda = target_l1_lambda * (epoch / warmup_epochs)
                    else:
                        current_l1_lambda = target_l1_lambda
                    
                    l1_norm = sum(p.abs().sum() for p in model.classifier_head.parameters())
                    loss_t = loss_t + current_l1_lambda * l1_norm
                
            loss_t.backward()
            optimizer.step()
            
            # Apply proximal hard-thresholding to GatedSparseNAMHead gates for exact sparsity
            if hasattr(model.classifier_head, 'concept_gates'):
                with torch.no_grad():
                    threshold = 0.05
                    model.classifier_head.concept_gates.copy_(
                        torch.where(
                            model.classifier_head.concept_gates.abs() < threshold,
                            torch.zeros_like(model.classifier_head.concept_gates),
                            model.classifier_head.concept_gates
                        )
                    )
            if hasattr(model.classifier_head, 'latent_gates') and model.classifier_head.latent_gates is not None:
                with torch.no_grad():
                    threshold = 0.05
                    model.classifier_head.latent_gates.copy_(
                        torch.where(
                            model.classifier_head.latent_gates.abs() < threshold,
                            torch.zeros_like(model.classifier_head.latent_gates),
                            model.classifier_head.latent_gates
                        )
                    )
            
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
                val_images = val_images.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True)
                
                v_class_logits, _, _ = model(val_images)
                
                if isinstance(target_criterion, nn.BCEWithLogitsLoss):
                    v_loss_t = target_criterion(v_class_logits, val_targets)
                elif num_classes == 1:
                    v_loss_t = target_criterion(v_class_logits, val_targets)
                else:
                    v_loss_t = target_criterion(v_class_logits, val_targets.view(-1).long())
                    
                val_loss_t += v_loss_t.item()
                val_acc_t += calculate_accuracy(v_class_logits, val_targets)
                
        avg_val_loss_t = val_loss_t / len(val_loader)
        avg_val_acc_t = val_acc_t / len(val_loader)
        
        # 에포크 정보 한 줄 출력 (스크롤 이력 보존)
        tqdm.write(f"{BOLD}{MAGENTA}[Phase 2]{RESET} Epoch {epoch+1:02d}/{phase2_epochs:02d} | Train Target Loss: {avg_loss_t:.4f} | Val Target Loss: {avg_val_loss_t:.4f} | Val Target Acc: {BOLD}{GREEN}{avg_val_acc_t * 100:.2f}%{RESET}")
        
        # Track Active Gates if GatedSparseNAMHead is used
        active_count = None
        gate_mean_val = None
        if hasattr(model.classifier_head, 'concept_gates'):
            with torch.no_grad():
                gates = model.classifier_head.concept_gates.detach().cpu()
                active_count = (gates.abs() > 0.0).sum().item()
                gate_mean_val = gates.abs().mean().item()
                tqdm.write(f"  {BOLD}{CYAN}[NAM Gating]{RESET} Active Gates: {active_count}/{gates.size(0)} | Gate Mean: {gate_mean_val:.4f}")
        if hasattr(model.classifier_head, 'latent_gates') and model.classifier_head.latent_gates is not None:
            with torch.no_grad():
                l_gates = model.classifier_head.latent_gates.detach().cpu()
                active_l_count = (l_gates.abs() > 0.0).sum().item()
                tqdm.write(f"  {BOLD}{CYAN}[NAM Latent Gating]{RESET} Active Latent Gates: {active_l_count}/{l_gates.size(0)} | Latent Mean: {l_gates.abs().mean().item():.4f}")
        
        if scheduler is not None:
            scheduler.step()
            
        early_stop_triggered = False
        if es_handler is not None and epoch >= phase2_early_stopping_start_epoch:
            phase2_metrics = {
                "val_target_loss": avg_val_loss_t,
                "val_target_acc": avg_val_acc_t,
            }
            monitor_score = get_monitor_score("phase2", es_handler.monitor, phase2_metrics)
            es_handler(monitor_score, model)
            early_stop_triggered = es_handler.early_stop
                
        if args.use_wandb:
            import wandb
            log_dict = {
                "phase2_epoch": epoch + 1,
                "train/target_loss": avg_loss_t,
                "val/target_loss": avg_val_loss_t,
                "val/accuracy": avg_val_acc_t
            }
            if active_count is not None:
                log_dict.update({
                    "val/active_gates": active_count,
                    "val/gate_mean": gate_mean_val
                })
            if es_handler is not None:
                log_dict.update({
                    "early_stop/phase2_counter": es_handler.counter,
                    "early_stop/phase2_counting": epoch >= phase2_early_stopping_start_epoch,
                    "early_stop/phase2_triggered": early_stop_triggered,
                })
            wandb.log(log_dict)

        if early_stop_triggered:
            tqdm.write(f"  {BOLD}{YELLOW}[Early Stop]{RESET} Early stopping Phase 2 at Epoch {epoch + 1}. Restoring best Phase 2 weights.")
            model.load_state_dict(es_handler.best_weights)
            break

    # 💧 Phase 2 종료 후 드롭아웃 원복
    if hasattr(model, 'dropout'):
        model.dropout.p = original_dropout_p
        tqdm.write(f"  {BOLD}{YELLOW}[Dropout]{RESET} Restored dropout probability after Phase 2: {model.dropout.p}")

    # Phase 2 루프 완료 후 (조기 종료되지 않고 완주한 경우에도) 베스트 가중치 복원
    if es_handler is not None and not es_handler.early_stop and es_handler.best_weights is not None:
        tqdm.write(f"  {BOLD}{YELLOW}[Restore]{RESET} Phase 2 completed. Restoring best Phase 2 weights.")
        model.load_state_dict(es_handler.best_weights)

    if hasattr(model.classifier_head, 'concept_gates'):
        with torch.no_grad():
            gates = model.classifier_head.concept_gates.detach().cpu()
            final_active_count = (gates.abs() > 0.0).sum().item()
            tqdm.write(f"  {BOLD}{GREEN}[NAM Gating Post-Phase 2]{RESET} Final Active Gates (threshold=0.05): {final_active_count}/{gates.size(0)}")
    if hasattr(model.classifier_head, 'latent_gates') and model.classifier_head.latent_gates is not None:
        with torch.no_grad():
            l_gates = model.classifier_head.latent_gates.detach().cpu()
            final_l_active = (l_gates.abs() > 0.0).sum().item()
            tqdm.write(f"  {BOLD}{GREEN}[NAM Latent Gating Post-Phase 2]{RESET} Final Active Latent Gates (threshold=0.05): {final_l_active}/{l_gates.size(0)}")

def train_phase3(model, train_loader, val_loader, target_criterion, concept_criterion, device, args, config_data, run_name, num_concepts_supervised, resolved_config, num_classes):
    tqdm.write(f"\n{BOLD}{MAGENTA}{'-'*60}{RESET}")
    tqdm.write(f"  {BOLD}{MAGENTA}[Phase 3] Joint Fine-Tuning{RESET}")
    tqdm.write(f"{BOLD}{MAGENTA}{'-'*60}{RESET}")
    
    # 1. Phase 3 always tunes the concept/classifier heads; backbone trainability is mode-controlled.
    model.unfreeze_supervised_attention()
    if model.num_latent_concepts > 0:
        model.unfreeze_latent_attention()
    model.set_backbone_train_mode(getattr(args, "backbone_train_mode", "full"))
    model.unfreeze_classifier()
    tqdm.write(f"  {BOLD}{GREEN}[Unfreeze]{RESET} Concept Head and Classifier Head enabled for fine-tuning.")

    # Adjust dropout for Phase 3 to 0.3
    original_dropout_p = getattr(model.dropout, 'p', 0.2)
    if hasattr(model, 'dropout'):
        model.dropout.p = 0.3
        tqdm.write(f"  {BOLD}{YELLOW}[Dropout]{RESET} Adjusted dropout probability for Phase 3: {original_dropout_p} -> 0.3")
    
    opt_cfg = config_data.get("optimizer", {})
    opt_type = opt_cfg.get("type", "adam").lower()
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    
    # Use very small learning rate for joint tuning
    phase3_lr = args.phase3_lr if args.phase3_lr is not None else 1e-5
    phase3_epochs = args.phase3_epochs
    
    # Use differential learning rates: phase3_lr for backbone/concept head, and 50x higher LR for classifier head
    backbone_params = []
    head_params = []
    concept_head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "backbone" in name:
                backbone_params.append(param)
            elif "supervised_attention" in name or "latent_attention" in name:
                concept_head_params.append(param)
            else:
                head_params.append(param)
                 
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": phase3_lr})
    if concept_head_params:
        param_groups.append({"params": concept_head_params, "lr": phase3_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": phase3_lr * 50})  # 50x higher learning rate for classifier/gating parameters
    
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    tqdm.write(f"  {BOLD}{BLUE}[Model]{RESET} Trainable parameters in Phase 3: {len(backbone_params) + len(concept_head_params) + len(head_params)} tensors")
    tqdm.write(f"     └─ Modules: {', '.join(dict.fromkeys(n.split('.')[0] for n in trainable_names))}")
    tqdm.write(f"     └─ Differential LRs: backbone={phase3_lr:.6f}, concept_head={phase3_lr:.6f}, classifier_head={phase3_lr * 50:.6f}")
    
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
    
    if sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase3_epochs, eta_min=sched_cfg.get("eta_min", 1e-6))
    elif sched_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_cfg.get("step_size", 10), gamma=sched_cfg.get("gamma", 0.1))
        
    es_cfg = config_data.get("early_stopping", {})
    es_handler = None
    if es_cfg.get("enabled", False):
        phase3_monitor = args.phase3_monitor if getattr(args, "phase3_monitor", None) is not None else es_cfg.get("phase3_monitor", es_cfg.get("monitor", "val_target_loss"))
        validate_monitor_name("phase3", phase3_monitor)
        phase3_patience = args.phase3_patience if args.phase3_patience is not None else es_cfg.get("phase3_patience", es_cfg.get("patience", 5))
        min_delta = es_cfg.get("min_delta", 0.0)
        es_handler = EarlyStopping(patience=phase3_patience, min_delta=min_delta, monitor=phase3_monitor)
        tqdm.write(f"  {BOLD}{YELLOW}[Early Stop]{RESET} Phase 3 Early stopping: monitor={phase3_monitor}, patience={phase3_patience}, min_delta={min_delta}")
        
    for epoch in range(phase3_epochs):
        # Calculate current beta for PCBM KL Divergence
        warmup_epochs = getattr(args, "pcbm_beta_warmup_epochs", 10)
        anneal_epochs = getattr(args, "pcbm_beta_anneal_epochs", 10)
        target_beta = getattr(args, "pcbm_beta", 0.001)
        beta_min = getattr(args, "pcbm_beta_min", 0.0001)
        
        if epoch < warmup_epochs:
            current_beta = 0.0
        else:
            if epoch >= warmup_epochs + anneal_epochs:
                current_beta = target_beta
            else:
                ratio = (epoch - warmup_epochs) / anneal_epochs
                current_beta = beta_min + (target_beta - beta_min) * ratio
                
        if getattr(model, "use_probabilistic_cbm", False):
            tqdm.write(f"  {BOLD}{CYAN}[PCBM Beta]{RESET} Epoch {epoch+1} KL Weight Beta: {current_beta:.6f}")

        model.train()
        total_loss_joint = 0.0
        total_loss_t = 0.0
        total_loss_c = 0.0
        total_acc_t = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"  P3 Epoch {epoch+1}/{phase3_epochs}", bar_format="{l_bar}{bar:25}{r_bar}", leave=False)
        for images, concepts, targets in train_pbar:
            images = images.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Manual forward pass to support Scheduled Sampling end-to-end in Phase 3
            features = model.backbone(images)
            if isinstance(features, tuple):
                features = features[0]
                
            if model.backbone_name.startswith('resnet'):
                if getattr(model, "use_probabilistic_cbm", False):
                    supervised_mean, supervised_logvar, supervised_attn, supervised_features = model.supervised_attention(features)
                    model.last_mean = supervised_mean
                    model.last_logvar = supervised_logvar
                    if model.training:
                        std = torch.exp(0.5 * supervised_logvar)
                        eps = torch.randn_like(std)
                        supervised_logits = supervised_mean + std * eps
                    else:
                        supervised_logits = supervised_mean
                else:
                    supervised_logits, supervised_attn, supervised_features = model.supervised_attention(features)
                
                # Apply scheduled sampling (concept noise injection) if enabled in Phase 3
                if getattr(args, "phase2_scheduled_sampling", False):
                    supervised_logits = inject_concept_noise(
                        pred_logits=supervised_logits,
                        gt_labels=concepts[:, :num_concepts_supervised],
                        replace_prob=getattr(args, "scheduled_sampling_prob", 0.3),
                        epsilon=getattr(args, "scheduled_sampling_epsilon", 0.05)
                    )
                    
                if model.num_latent_concepts > 0:
                    if getattr(model, "use_probabilistic_cbm", False):
                        latent_mean, latent_logvar, latent_attn, latent_features = model.latent_attention(features)
                        if model.training:
                            std_l = torch.exp(0.5 * latent_logvar)
                            eps_l = torch.randn_like(std_l)
                            latent_logits = latent_mean + std_l * eps_l
                        else:
                            latent_logits = latent_mean
                    else:
                        latent_logits, latent_attn, latent_features = model.latent_attention(features)
                    concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
                    attn_weights = torch.cat([supervised_attn, latent_attn], dim=1)
                else:
                    concept_logits = supervised_logits
                    attn_weights = supervised_attn
                    latent_features = None
            else:
                # ViT / DINOv2 backbones
                if model.use_group_broadcasting:
                    # GroupToConceptAttention
                    if getattr(model, "use_probabilistic_cbm", False):
                        supervised_mean, supervised_logvar, attn_weights, concept_features = model.supervised_attention(features)
                        model.last_mean = supervised_mean
                        model.last_logvar = supervised_logvar
                        if model.training:
                            std = torch.exp(0.5 * supervised_logvar)
                            eps = torch.randn_like(std)
                            concept_logits = supervised_mean + std * eps
                        else:
                            concept_logits = supervised_mean
                    else:
                        concept_logits, attn_weights, concept_features = model.supervised_attention(features)
                    supervised_features = concept_features
                    
                    # Apply scheduled sampling (concept noise injection) if enabled in Phase 3
                    if getattr(args, "phase2_scheduled_sampling", False):
                        concept_logits = inject_concept_noise(
                            pred_logits=concept_logits,
                            gt_labels=concepts[:, :num_concepts_supervised],
                            replace_prob=getattr(args, "scheduled_sampling_prob", 0.3),
                            epsilon=getattr(args, "scheduled_sampling_epsilon", 0.05)
                        )
                    latent_features = None
                elif getattr(model, "use_concept_attention", False):
                    # ViTCrossAttentionLayer
                    if getattr(model, "use_probabilistic_cbm", False):
                        supervised_mean, supervised_logvar, attn_weights, supervised_features = model.supervised_attention(features)
                        model.last_mean = supervised_mean
                        model.last_logvar = supervised_logvar
                        if model.training:
                            std = torch.exp(0.5 * supervised_logvar)
                            eps = torch.randn_like(std)
                            supervised_logits = supervised_mean + std * eps
                        else:
                            supervised_logits = supervised_mean
                    else:
                        supervised_logits, attn_weights, supervised_features = model.supervised_attention(features)
                    
                    # Apply scheduled sampling (concept noise injection) if enabled in Phase 3
                    if getattr(args, "phase2_scheduled_sampling", False):
                        supervised_logits = inject_concept_noise(
                            pred_logits=supervised_logits,
                            gt_labels=concepts[:, :num_concepts_supervised],
                            replace_prob=getattr(args, "scheduled_sampling_prob", 0.3),
                            epsilon=getattr(args, "scheduled_sampling_epsilon", 0.05)
                        )
                    
                    supervised_attn = attn_weights
                    
                    if model.num_latent_concepts > 0:
                        # Latent attention is PatchWiseMLPConceptHead for ViT
                        k_val = 3
                        B_size = features.size(0)
                        N_patches = features.size(1)
                        H_attn = int(N_patches ** 0.5)
                        D_dim = features.size(-1)
                        
                        if getattr(model, "use_probabilistic_cbm", False):
                            latent_mean, latent_logvar, latent_topk_indices, latent_weights = model.latent_attention(features, return_weights=True)
                            if model.training:
                                std_l = torch.exp(0.5 * latent_logvar)
                                eps_l = torch.randn_like(std_l)
                                latent_logits = latent_mean + std_l * eps_l
                            else:
                                latent_logits = latent_mean
                        else:
                            latent_logits, latent_topk_indices, latent_weights = model.latent_attention(features, return_weights=True)
                        
                        latent_indices_transposed = latent_topk_indices.permute(0, 2, 1)
                        latent_weights_transposed = latent_weights.permute(0, 2, 1)
                        latent_k_val = latent_topk_indices.size(1)
                        
                        from torchvision.transforms.functional import gaussian_blur
                        sparse_latent_maps = torch.zeros(B_size, model.num_latent_concepts, N_patches, device=device)
                        sparse_latent_maps.scatter_(2, latent_indices_transposed, latent_weights_transposed)
                        sparse_latent_maps = sparse_latent_maps.view(B_size, model.num_latent_concepts, H_attn, H_attn)
                        latent_attn = gaussian_blur(sparse_latent_maps, kernel_size=[3, 3], sigma=[1.0, 1.0])
                        
                        latent_flat_indices = latent_indices_transposed.reshape(B_size, model.num_latent_concepts * latent_k_val)
                        latent_gathered_flat = torch.gather(
                            features,
                            dim=1,
                            index=latent_flat_indices.unsqueeze(-1).expand(-1, -1, D_dim)
                        )
                        latent_gathered_features = latent_gathered_flat.view(B_size, model.num_latent_concepts, latent_k_val, D_dim)
                        latent_features = torch.sum(latent_gathered_features * latent_weights_transposed.unsqueeze(-1), dim=2)
                        
                        concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
                        attn_weights = torch.cat([supervised_attn, latent_attn], dim=1)
                    else:
                        concept_logits = supervised_logits
                        latent_features = None
                else:
                    # PatchWiseMLPConceptHead
                    if getattr(model, "use_probabilistic_cbm", False):
                        supervised_mean, supervised_logvar, supervised_topk_indices, supervised_weights = model.supervised_attention(features, return_weights=True)
                        model.last_mean = supervised_mean
                        model.last_logvar = supervised_logvar
                        if model.training:
                            std = torch.exp(0.5 * supervised_logvar)
                            eps = torch.randn_like(std)
                            supervised_logits = supervised_mean + std * eps
                        else:
                            supervised_logits = supervised_mean
                    else:
                        supervised_logits, supervised_topk_indices, supervised_weights = model.supervised_attention(features, return_weights=True)
                    
                    # Apply scheduled sampling (concept noise injection) if enabled in Phase 3
                    if getattr(args, "phase2_scheduled_sampling", False):
                        supervised_logits = inject_concept_noise(
                            pred_logits=supervised_logits,
                            gt_labels=concepts[:, :num_concepts_supervised],
                            replace_prob=getattr(args, "scheduled_sampling_prob", 0.3),
                            epsilon=getattr(args, "scheduled_sampling_epsilon", 0.05)
                        )
                        
                    # Reconstruct spatial maps and gathered features
                    k_val = supervised_topk_indices.size(1)
                    indices_transposed = supervised_topk_indices.permute(0, 2, 1) # [B, num_concepts_supervised, k]
                    weights_transposed = supervised_weights.permute(0, 2, 1) # [B, num_concepts_supervised, k]
                    
                    B_size = features.size(0)
                    N_patches = features.size(1)
                    H_attn = int(N_patches ** 0.5)
                    D_dim = features.size(-1)
                    
                    sparse_maps = torch.zeros(B_size, num_concepts_supervised, N_patches, device=device)
                    sparse_maps.scatter_(2, indices_transposed, weights_transposed)
                    sparse_maps = sparse_maps.view(B_size, num_concepts_supervised, H_attn, H_attn)
                    from torchvision.transforms.functional import gaussian_blur
                    supervised_attn = gaussian_blur(sparse_maps, kernel_size=[3, 3], sigma=[1.0, 1.0])
                    
                    flat_indices = indices_transposed.reshape(B_size, num_concepts_supervised * k_val)
                    gathered_flat = torch.gather(
                        features,
                        dim=1,
                        index=flat_indices.unsqueeze(-1).expand(-1, -1, D_dim)
                    )
                    gathered_features = gathered_flat.view(B_size, num_concepts_supervised, k_val, D_dim)
                    supervised_features = torch.sum(gathered_features * weights_transposed.unsqueeze(-1), dim=2)
                    
                    if model.num_latent_concepts > 0:
                        if getattr(model, "use_probabilistic_cbm", False):
                            latent_mean, latent_logvar, latent_topk_indices, latent_weights = model.latent_attention(features, return_weights=True)
                            if model.training:
                                std_l = torch.exp(0.5 * latent_logvar)
                                eps_l = torch.randn_like(std_l)
                                latent_logits = latent_mean + std_l * eps_l
                            else:
                                latent_logits = latent_mean
                        else:
                            latent_logits, latent_topk_indices, latent_weights = model.latent_attention(features, return_weights=True)
                        
                        latent_indices_transposed = latent_topk_indices.permute(0, 2, 1)
                        latent_weights_transposed = latent_weights.permute(0, 2, 1)
                        latent_k_val = latent_topk_indices.size(1)
                        
                        sparse_latent_maps = torch.zeros(B_size, model.num_latent_concepts, N_patches, device=device)
                        sparse_latent_maps.scatter_(2, latent_indices_transposed, latent_weights_transposed)
                        sparse_latent_maps = sparse_latent_maps.view(B_size, model.num_latent_concepts, H_attn, H_attn)
                        latent_attn = gaussian_blur(sparse_latent_maps, kernel_size=[3, 3], sigma=[1.0, 1.0])
                        
                        latent_flat_indices = latent_indices_transposed.reshape(B_size, model.num_latent_concepts * latent_k_val)
                        latent_gathered_flat = torch.gather(
                            features,
                            dim=1,
                            index=latent_flat_indices.unsqueeze(-1).expand(-1, -1, D_dim)
                        )
                        latent_gathered_features = latent_gathered_flat.view(B_size, model.num_latent_concepts, latent_k_val, D_dim)
                        latent_features = torch.sum(latent_gathered_features * latent_weights_transposed.unsqueeze(-1), dim=2)
                        
                        concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
                        attn_weights = torch.cat([supervised_attn, latent_attn], dim=1)
                    else:
                        concept_logits = supervised_logits
                        attn_weights = supervised_attn
                        latent_features = None
                        
            # Pass concept_logits through dropout and classifier head (end-to-end gradients intact)
            concept_logits_for_classifier = concept_logits
            intervention_prob = getattr(args, "intervention_prob", 0.0)
            if intervention_prob > 0.0 and torch.rand(1).item() < intervention_prob:
                supervised_logits_injected = inject_concept_noise(
                    pred_logits=concept_logits[:, :num_concepts_supervised],
                    gt_labels=concepts[:, :num_concepts_supervised],
                    replace_prob=getattr(args, "scheduled_sampling_prob", 0.3),
                    epsilon=getattr(args, "scheduled_sampling_epsilon", 0.05)
                )
                if model.num_latent_concepts > 0:
                    concept_logits_for_classifier = torch.cat([supervised_logits_injected, concept_logits[:, num_concepts_supervised:]], dim=1)
                else:
                    concept_logits_for_classifier = supervised_logits_injected

            concept_logits_for_classifier = model.apply_concept_bias(concept_logits_for_classifier)
            concept_logits_dropout = model.dropout(concept_logits_for_classifier)
            class_logits = model.classifier_head(concept_logits_dropout)
            
            # 1. Target Loss
            if isinstance(target_criterion, nn.BCEWithLogitsLoss):
                loss_t = target_criterion(class_logits, targets)
            elif num_classes == 1:
                loss_t = target_criterion(class_logits, targets)
            else:
                loss_t = target_criterion(class_logits, targets.view(-1).long())
                
            # 2. Concept Loss (on supervised concept predictions)
            supervised_logits = concept_logits[:, :num_concepts_supervised]
            
            # Apply label smoothing if configured to maintain logit range consistency
            smooth_epsilon = getattr(args, "phase1_label_smoothing", 0.05)
            smoothed_concepts = apply_label_smoothing(concepts, epsilon=smooth_epsilon)
            
            loss_c = concept_criterion(supervised_logits, smoothed_concepts)
            if getattr(model, "use_probabilistic_cbm", False):
                mean = model.last_mean
                logvar = model.last_logvar
                kl_loss_raw = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
                
                # Filter out ignored/missing concepts (< 0.0, e.g. -1)
                valid_mask = (concepts >= 0.0)
                pos_mask = (concepts > 0.5) & valid_mask
                neg_mask = (concepts <= 0.5) & valid_mask
                
                # Inverse frequency weighting to balance positive/negative components and prevent negative bias dominance
                pos_sum = (kl_loss_raw * pos_mask.float()).sum(dim=0)
                neg_sum = (kl_loss_raw * neg_mask.float()).sum(dim=0)
                
                P_c = pos_mask.sum(dim=0).float()
                N_c = neg_mask.sum(dim=0).float()
                
                # Prevent division by zero to avoid NaN generation in torch.where
                P_c_safe = torch.where(P_c > 0, P_c, torch.ones_like(P_c))
                N_c_safe = torch.where(N_c > 0, N_c, torch.ones_like(N_c))
                
                avg_kl_pos = torch.where(P_c > 0, pos_sum / P_c_safe, torch.zeros_like(pos_sum))
                avg_kl_neg = torch.where(N_c > 0, neg_sum / N_c_safe, torch.zeros_like(neg_sum))
                
                asym_weight = getattr(args, "pcbm_asymmetric_kl_weight", 0.1)
                kl_loss_per_concept = avg_kl_neg + asym_weight * avg_kl_pos
                kl_loss = kl_loss_per_concept.mean()
                
                loss_c = loss_c + current_beta * kl_loss
            
            # 3. Latent Regularization Losses
            loss_latent_ortho = torch.tensor(0.0, device=device)
            loss_latent_l1 = torch.tensor(0.0, device=device)
            
            concept_probs = model.concept_activation(model.apply_concept_bias(concept_logits))
            
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
            
            # L1 Lasso Regularization on classifier_head parameters (increased to 0.05 during Phase 3)
            l1_lambda = getattr(args, "phase3_l1_lambda", 0.05)
            if l1_lambda > 0:
                if hasattr(model.classifier_head, "get_sparsity_loss"):
                    latent_penalty_scale = getattr(args, "latent_penalty_scale", 1.0)
                    total_loss = total_loss + l1_lambda * model.classifier_head.get_sparsity_loss(latent_penalty_scale=latent_penalty_scale)
                else:
                    l1_norm = sum(p.abs().sum() for p in model.classifier_head.parameters())
                    total_loss = total_loss + l1_lambda * l1_norm
                
            total_loss.backward()
            optimizer.step()
            
            # Apply proximal hard-thresholding to GatedSparseNAMHead gates for exact sparsity
            if hasattr(model.classifier_head, 'concept_gates'):
                with torch.no_grad():
                    threshold = 0.05
                    model.classifier_head.concept_gates.copy_(
                        torch.where(
                            model.classifier_head.concept_gates.abs() < threshold,
                            torch.zeros_like(model.classifier_head.concept_gates),
                            model.classifier_head.concept_gates
                        )
                    )
            if hasattr(model.classifier_head, 'latent_gates') and model.classifier_head.latent_gates is not None:
                with torch.no_grad():
                    threshold = 0.05
                    model.classifier_head.latent_gates.copy_(
                        torch.where(
                            model.classifier_head.latent_gates.abs() < threshold,
                            torch.zeros_like(model.classifier_head.latent_gates),
                            model.classifier_head.latent_gates
                        )
                    )
            
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
        val_loss_joint = 0.0
        val_loss_t = 0.0
        val_loss_c = 0.0
        val_acc_t = 0.0
        
        with torch.no_grad():
            for val_images, val_concepts, val_targets in val_loader:
                val_images = val_images.to(device, non_blocking=True)
                val_concepts = val_concepts.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True)
                
                v_class_logits, v_concept_logits, _, v_supervised_features, v_latent_features = model(val_images, return_features=True)
                
                if isinstance(target_criterion, nn.BCEWithLogitsLoss):
                    v_loss_t = target_criterion(v_class_logits, val_targets)
                elif num_classes == 1:
                    v_loss_t = target_criterion(v_class_logits, val_targets)
                else:
                    v_loss_t = target_criterion(v_class_logits, val_targets.view(-1).long())

                val_supervised_concepts = val_concepts[:, :num_concepts_supervised]
                val_smoothed_concepts = apply_label_smoothing(
                    val_supervised_concepts,
                    epsilon=getattr(args, "phase1_label_smoothing", 0.05)
                )
                v_loss_c = concept_criterion(
                    v_concept_logits[:, :num_concepts_supervised],
                    val_smoothed_concepts
                )
                if getattr(model, "use_probabilistic_cbm", False):
                    mean = model.last_mean
                    logvar = model.last_logvar
                    kl_loss_raw = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())

                    valid_mask = (val_supervised_concepts >= 0.0)
                    pos_mask = (val_supervised_concepts > 0.5) & valid_mask
                    neg_mask = (val_supervised_concepts <= 0.5) & valid_mask

                    pos_sum = (kl_loss_raw * pos_mask.float()).sum(dim=0)
                    neg_sum = (kl_loss_raw * neg_mask.float()).sum(dim=0)

                    P_c = pos_mask.sum(dim=0).float()
                    N_c = neg_mask.sum(dim=0).float()
                    P_c_safe = torch.where(P_c > 0, P_c, torch.ones_like(P_c))
                    N_c_safe = torch.where(N_c > 0, N_c, torch.ones_like(N_c))

                    avg_kl_pos = torch.where(P_c > 0, pos_sum / P_c_safe, torch.zeros_like(pos_sum))
                    avg_kl_neg = torch.where(N_c > 0, neg_sum / N_c_safe, torch.zeros_like(neg_sum))

                    asym_weight = getattr(args, "pcbm_asymmetric_kl_weight", 0.1)
                    kl_loss = (avg_kl_neg + asym_weight * avg_kl_pos).mean()
                    v_loss_c = v_loss_c + current_beta * kl_loss

                v_loss_latent_ortho = torch.tensor(0.0, device=device)
                v_loss_latent_l1 = torch.tensor(0.0, device=device)
                if model.num_latent_concepts > 0 and v_latent_features is not None:
                    explicit_norm = F.normalize(v_supervised_features, p=2, dim=-1)
                    latent_norm = F.normalize(v_latent_features, p=2, dim=-1)
                    cos_sim = torch.bmm(latent_norm, explicit_norm.transpose(1, 2))
                    v_loss_latent_ortho = (cos_sim ** 2).mean()

                    v_concept_probs = model.concept_activation(v_concept_logits)
                    latent_activations = v_concept_probs[:, num_concepts_supervised:]
                    v_loss_latent_l1 = latent_activations.abs().mean()

                v_loss_joint = v_loss_t + (args.lambda_c * v_loss_c) + (args.lambda_latent_ortho * v_loss_latent_ortho) + (args.lambda_latent_l1 * v_loss_latent_l1)
                l1_lambda = getattr(args, "phase3_l1_lambda", 0.05)
                if l1_lambda > 0:
                    if hasattr(model.classifier_head, "get_sparsity_loss"):
                        latent_penalty_scale = getattr(args, "latent_penalty_scale", 1.0)
                        v_loss_joint = v_loss_joint + l1_lambda * model.classifier_head.get_sparsity_loss(latent_penalty_scale=latent_penalty_scale)
                    else:
                        l1_norm = sum(p.abs().sum() for p in model.classifier_head.parameters())
                        v_loss_joint = v_loss_joint + l1_lambda * l1_norm
                    
                val_loss_joint += v_loss_joint.item()
                val_loss_t += v_loss_t.item()
                val_loss_c += v_loss_c.item()
                val_acc_t += calculate_accuracy(v_class_logits, val_targets)
                
        avg_val_loss_joint = val_loss_joint / len(val_loader)
        avg_val_loss_t = val_loss_t / len(val_loader)
        avg_val_loss_c = val_loss_c / len(val_loader)
        avg_val_acc_t = val_acc_t / len(val_loader)
        
        tqdm.write(f"{BOLD}{MAGENTA}[Phase 3]{RESET} Epoch {epoch+1:02d}/{phase3_epochs:02d} | Train Joint Loss: {avg_loss_joint:.4f} | Val Joint Loss: {avg_val_loss_joint:.4f} | Val Target Loss: {avg_val_loss_t:.4f} | Val Target Acc: {BOLD}{GREEN}{avg_val_acc_t * 100:.2f}%{RESET}")
        
        # Track Active Gates if GatedSparseNAMHead is used
        active_count = None
        gate_mean_val = None
        if hasattr(model.classifier_head, 'concept_gates'):
            with torch.no_grad():
                gates = model.classifier_head.concept_gates.detach().cpu()
                active_count = (gates.abs() > 0.0).sum().item()
                gate_mean_val = gates.abs().mean().item()
                tqdm.write(f"  {BOLD}{CYAN}[NAM Gating]{RESET} Active Gates: {active_count}/{gates.size(0)} | Gate Mean: {gate_mean_val:.4f}")
        if hasattr(model.classifier_head, 'latent_gates') and model.classifier_head.latent_gates is not None:
            with torch.no_grad():
                l_gates = model.classifier_head.latent_gates.detach().cpu()
                active_l_count = (l_gates.abs() > 0.0).sum().item()
                tqdm.write(f"  {BOLD}{CYAN}[NAM Latent Gating]{RESET} Active Latent Gates: {active_l_count}/{l_gates.size(0)} | Latent Mean: {l_gates.abs().mean().item():.4f}")
        
        if scheduler is not None:
            scheduler.step()
            
        early_stop_triggered = False
        if es_handler is not None:
            phase3_metrics = {
                "val_target_loss": avg_val_loss_t,
                "val_joint_loss": avg_val_loss_joint,
                "val_concept_loss": avg_val_loss_c,
                "val_target_acc": avg_val_acc_t,
            }
            monitor_score = get_monitor_score("phase3", es_handler.monitor, phase3_metrics)
            es_handler(monitor_score, model)
            early_stop_triggered = es_handler.early_stop
                
        if args.use_wandb:
            import wandb
            log_dict = {
                "phase3_epoch": epoch + 1,
                "train/joint_loss": avg_loss_joint,
                "train/joint_target_loss": avg_loss_t,
                "train/joint_concept_loss": avg_loss_c,
                "val/joint_loss": avg_val_loss_joint,
                "val/joint_target_loss": avg_val_loss_t,
                "val/joint_concept_loss": avg_val_loss_c,
                "val/joint_accuracy": avg_val_acc_t
            }
            if active_count is not None:
                log_dict.update({
                    "val/active_gates": active_count,
                    "val/gate_mean": gate_mean_val
                })
            if es_handler is not None:
                log_dict.update({
                    "early_stop/phase3_counter": es_handler.counter,
                    "early_stop/phase3_triggered": early_stop_triggered,
                })
            wandb.log(log_dict)

        if early_stop_triggered:
            tqdm.write(f"  {BOLD}{YELLOW}[Early Stop]{RESET} Early stopping Phase 3 at Epoch {epoch + 1}. Restoring best Phase 3 weights.")
            model.load_state_dict(es_handler.best_weights)
            break

    # Phase 3 루프 완료 후 (조기 종료되지 않고 완주한 경우에도) 베스트 가중치 복원
    if es_handler is not None and not es_handler.early_stop and es_handler.best_weights is not None:
        tqdm.write(f"  {BOLD}{YELLOW}[Restore]{RESET} Phase 3 completed. Restoring best Phase 3 weights.")
        model.load_state_dict(es_handler.best_weights)

    if hasattr(model.classifier_head, 'concept_gates'):
        with torch.no_grad():
            gates = model.classifier_head.concept_gates.detach().cpu()
            final_active_count = (gates.abs() > 0.0).sum().item()
            tqdm.write(f"  {BOLD}{GREEN}[NAM Gating Post-Phase 3]{RESET} Final Active Gates (threshold=0.05): {final_active_count}/{gates.size(0)}")
    if hasattr(model.classifier_head, 'latent_gates') and model.classifier_head.latent_gates is not None:
        with torch.no_grad():
            l_gates = model.classifier_head.latent_gates.detach().cpu()
            final_l_active = (l_gates.abs() > 0.0).sum().item()
            tqdm.write(f"  {BOLD}{GREEN}[NAM Latent Gating Post-Phase 3]{RESET} Final Active Latent Gates (threshold=0.05): {final_l_active}/{l_gates.size(0)}")

    # Restore original dropout rate
    if hasattr(model, 'dropout'):
        model.dropout.p = original_dropout_p
        tqdm.write(f"  {BOLD}{YELLOW}[Dropout]{RESET} Restored dropout probability after Phase 3: {model.dropout.p}")
