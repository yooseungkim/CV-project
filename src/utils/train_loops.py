import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.utils.metrics import calculate_accuracy, calculate_concept_metrics, find_optimal_concept_thresholds
from src.utils.visualization import generate_concept_heatmaps
from src.utils.losses import calculate_orthogonality_loss
from src.utils.helpers import EarlyStopping

# ANSI terminal colors for highlighting
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

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
        tqdm.write(f"  {BOLD}{BLUE}[Orthogonality]{RESET} Detected {len(concept_groups_indices)} semantic attribute groups for separation.")
    
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
            images = images.to(device, non_blocking=True)
            concepts = concepts.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            _, concept_logits, attn_weights = model(images)
            
            # Apply label smoothing if configured
            smooth_epsilon = getattr(args, "phase1_label_smoothing", 0.05)
            smoothed_concepts = apply_label_smoothing(concepts, epsilon=smooth_epsilon)
            
            # Use raw logits with BCEWithLogitsLoss
            loss_c = concept_criterion(concept_logits[:, :num_concepts_supervised], smoothed_concepts)
            
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
                val_images = val_images.to(device, non_blocking=True)
                val_concepts = val_concepts.to(device, non_blocking=True)
                
                _, v_concept_logits, v_attn_weights = model(val_images)
                
                # Apply label smoothing if configured
                smooth_epsilon = getattr(args, "phase1_label_smoothing", 0.05)
                v_smoothed_concepts = apply_label_smoothing(val_concepts, epsilon=smooth_epsilon)
                
                # BCEWithLogitsLoss with raw logits
                v_loss_c = concept_criterion(v_concept_logits[:, :num_concepts_supervised], v_smoothed_concepts)
                
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
        
        # Select correct monitoring metric score dynamically
        monitor_metric = es_handler.monitor.lower()
        if "loss" in monitor_metric:
            monitor_score = avg_val_loss_c
        elif "acc" in monitor_metric or "accuracy" in monitor_metric:
            monitor_score = avg_val_acc_c
        else:
            monitor_score = avg_val_loss_c
            
        es_handler(monitor_score, model)
        
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
                tqdm.write(f"  {BOLD}{YELLOW}[Struggling Concepts]{RESET} Final Struggling Concepts (Balanced Acc): {lowest_3}")
            
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
            tqdm.write(f"  {BOLD}{YELLOW}[Early Stop]{RESET} Early stopping Phase 1 at Epoch {epoch + 1}. Restoring best Phase 1 weights.")
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
            if 'val_individual_accs' in locals():
                log_dict.update(val_individual_accs)
            wandb.log(log_dict)
            
    # ── Final Validation Optimal Threshold Search ─────────────────────────
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
    
    # 1. Compute legacy/default metrics (threshold = 0.0)
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
    
    # 2. Run Youden's J search
    tqdm.write(f"\n{BOLD}{BLUE}[Threshold Search]{RESET} Finding optimal per-concept validation thresholds using Youden's J statistic...")
    optimal_thresholds = find_optimal_concept_thresholds(
        all_val_logits,
        all_val_targets,
        concept_groups_info=concept_groups_info
    )
    
    # Register optimal thresholds in model's buffer
    model.concept_thresholds.copy_(optimal_thresholds)
    
    # 3. Compute optimal metrics
    opt_metrics = calculate_concept_metrics(
        all_val_logits,
        all_val_targets,
        concept_groups_info=concept_groups_info,
        threshold=optimal_thresholds
    )
    
    tqdm.write(f"\n{BOLD}{CYAN}[Comparison]{RESET} Phase 1 Validation side-by-side comparison:")
    tqdm.write(f"   ├─ Concept Mean Balanced Accuracy : {std_metrics['mean_balanced_acc']*100:.2f}% --> {opt_metrics['mean_balanced_acc']*100:.2f}% (J-Optimal)")
    tqdm.write(f"   ├─ Concept Mean True Positive Rate: {std_metrics['tpr']*100:.2f}% --> {opt_metrics['tpr']*100:.2f}% (J-Optimal)")
    tqdm.write(f"   └─ Concept Mean True Negative Rate: {std_metrics['tnr']*100:.2f}% --> {opt_metrics['tnr']*100:.2f}% (J-Optimal)")
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
        tqdm.write(f"  {BOLD}{YELLOW}[Early Stop]{RESET} Phase 2 Early stopping: monitor={phase2_monitor}, patience={phase2_patience}")
        
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
                    latent_logits, _, latent_features = model.latent_attention(features)
                concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
            else:
                concept_logits = supervised_logits
                latent_features = None
                
            concept_probs = model.concept_activation(concept_logits)
            concept_logits_dropout = model.dropout(concept_logits)
            class_logits = model.classifier_head(concept_logits_dropout)
            
            if num_classes == 1:
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
                val_images = val_images.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True)
                
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
        tqdm.write(f"{BOLD}{MAGENTA}[Phase 2]{RESET} Epoch {epoch+1:02d}/{phase2_epochs:02d} | Train Target Loss: {avg_loss_t:.4f} | Val Target Loss: {avg_val_loss_t:.4f} | Val Target Acc: {BOLD}{GREEN}{avg_val_acc_t * 100:.2f}%{RESET}")
        
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
                tqdm.write(f"  {BOLD}{YELLOW}[Early Stop]{RESET} Early stopping Phase 2 at Epoch {epoch + 1}. Restoring best Phase 2 weights.")
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

    # 💧 Phase 2 종료 후 드롭아웃 원복
    if hasattr(model, 'dropout'):
        model.dropout.p = original_dropout_p
        tqdm.write(f"  {BOLD}{YELLOW}[Dropout]{RESET} Restored dropout probability after Phase 2: {model.dropout.p}")

def train_phase3(model, train_loader, val_loader, target_criterion, concept_criterion, device, args, config_data, run_name, num_concepts_supervised, resolved_config, num_classes):
    tqdm.write(f"\n{BOLD}{MAGENTA}{'-'*60}{RESET}")
    tqdm.write(f"  {BOLD}{MAGENTA}[Phase 3] Backbone & Classifier Fine-Tuning (Concept Head Frozen){RESET}")
    tqdm.write(f"{BOLD}{MAGENTA}{'-'*60}{RESET}")
    
    # 1. Freeze Concept Head (supervised + latent attention) — preserve Phase 1 learned attention
    model.freeze_supervised_attention()
    model.freeze_latent_attention()
    tqdm.write(f"  {BOLD}{YELLOW}[Freeze]{RESET} Concept Head frozen: supervised & latent attention weights are fixed.")
    
    # 2. Unfreeze only LoRA backbone adapters + classifier head
    model.unfreeze_backbone()    # LoRA-active: only lora_A / lora_B params, pretrained weights stay frozen
    model.unfreeze_classifier()
    tqdm.write(f"  {BOLD}{GREEN}[Unfreeze]{RESET} Backbone (LoRA adapters) and Classifier Head unfrozen for fine-tuning.")
    
    opt_cfg = config_data.get("optimizer", {})
    opt_type = opt_cfg.get("type", "adam").lower()
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    
    # Use very small learning rate for joint tuning
    phase3_lr = args.phase3_lr if args.phase3_lr is not None else 1e-5
    phase3_epochs = args.phase3_epochs
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_names  = [n for n, p in model.named_parameters() if p.requires_grad]
    tqdm.write(f"  {BOLD}{BLUE}[Model]{RESET} Trainable parameters in Phase 3: {len(trainable_params)} tensors")
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
        tqdm.write(f"  {BOLD}{YELLOW}[Early Stop]{RESET} Phase 3 Early stopping: monitor={phase3_monitor}, patience={phase3_patience}")
        
    for epoch in range(phase3_epochs):
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
                else:
                    # PatchWiseMLPConceptHead
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
            concept_logits_dropout = model.dropout(concept_logits)
            class_logits = model.classifier_head(concept_logits_dropout)
            
            # 1. Target Loss
            if num_classes == 1:
                loss_t = target_criterion(class_logits, targets)
            else:
                loss_t = target_criterion(class_logits, targets.view(-1).long())
                
            # 2. Concept Loss (on supervised concept predictions)
            supervised_logits = concept_logits[:, :num_concepts_supervised]
            
            # Apply label smoothing if configured to maintain logit range consistency
            smooth_epsilon = getattr(args, "phase1_label_smoothing", 0.05)
            smoothed_concepts = apply_label_smoothing(concepts, epsilon=smooth_epsilon)
            
            loss_c = concept_criterion(supervised_logits, smoothed_concepts)
            
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
                val_images = val_images.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True)
                
                v_class_logits, _, _ = model(val_images)
                
                if num_classes == 1:
                    v_loss_t = target_criterion(v_class_logits, val_targets)
                else:
                    v_loss_t = target_criterion(v_class_logits, val_targets.view(-1).long())
                    
                val_loss_t += v_loss_t.item()
                val_acc_t += calculate_accuracy(v_class_logits, val_targets)
                
        avg_val_loss_t = val_loss_t / len(val_loader)
        avg_val_acc_t = val_acc_t / len(val_loader)
        
        tqdm.write(f"{BOLD}{MAGENTA}[Phase 3]{RESET} Epoch {epoch+1:02d}/{phase3_epochs:02d} | Train Joint Loss: {avg_loss_joint:.4f} | Val Target Loss: {avg_val_loss_t:.4f} | Val Target Acc: {BOLD}{GREEN}{avg_val_acc_t * 100:.2f}%{RESET}")
        
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
                tqdm.write(f"  {BOLD}{YELLOW}[Early Stop]{RESET} Early stopping Phase 3 at Epoch {epoch + 1}. Restoring best Phase 3 weights.")
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
