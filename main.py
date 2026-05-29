import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    parser = argparse.ArgumentParser(description="Train a Modular CBM")
    choices = get_dataset_choices()
    parser.add_argument('--dataset', type=str, default='milk10k', choices=choices)
    parser.add_argument('--csv_path', type=str, default=None, help="Path to metadata CSV. If omitted, uses dataset default.")
    parser.add_argument('--image_dir', type=str, default=None, help="Directory containing images. If omitted, uses dataset default.")
    parser.add_argument('--backbone_type', type=str, default='timm', choices=['timm', 'clip'])
    parser.add_argument('--backbone_name', type=str, default='resnet50')
    parser.add_argument('--num_concepts', type=int, default=None, help="Deprecated/optional. Number of concepts is auto-inferred from concept_cols.")
    parser.add_argument('--concept_cols', type=str, default=None, help="Comma-separated list of concept columns in the CSV. If omitted, auto-detects MONET_ columns.")
    parser.add_argument('--concept_config_path', type=str, default=None, help="Path to the JSON/YAML concept config file for dynamic CBM bottleneck extraction.")
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_c', type=float, default=1.0, help="Weight for the concept loss.")
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze vision backbone parameters.")
    parser.add_argument('--freeze_head', action='store_true', help="Freeze classifier head parameters.")
    parser.add_argument('--use_wandb', type=str2bool, default=True, help="Whether to use Weights & Biases logging.")
    parser.add_argument('--save_dir', type=str, default='checkpoints', help="Directory to save model weights.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
        config=dataset_config
    )
    val_dataset = dataset_class(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        split='val',
        config=dataset_config
    )

    # Use final resolved configuration from dataset instance
    resolved_config = train_dataset.config
    num_concepts = resolved_config["num_concepts"]
    num_classes = resolved_config["num_classes"]

    print(f"Loaded train dataset '{args.dataset}' with resolved config: {resolved_config}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Model Initialization
    print(f"Initializing UniversalFlexibleCBM with {args.backbone_type} ({args.backbone_name})...")
    model = UniversalFlexibleCBM(
        backbone_type=args.backbone_type,
        backbone_name=args.backbone_name,
        num_concepts=num_concepts,
        num_classes=num_classes
    )
    
    if args.freeze_backbone:
        print("Freezing backbone parameters.")
        model.freeze_backbone()
        
    if args.freeze_head:
        print("Freezing classifier head parameters.")
        model.freeze_classifier()
        
    model.to(device)

    # 3. Loss & Optimizer
    concept_criterion = nn.BCELoss()
    
    if num_classes == 1:
        target_criterion = nn.BCEWithLogitsLoss()
    else:
        target_criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Create timestamp for run names and filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 3b. Weights & Biases Initialization
    if args.use_wandb:
        import wandb
        print("Initializing Weights & Biases run...")
        wandb.init(
            project="cbm-pipeline",
            name=f"{args.backbone_name}-cbm-{timestamp}",
            config=vars(args)
        )

    # 4. Training Loop
    print(f"Starting training for {args.epochs} epoch(s)...")
    for epoch in range(args.epochs):
        model.train()
        total_concept_loss = 0.0
        total_target_loss = 0.0
        total_concept_acc = 0.0
        total_target_acc = 0.0
        
        for batch_idx, (images, concepts, targets) in enumerate(train_loader):
            images = images.to(device)
            concepts = concepts.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_logits, concept_logits, attn_weights = model(images)
            concept_probs = torch.sigmoid(concept_logits)
            
            # Loss calculation
            loss_c = concept_criterion(concept_probs, concepts)
            
            if num_classes == 1:
                loss_t = target_criterion(class_logits, targets)
            else:
                loss_t = target_criterion(class_logits, targets.view(-1).long())
                
            loss = loss_t + args.lambda_c * loss_c
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_concept_loss += loss_c.item()
            total_target_loss += loss_t.item()
            total_concept_acc += calculate_concept_accuracy(concept_probs.detach(), concepts)
            total_target_acc += calculate_accuracy(class_logits.detach(), targets)
            
        # Train stats
        avg_concept_loss = total_concept_loss / len(train_loader)
        avg_target_loss = total_target_loss / len(train_loader)
        avg_concept_acc = total_concept_acc / len(train_loader)
        avg_target_acc = total_target_acc / len(train_loader)
        
        # Validation Evaluation
        model.eval()
        val_concept_loss = 0.0
        val_target_loss = 0.0
        val_concept_acc = 0.0
        val_target_acc = 0.0
        val_visualized = False
        
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
                
                # Visual overlay heatmap logging on the first validation batch of each epoch
                if not val_visualized:
                    num_samples = min(4, val_images.size(0))
                    vis_images = val_images[:num_samples]
                    vis_attn = v_attn_weights[:num_samples]
                    
                    heatmap_images = generate_concept_heatmaps(
                        image_tensor=vis_images,
                        attn_weights=vis_attn,
                        concept_names=resolved_config["concepts"]
                    )
                    
                    # 1. Always save heatmaps locally to 'visualizations/' folder
                    vis_dir = "visualizations"
                    os.makedirs(vis_dir, exist_ok=True)
                    for idx, img in enumerate(heatmap_images):
                        save_path = os.path.join(vis_dir, f"epoch_{epoch + 1}_sample_{idx + 1}.png")
                        img.save(save_path)
                    print(f"Saved {num_samples} validation concept heatmaps locally to '{vis_dir}/' folder.")
                    
                    # 2. Log to Weights & Biases if enabled
                    if args.use_wandb:
                        import wandb
                        wandb.log({
                            "val/concept_heatmaps": [
                                wandb.Image(img, caption=f"Validation Sample {idx + 1}")
                                for idx, img in enumerate(heatmap_images)
                            ]
                        }, commit=False)
                    
                    val_visualized = True
                
        avg_val_concept_loss = val_concept_loss / len(val_loader)
        avg_val_target_loss = val_target_loss / len(val_loader)
        avg_val_concept_acc = val_concept_acc / len(val_loader)
        avg_val_target_acc = val_target_acc / len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train C-Loss: {avg_concept_loss:.4f} | Train T-Loss: {avg_target_loss:.4f} | "
              f"Val C-Loss: {avg_val_concept_loss:.4f} | Val T-Loss: {avg_val_target_loss:.4f} | "
              f"Val C-Acc: {avg_val_concept_acc:.4f} | Val T-Acc: {avg_val_target_acc:.4f}")
              
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/total_loss": avg_concept_loss + avg_target_loss,
                "train/concept_loss": avg_concept_loss,
                "train/target_loss": avg_target_loss,
                "val/total_loss": avg_val_concept_loss + avg_val_target_loss,
                "val/concept_loss": avg_val_concept_loss,
                "val/target_loss": avg_val_target_loss,
                "val/accuracy": avg_val_target_acc,
                "val/concept_accuracy": avg_val_concept_acc
            })
              
    print("Training complete.")
    
    # Save Model Weights
    mode = "frozen_backbone" if args.freeze_backbone else "full"
    save_subdir = os.path.join(args.save_dir, args.backbone_name)
    os.makedirs(save_subdir, exist_ok=True)
    
    save_filename = f"{timestamp}_cbm_{mode}.pth"
    save_path = os.path.join(save_subdir, save_filename)
    
    print(f"Saving model weights to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Weights saved successfully.")

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
