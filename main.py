import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.milk10k import MILK10KDataset
from src.models.cbm_factory import UniversalFlexibleCBM
from src.utils.metrics import calculate_accuracy, calculate_concept_accuracy

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Modular CBM")
    parser.add_argument('--dataset', type=str, default='milk10k', choices=['milk10k'])
    parser.add_argument('--csv_path', type=str, default='data/metadata.csv')
    parser.add_argument('--image_dir', type=str, default='data/images')
    parser.add_argument('--backbone_type', type=str, default='timm', choices=['timm', 'clip'])
    parser.add_argument('--backbone_name', type=str, default='resnet50')
    parser.add_argument('--num_concepts', type=int, default=7)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--freeze_backbone', action='store_true', help="Freeze vision backbone parameters.")
    parser.add_argument('--freeze_head', action='store_true', help="Freeze classifier head parameters.")
    parser.add_argument('--use_wandb', type=str2bool, default=True, help="Whether to use Weights & Biases logging.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader
    if args.dataset == 'milk10k':
        dataset = MILK10KDataset(
            csv_path=args.csv_path,
            image_dir=args.image_dir,
            target_col='Malignancy'
        )
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Model Initialization
    print(f"Initializing UniversalFlexibleCBM with {args.backbone_type} ({args.backbone_name})...")
    model = UniversalFlexibleCBM(
        backbone_type=args.backbone_type,
        backbone_name=args.backbone_name,
        num_concepts=args.num_concepts,
        num_classes=args.num_classes
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
    
    if args.num_classes == 1:
        target_criterion = nn.BCEWithLogitsLoss()
    else:
        target_criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # 3b. Weights & Biases Initialization
    if args.use_wandb:
        import wandb
        print("Initializing Weights & Biases run...")
        wandb.init(
            project="cbm-pipeline",
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
        
        for batch_idx, (images, concepts, targets) in enumerate(dataloader):
            images = images.to(device)
            concepts = concepts.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            concept_probs, class_logits = model(images)
            
            # Loss calculation
            loss_c = concept_criterion(concept_probs, concepts)
            
            if args.num_classes == 1:
                loss_t = target_criterion(class_logits, targets)
            else:
                loss_t = target_criterion(class_logits, targets.long().squeeze())
                
            loss = loss_c + loss_t
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_concept_loss += loss_c.item()
            total_target_loss += loss_t.item()
            total_concept_acc += calculate_concept_accuracy(concept_probs.detach(), concepts)
            total_target_acc += calculate_accuracy(class_logits.detach(), targets)
            
        # Epoch stats
        avg_concept_loss = total_concept_loss / len(dataloader)
        avg_target_loss = total_target_loss / len(dataloader)
        avg_concept_acc = total_concept_acc / len(dataloader)
        avg_target_acc = total_target_acc / len(dataloader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"C-Loss: {avg_concept_loss:.4f} | T-Loss: {avg_target_loss:.4f} | "
              f"C-Acc: {avg_concept_acc:.4f} | T-Acc: {avg_target_acc:.4f}")
              
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/total_loss": avg_concept_loss + avg_target_loss,
                "train/concept_loss": avg_concept_loss,
                "train/target_loss": avg_target_loss,
                "val/accuracy": avg_target_acc
            })
              
    print("Training complete.")
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
