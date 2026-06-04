import os
import argparse
import yaml
import datetime
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.cub import CUB2011Dataset
from src.data.derm7pt import Derm7PtDataset

# Early stopping class
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, monitor: str = "val_loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
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


# Attributes-Only MLP Classifier
class AttributesOnlyClassifier(nn.Module):
    def __init__(self, num_concepts: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, num_concepts]
        return self.classifier(concepts)


def parse_args():
    parser = argparse.ArgumentParser(description="Attributes-Only CUB Classification Upper Bound Benchmark")
    parser.add_argument('--config_path', type=str, default='configs/cub_train_config.yaml', help="Path to config yaml")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension for MLP")
    parser.add_argument('--use_wandb', type=str, default="false")
    parser.add_argument('--cache_in_memory', type=str, default="true")
    parser.add_argument('--max_cache_size_gb', type=float, default=15.0)
    parser.add_argument('--save_dir', type=str, default='checkpoints/upperbounds')
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=16)
    
    # helper for bool conversion
    def str2bool(v):
        return str(v).lower() in ("true", "1", "yes")
        
    args = parser.parse_args()
    args.use_wandb = str2bool(args.use_wandb)
    args.cache_in_memory = str2bool(args.cache_in_memory)
    return args


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for _, concepts, targets in pbar:
        # Ignore image, only use ground-truth concepts/attributes
        concepts = concepts.to(device)
        targets = targets.squeeze(-1).long().to(device)
        
        optimizer.zero_grad()
        logits = model(concepts)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * concepts.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        pbar.set_postfix(loss=loss.item(), acc=f"{(correct/total)*100:.2f}%")
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Validating")
    for _, concepts, targets in pbar:
        concepts = concepts.to(device)
        targets = targets.squeeze(-1).long().to(device)
        
        logits = model(concepts)
        loss = criterion(logits, targets)
        
        running_loss += loss.item() * concepts.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        pbar.set_postfix(loss=loss.item(), acc=f"{(correct/total)*100:.2f}%")
        
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Load config defaults to resolve dataset paths
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    ds_cfg = config_data.get("dataset", {})
    dataset_name = ds_cfg.get("dataset", "cub").lower()
    
    if dataset_name == 'derm7pt':
        dataset_class = Derm7PtDataset
        csv_path = ds_cfg.get("csv_path", "data/derm7pt/meta/meta.csv")
        image_dir = ds_cfg.get("image_dir", "data/derm7pt/images")
        concept_config_path = ds_cfg.get("concept_config_path", "data/derm7pt/concept_config.json")
    else:
        dataset_class = CUB2011Dataset
        csv_path = ds_cfg.get("csv_path", "data/CUB_200_2011/images.txt")
        image_dir = ds_cfg.get("image_dir", "data/CUB_200_2011/images")
        concept_config_path = ds_cfg.get("concept_config_path", "data/CUB_200_2011/concept_config.json")
    
    tqdm.write(f"\n============================================================")
    tqdm.write(f"  🚀 Upper Bound Attributes-Only Classification ({dataset_name.upper()})")
    tqdm.write(f"  📂 Concept-to-Class MLP Head | Hidden Dim: {args.hidden_dim}")
    tqdm.write(f"============================================================")
    
    # 2. Build Datasets
    filter_rare_concepts = ds_cfg.get("filter_rare_concepts", False)
    use_paper_preprocessing = ds_cfg.get("use_paper_preprocessing", False)
    if (filter_rare_concepts or use_paper_preprocessing) and dataset_name == 'cub':
        filtered_path = concept_config_path.replace(".json", "_filtered.json")
        if os.path.exists(filtered_path):
            concept_config_path = filtered_path
            tqdm.write(f"     [Config] Redirected concept_config to: {concept_config_path}")
            
    dataset_config = dataset_class.get_default_config()
    dataset_config["concept_config_path"] = concept_config_path
    dataset_config["filter_rare_concepts"] = filter_rare_concepts
    dataset_config["use_paper_preprocessing"] = use_paper_preprocessing
    
    train_dataset = dataset_class(
        csv_path=csv_path,
        image_dir=image_dir,
        split='train',
        config=dataset_config,
        cache_in_memory=args.cache_in_memory,
        max_cache_size_gb=args.max_cache_size_gb
    )
    val_dataset = dataset_class(
        csv_path=csv_path,
        image_dir=image_dir,
        split='val',
        config=dataset_config,
        cache_in_memory=args.cache_in_memory,
        max_cache_size_gb=args.max_cache_size_gb
    )
    
    num_workers = args.num_workers
    if args.cache_in_memory:
        tqdm.write("  ⚡ In-memory caching enabled: Setting num_workers = 0 to eliminate multiprocessing IPC copy overhead.")
        num_workers = 0
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    num_classes = train_dataset.config["num_classes"]
    num_concepts = train_dataset.config["num_concepts"]
    tqdm.write(f"  📊 Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    tqdm.write(f"  📂 Concept Bottleneck size (Attribute dimension): {num_concepts}")
    
    # 3. Create Model
    model = AttributesOnlyClassifier(num_concepts=num_concepts, num_classes=num_classes, hidden_dim=args.hidden_dim)
    model = model.to(device)
    
    # 4. Optimizer and Scheduler setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=args.patience, monitor="val_acc")
    
    # 5. Optional WandB setup
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=f"{dataset_name.upper()}-Upperbound", name=f"attributes_only-{datetime.datetime.now().strftime('%m%d_%H%M')}", config=vars(args))
        except ImportError:
            tqdm.write("  ⚠️ wandb not installed. Disabling wandb logging.")
            args.use_wandb = False
            
    # 6. Training Loop
    for epoch in range(1, args.epochs + 1):
        tqdm.write(f"\n🎬 Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        tqdm.write(f"  📊 Epoch Result - Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[-1]["lr"]
            })
            
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            tqdm.write("🛑 Early Stopping triggered! Restoring best weights...")
            model.load_state_dict(early_stopping.best_weights)
            break
            
    # 7. Save weights
    save_path = os.path.join(args.save_dir, "upperbound_attributes_only.pt")
    torch.save({
        'state_dict': model.state_dict(),
        'args': vars(args)
    }, save_path)
    tqdm.write(f"\n============================================================")
    tqdm.write(f"  ✅ Training complete!")
    tqdm.write(f"  💾 Best Val Acc: {early_stopping.best_score*100 if early_stopping.best_score is not None else val_acc*100:.2f}%")
    tqdm.write(f"  💾 Saved weights: {save_path}")
    tqdm.write(f"============================================================\n")


if __name__ == "__main__":
    main()
