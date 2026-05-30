import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class ConceptMLP(nn.Module):
    def __init__(self, input_dim=312, hidden_dim=256, num_classes=200, dropout=0.3):
        super().__init__()
        # Multi-Layer Perceptron with Batch Normalization, ReLU, and Dropout
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.mlp(x)

def run_oracle_sanity_check():
    print("=" * 70)
    print("🧪 Starting CUB Concept Oracle Upper Bound (Sanity Check) MLP Experiment")
    print("=" * 70)
    
    # 1. Resolve paths
    data_root = "data/CUB_200_2011"
    images_file = os.path.join(data_root, "images.txt")
    labels_file = os.path.join(data_root, "image_class_labels.txt")
    split_file = os.path.join(data_root, "train_test_split.txt")
    attr_file = os.path.join(data_root, "attributes", "image_attribute_labels.txt")
    
    if not all(os.path.exists(f) for f in [images_file, labels_file, split_file, attr_file]):
        print("❌ Error: CUB dataset files not found at expected location.")
        return
        
    # 2. Load CUB base mappings
    print("📊 Loading metadata and labels...")
    images_df = pd.read_csv(images_file, sep=r'\s+', header=None, names=['image_id', 'image_path'])
    labels_df = pd.read_csv(labels_file, sep=r'\s+', header=None, names=['image_id', 'class_id'])
    split_df = pd.read_csv(split_file, sep=r'\s+', header=None, names=['image_id', 'is_train'])
    
    # Merge base frames
    merged = images_df.merge(labels_df, on='image_id').merge(split_df, on='image_id')
    
    # Slice deterministic splits matching CUB2011Dataset implementation
    train_raw = merged[merged['is_train'] == 1].reset_index(drop=True)
    test_val_raw = merged[merged['is_train'] == 0].reset_index(drop=True)
    
    # Deterministic split of test set into 50% val and 50% test
    shuffled_test_val = test_val_raw.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_test_val = len(shuffled_test_val)
    val_end = n_test_val // 2
    
    val_raw = shuffled_test_val.iloc[:val_end].reset_index(drop=True)
    test_raw = shuffled_test_val.iloc[val_end:].reset_index(drop=True)
    
    # 3. Load Attribute concept presence annotations [11788, 312]
    print("🧬 Loading ground-truth concept annotations...")
    attr_df = pd.read_csv(
        attr_file, sep=r'\s+', header=None, usecols=[0, 1, 2],
        names=['image_id', 'attribute_id', 'is_present']
    )
    concept_matrix = attr_df['is_present'].values.reshape(11788, 312)
    
    # Extract features and targets matching CUB split index mapping
    def prepare_tensors(df):
        image_idxs = df['image_id'].values - 1
        X = torch.tensor(concept_matrix[image_idxs], dtype=torch.float32)
        y = torch.tensor(df['class_id'].values - 1, dtype=torch.long)
        return X, y
        
    X_train, y_train = prepare_tensors(train_raw)
    X_val, y_val = prepare_tensors(val_raw)
    X_test, y_test = prepare_tensors(test_raw)
    
    print(f"   Train samples: {len(X_train)} | Inputs shape: {X_train.shape}")
    print(f"   Val samples  : {len(X_val)} | Inputs shape: {X_val.shape}")
    print(f"   Test samples : {len(X_test)} | Inputs shape: {X_test.shape}")
    
    # 4. DataLoaders
    batch_size = 128
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # 5. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConceptMLP(input_dim=312, hidden_dim=512, num_classes=200, dropout=0.35).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)
    
    # 6. Training Loop
    epochs = 80
    best_val_acc = 0.0
    best_test_acc = 0.0
    
    print("\n🚀 Training standalone Concept-to-Class MLP model...")
    print(f"   Device: {device} | Hidden dimensions: 512 | Epochs: {epochs}")
    print("-" * 70)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
                
        val_acc = val_correct / val_total
        
        # Evaluate on Test if validation is the best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Test
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits = model(batch_x)
                    _, predicted = logits.max(1)
                    test_total += batch_y.size(0)
                    test_correct += predicted.eq(batch_y).sum().item()
            best_test_acc = test_correct / test_total
            
        scheduler.step()
        
        # Log progress periodically
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Best Val: {best_val_acc*100:.2f}%")
            
    print("=" * 70)
    print("🎉 Experiment Finished successfully!")
    print(f"👉 Oracle Concept Upper Bound (Best Val Acc) : {best_val_acc * 100:.2f}%")
    print(f"👉 Oracle Concept Upper Bound (Best Test Acc): {best_test_acc * 100:.2f}%")
    print("=" * 70)
    
    # Scientific Conclusion based on Oracle Bound results
    print("\n🧠 Scientific Interpretation of this Sanity Check:")
    if best_val_acc >= 0.85:
        print("🟢 SUCCESS (Expressive Representation):")
        print("   The 312 anatomical concept attributes are mathematically sufficient to uniquely distinguish CUB species.")
        print("   Any bottleneck accuracy constraints are entirely caused by Phase 1's feature extractor learning mapping issues.")
    else:
        print("🔴 LIMITATION (Information Bottleneck):")
        print("   The 312 conceptual languages themselves lack complete discriminative power to separate the 200 bird species.")
        print("   No matter how perfectly you learn the spatial attention, this ceiling limits the final accuracy.")
        print("   Adding latent concepts (unsupervised vectors) or residual backbone pathways is recommended.")
    print("=" * 70)

if __name__ == "__main__":
    run_oracle_sanity_check()
