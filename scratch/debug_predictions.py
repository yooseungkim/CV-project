import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.cbm_factory import UniversalFlexibleCBM
from src.data.cub import CUB2011Dataset

def debug():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'checkpoints/vit_base_patch14_dinov2/cub_vit_base_patch14_dinov2_latent20_20260604_2055.pt'
    concept_config_path = 'data/CUB_200_2011/concept_config.json'
    
    # 1. Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    ckpt_args = ckpt.get('args', {})
    
    # 2. Get CUB dataset & target classes
    classes_path = "data/CUB_200_2011/classes.txt"
    target_classes = []
    with open(classes_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    name = " ".join(parts[1:])
                    if '.' in name:
                        name = name.split('.', 1)[1]
                    name = name.replace('_', ' ')
                    target_classes.append(name)
                    
    print(f"Total target classes: {len(target_classes)}")
    print(f"Class 0: {target_classes[0]}, Class 1: {target_classes[1]}")
    
    # 3. Initialize dataset
    # We want to use paper preprocessing if the checkpoint used it
    use_paper = ckpt_args.get("use_paper_preprocessing", True)
    dataset = CUB2011Dataset(
        csv_path='data/CUB_200_2011/images.txt',
        image_dir='data/CUB_200_2011/images',
        split='test',
        config={
            "num_concepts": 312,
            "num_classes": 200,
            "concepts": [],
            "target_col": 'class_id',
            "default_csv_path": 'data/CUB_200_2011/images.txt',
            "default_image_dir": 'data/CUB_200_2011/images',
            "filter_rare_concepts": False,
            "use_paper_preprocessing": use_paper,
            "concept_config_path": concept_config_path
        }
    )
    
    # 4. Initialize model
    num_concepts = dataset.config["num_concepts"]
    latent_concepts = ckpt_args.get("latent_concepts", 20)
    use_lora = ckpt_args.get("use_lora", True)
    lora_r = ckpt_args.get("lora_r", 16)
    lora_alpha = ckpt_args.get("lora_alpha", 16.0)
    use_concept_attention = ckpt_args.get("use_concept_attention", True)
    use_nam_head = ckpt_args.get("use_nam_head", True)
    nam_hidden_dim = ckpt_args.get("nam_hidden_dim", 8)
    use_gated_nam = ckpt_args.get("use_gated_nam", True)
    use_probabilistic_cbm = ckpt_args.get("use_probabilistic_cbm", True)
    
    # Mutually exclusive concept groups
    concept_groups_info = []
    total_dims = 0
    with open(concept_config_path, 'r') as f:
        concept_config = json.load(f)
    
    # Filter concept config matching dataset logic
    is_already_filtered = "filtered" in os.path.basename(concept_config_path)
    # The dataset filters concepts if use_paper_preprocessing is True
    # We should filter concept_config similarly
    valid_indices = dataset.valid_indices
    original_idx = 0
    concept_groups = []
    
    for name, info in concept_config.items():
        ctype = info.get("type", "numerical")
        if ctype == "categorical":
            classes = info.get("classes", [])
            valid_classes_in_group = []
            for cls_val in classes:
                if is_already_filtered or valid_indices is None or original_idx in valid_indices:
                    valid_classes_in_group.append(cls_val)
                original_idx += 1
            if len(valid_classes_in_group) > 0:
                group = {
                    "name": name,
                    "type": "categorical",
                    "classes": [str(c) for c in valid_classes_in_group],
                    "flat_indices": list(range(total_dims, total_dims + len(valid_classes_in_group)))
                }
                concept_groups.append(group)
                total_dims += len(valid_classes_in_group)
        else:
            is_valid = (is_already_filtered or valid_indices is None or original_idx in valid_indices)
            if is_valid:
                group = {
                    "name": name,
                    "type": "numerical",
                    "min": float(info.get("min", 0.0)),
                    "max": float(info.get("max", 1.0)),
                    "flat_indices": [total_dims]
                }
                concept_groups.append(group)
                total_dims += 1
            original_idx += 1

    # Print total concepts
    print(f"Computed concepts: {total_dims}")
    
    for group in concept_groups:
        start = group["flat_indices"][0]
        num = len(group["flat_indices"])
        concept_groups_info.append((start, num))
        
    model = UniversalFlexibleCBM(
        backbone_type=ckpt_args.get("backbone_type", "timm"),
        backbone_name=ckpt_args.get("backbone_name", "vit_base_patch14_dinov2"),
        num_supervised_concepts=num_concepts,
        num_classes=200,
        num_latent_concepts=latent_concepts,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        concept_groups_info=concept_groups_info,
        use_nam_head=use_nam_head or use_gated_nam,
        nam_hidden_dim=nam_hidden_dim,
        use_probabilistic_cbm=use_probabilistic_cbm,
        use_concept_attention=use_concept_attention
    )
    
    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    model.to(device)
    model.eval()
    
    app_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test on a few samples
    indices = [0, 10, 20, 50, 100, 200, 500, 1000]
    for idx in indices:
        if idx >= len(dataset):
            break
        # 1. Using dataset __getitem__ (crops if bbox exists)
        img_tensor, concepts, target = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 2. Run prediction with dataset cropped image
        with torch.no_grad():
            class_logits, concept_logits, _ = model(img_tensor)
            pred_class_cropped = torch.softmax(class_logits, dim=-1).argmax(dim=-1).item()
            pred_prob_cropped = torch.softmax(class_logits, dim=-1)[0, pred_class_cropped].item()
            
        # 3. Load raw image without crop, apply app transform
        row = dataset.df.iloc[idx]
        img_name = row['image_path']
        img_path = os.path.join(dataset.image_dir, str(img_name))
        raw_pil = Image.open(img_path).convert('RGB')
        app_tensor = app_transform(raw_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            class_logits_raw, _, _ = model(app_tensor)
            pred_class_raw = torch.softmax(class_logits_raw, dim=-1).argmax(dim=-1).item()
            pred_prob_raw = torch.softmax(class_logits_raw, dim=-1)[0, pred_class_raw].item()
            
        gt_class = target.item()
        gt_name = target_classes[gt_class]
        pred_name_cropped = target_classes[pred_class_cropped]
        pred_name_raw = target_classes[pred_class_raw]
        
        print(f"\n--- Sample {idx} ({img_name}) ---")
        print(f"  GT Class: {gt_class} ({gt_name})")
        print(f"  Cropped Pred: {pred_class_cropped} ({pred_name_cropped}) [Prob: {pred_prob_cropped:.4f}]")
        print(f"  Raw Pred: {pred_class_raw} ({pred_name_raw}) [Prob: {pred_prob_raw:.4f}]")

if __name__ == '__main__':
    debug()
