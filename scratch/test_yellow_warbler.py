import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.cbm_factory import UniversalFlexibleCBM
from src.data.cub import CUB2011Dataset

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'checkpoints/vit_base_patch14_dinov2/cub_vit_base_patch14_dinov2_latent20_20260604_2055.pt'
    concept_config_path = 'data/CUB_200_2011/concept_config.json'
    
    # 1. Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    ckpt_args = ckpt.get('args', {})
    concept_metadata = ckpt.get('concept_metadata', {}) if isinstance(ckpt, dict) else {}
    
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
                    
    # 3. Initialize dataset to get valid indices
    use_paper = ckpt_args.get("use_paper_preprocessing", True)
    dataset_config = {
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
    if concept_metadata:
        dataset_config["concept_metadata"] = concept_metadata

    dataset = CUB2011Dataset(
        csv_path='data/CUB_200_2011/images.txt',
        image_dir='data/CUB_200_2011/images',
        split='test',
        config=dataset_config
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
    for info in dataset.concept_features_info:
        concept_groups_info.append((info["start_idx"], info["num_feats"]))
        
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
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    app_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Find all Yellow Warbler samples in the dataset
    yw_indices = []
    for i in range(len(dataset)):
        row = dataset.df.iloc[i]
        class_id = int(row['class_id'])
        if class_id == 182: # Yellow Warbler is 182 in 1-indexed
            yw_indices.append(i)
            
    print(f"Found {len(yw_indices)} Yellow Warbler images in test split.")
    
    for idx in yw_indices[:5]:
        # 1. Dataset sample (cropped)
        img_tensor, concepts, target = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            class_logits_cropped, _, _ = model(img_tensor)
            pred_class_cropped = torch.softmax(class_logits_cropped, dim=-1).argmax(dim=-1).item()
            pred_prob_cropped = torch.softmax(class_logits_cropped, dim=-1)[0, pred_class_cropped].item()
            
        # 2. Raw image (uncropped)
        row = dataset.df.iloc[idx]
        img_name = row['image_path']
        img_path = os.path.join(dataset.image_dir, str(img_name))
        raw_pil = Image.open(img_path).convert('RGB')
        app_tensor = app_transform(raw_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            class_logits_raw, _, _ = model(app_tensor)
            pred_class_raw = torch.softmax(class_logits_raw, dim=-1).argmax(dim=-1).item()
            pred_prob_raw = torch.softmax(class_logits_raw, dim=-1)[0, pred_class_raw].item()
            
        print(f"\nImage: {img_name}")
        print(f"  Cropped Pred: {pred_class_cropped} ({target_classes[pred_class_cropped]}) [Prob: {pred_prob_cropped:.4f}]")
        print(f"  Raw Pred: {pred_class_raw} ({target_classes[pred_class_raw]}) [Prob: {pred_prob_raw:.4f}]")

if __name__ == '__main__':
    test()
