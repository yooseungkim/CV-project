import torch
from src.models.cbm_factory import UniversalFlexibleCBM
import json
import os

def test():
    checkpoint_path = "checkpoints/vit_base_patch14_dinov2/cub_vit_base_patch14_dinov2_latent20_20260604_2055.pt"
    if not os.path.exists(checkpoint_path):
        import glob
        pths = glob.glob("checkpoints/vit_base_patch14_dinov2/*.pt")
        if pths:
            checkpoint_path = pths[-1]
            print(f"Found checkpoint at: {checkpoint_path}")
        else:
            print("No checkpoints found!")
            return
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint from {checkpoint_path} on {device}")
    
    loaded = torch.load(checkpoint_path, map_location=device)
    state_dict = loaded.get('state_dict', loaded)
    
    # Let's inspect the keys to count concepts
    num_concepts = 86 # default supervised concepts
    num_classes = 200
    latent_concepts = 20
    
    model = UniversalFlexibleCBM(
        backbone_type="timm",
        backbone_name="vit_base_patch14_dinov2",
        num_supervised_concepts=num_concepts,
        num_classes=num_classes,
        num_latent_concepts=latent_concepts,
        use_lora=True,
        lora_r=16,
        lora_alpha=16.0,
        concept_groups_info=None,
        use_cosine_attention=False,
        use_group_broadcasting=False,
        use_nam_head=True,
        nam_hidden_dim=8,
        use_probabilistic_cbm=True,
        use_concept_attention=True,
        use_pairwise_nam=False
    )
    
    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    model = model.to(device)
    model.eval()
    
    # Load real image
    from PIL import Image
    from torchvision import transforms
    img_path = "data/CUB_200_2011/images/017.Cardinal/Cardinal_0001_17057.jpg"
    print(f"Loading image from: {img_path}")
    image = Image.open(img_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        class_logits, concept_logits, attn_weights = model(img_tensor)
        
    print("Class logits stats:")
    print("  Min:", class_logits.min().item())
    print("  Max:", class_logits.max().item())
    print("  Mean:", class_logits.mean().item())
    print("  Contains NaN:", torch.isnan(class_logits).any().item())
    
    # Load class names
    classes_path = "data/CUB_200_2011/classes.txt"
    classes = []
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
                    classes.append(name)
                    
    probs = torch.softmax(class_logits, dim=-1)
    max_prob, max_idx = torch.max(probs, dim=-1)
    predicted_class_name = classes[max_idx.item()] if max_idx.item() < len(classes) else f"Class {max_idx.item()}"
    print(f"Predicted class: {predicted_class_name} (index {max_idx.item()}) with prob {max_prob.item():.4f}")

if __name__ == "__main__":
    test()
