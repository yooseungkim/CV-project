import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from src.models.cbm_factory import UniversalFlexibleCBM

# CheXpert specific defaults
CHEXPERT_CONCEPTS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 
    'Lung Lesion', 'Pneumonia', 'Pneumothorax', 'Pleural Other', 
    'Fracture', 'Support Devices'
]
CHEXPERT_CLASSES = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'
]

def main():
    parser = argparse.ArgumentParser(description="Visualize GatedSparseNAMHead gates and shape functions.")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/convnext_tiny/chexpert_convnext_tiny_latent0_phase1.pt",
                        help="Path to model checkpoint")
    parser.add_argument('--save_dir', type=str, default="scratch", help="Directory to save figures")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        # Try finding any checkpoint in checkpoints directory
        print(f"⚠️ Checkpoint not found at {args.checkpoint}. Searching checkpoints/...")
        found = False
        if os.path.exists("checkpoints"):
            for root, dirs, files in os.walk("checkpoints"):
                for file in files:
                    if file.endswith(".pt") or file.endswith(".pth"):
                        args.checkpoint = os.path.join(root, file)
                        print(f"🔍 Found checkpoint: {args.checkpoint}")
                        found = True
                        break
                if found:
                    break
        if not found:
            print("❌ No checkpoint found. Exiting.")
            return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Extract training args to match architecture
    ckpt_args = checkpoint.get('args', {})
    
    # Standardize names
    dataset_name = ckpt_args.get('dataset', 'chexpert')
    backbone_type = ckpt_args.get('backbone_type', 'timm')
    backbone_name = ckpt_args.get('backbone_name', 'convnext_tiny')
    use_lora = ckpt_args.get('use_lora', False)
    lora_r = ckpt_args.get('lora_r', 8)
    lora_alpha = ckpt_args.get('lora_alpha', 16.0)
    use_cosine_attention = ckpt_args.get('use_cosine_attention', False)
    use_group_broadcasting = ckpt_args.get('use_group_broadcasting', False)
    use_concept_attention = ckpt_args.get('use_concept_attention', True)
    latent_concepts = ckpt_args.get('latent_concepts', 0)
    num_classes = ckpt_args.get('num_classes', 5)
    
    # CheXpert defaults
    num_concepts_supervised = 9
    concepts = CHEXPERT_CONCEPTS
    classes = CHEXPERT_CLASSES
    
    # Instantiate the model structure
    model = UniversalFlexibleCBM(
        backbone_type=backbone_type,
        backbone_name=backbone_name,
        num_supervised_concepts=num_concepts_supervised,
        num_classes=num_classes,
        num_latent_concepts=latent_concepts,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        use_cosine_attention=use_cosine_attention,
        use_group_broadcasting=use_group_broadcasting,
        use_concept_attention=use_concept_attention,
        use_nam_head=ckpt_args.get('use_nam_head', True) or ckpt_args.get('use_gated_nam', True),
        nam_hidden_dim=ckpt_args.get('nam_hidden_dim', 16),
        use_pairwise_nam=ckpt_args.get('use_pairwise_nam', True)
    )
    
    # Load state dict non-strictly (attention keys migration might fail strictly, but classifier head will load fully)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Get classifier head
    nam_head = model.classifier_head
    if not hasattr(nam_head, 'concept_gates'):
        print("❌ Loaded model does not contain GatedSparseNAMHead! Is use_nam_head=True configured in training?")
        return
        
    print("✅ Successfully loaded GatedSparseNAMHead.")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Visualize Concept Gates (Linear Scaling Weights)
    gates = nam_head.concept_gates.detach().cpu().numpy()
    print(f"Gates values: {gates}")
    
    plt.figure(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(concepts)))
    bars = plt.barh(concepts, gates, color=colors, edgecolor='black', alpha=0.8)
    plt.axvline(x=0.05, color='red', linestyle='--', label='Pruning Threshold (0.05)')
    plt.title('GatedSparseNAMHead: Concept Gating Weights (concept_gates)', fontsize=14, pad=15)
    plt.xlabel('Gate Weight Value', fontsize=12)
    plt.ylabel('Concepts', fontsize=12)
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    gates_save_path = os.path.join(args.save_dir, "nam_gates.png")
    plt.savefig(gates_save_path, dpi=150)
    plt.close()
    print(f"💾 Saved gates plot to {gates_save_path}")
    
    # 2. Visualize Shape Functions (Contribution curves)
    # The input to classifier_head is concept logits (ranging from -6 to 6 usually).
    # Let's generate a grid of input logit values.
    grid_size = 200
    logits_grid = torch.linspace(-6.0, 6.0, grid_size).to(device)
    probs_grid = torch.sigmoid(logits_grid).cpu().numpy() # For plotting by concept probability
    
    # Plot shape functions in a grid of subplots (num_concepts x num_classes)
    fig, axes = plt.subplots(num_concepts_supervised, num_classes, figsize=(16, 22), sharex=True)
    
    for i in range(num_concepts_supervised):
        # Prepare batch input: shape [grid_size, num_concepts + latent_concepts]
        # Set all values to 0, except for concept i which varies along the logits_grid
        dummy_in = torch.zeros(grid_size, num_concepts_supervised + latent_concepts).to(device)
        dummy_in[:, i] = logits_grid
        
        with torch.no_grad():
            # Pass through NAM layers directly to isolate concept i's contribution
            # y_i = gate_i * MLP_i(x_i)
            supervised_x = dummy_in[:, :nam_head.num_concepts].unsqueeze(-1) # [200, 9, 1]
            
            # Map each concept independently
            h = F.relu(nam_head.conv1(supervised_x))
            # No dropout during evaluation
            y = nam_head.conv2(h) # [200, 9 * 5, 1]
            y = y.view(grid_size, nam_head.num_concepts, nam_head.num_classes) # [200, 9, 5]
            
            # Scale by gates
            gated_y = y * nam_head.concept_gates.view(1, nam_head.num_concepts, 1) # [200, 9, 5]
            
            # Extract contributions of concept i for all classes
            contributions = gated_y[:, i, :].cpu().numpy() # [200, 5]
            
        for j in range(num_classes):
            ax = axes[i, j]
            ax.plot(probs_grid * 100, contributions[:, j], color='darkblue', linewidth=2.5)
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            ax.grid(True, linestyle=':', alpha=0.5)
            
            # Titles and labels
            if i == 0:
                ax.set_title(classes[j], fontsize=12, fontweight='bold', pad=10)
            if j == 0:
                ax.set_ylabel(concepts[i], fontsize=10, fontweight='bold', rotation=0, labelpad=85, ha='right')
                
            # Formatting ticks
            ax.tick_params(axis='both', which='major', labelsize=9)
            
    # Set global labels
    fig.text(0.5, 0.015, 'Predicted Concept Probability (%)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.015, 0.5, 'Additive Contribution to Class Logit', va='center', rotation='vertical', fontsize=14, fontweight='bold')
    
    plt.suptitle("NAM Shape Functions (Contribution Curves per Concept-Class Pair)", fontsize=18, fontweight='bold', y=0.995)
    plt.subplots_adjust(left=0.22, right=0.96, top=0.96, bottom=0.04, hspace=0.35, wspace=0.25)
    
    curves_save_path = os.path.join(args.save_dir, "nam_shape_functions.png")
    plt.savefig(curves_save_path, dpi=150)
    plt.close()
    print(f"💾 Saved shape functions plot to {curves_save_path}")
    
    print("🎉 Visualization complete!")

if __name__ == "__main__":
    main()
