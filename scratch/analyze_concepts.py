import os
import json
import torch
from torch.utils.data import DataLoader
from src.data.cub import CUB2011Dataset
from src.data.derm7pt import Derm7PtDataset
from src.models.cbm_factory import UniversalFlexibleCBM
from src.utils.metrics import calculate_concept_metrics
import pandas as pd

def main():
    # Attempt to load the dry-run checkpoint, otherwise find the latest checkpoint in checkpoints folder
    checkpoint_path = "checkpoints/vit_base_patch14_dinov2/cub_vit_base_patch14_dinov2_latent0_20260601_0015.pt"
    if not os.path.exists(checkpoint_path):
        ckpt_dir = "checkpoints/vit_base_patch14_dinov2"
        if os.path.exists(ckpt_dir):
            files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt") or f.endswith(".pth")]
            if files:
                checkpoint_path = max(files, key=os.path.getmtime)
                print(f"🔍 Auto-detected latest checkpoint: {checkpoint_path}")
            else:
                print(f"❌ Error: No CBM checkpoints found in '{ckpt_dir}'. Please ensure your training finishes.")
                return
        else:
            print(f"❌ Error: Checkpoint directory '{ckpt_dir}' not found.")
            return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    loaded = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_args = loaded.get('args', {})
    state_dict = loaded.get('state_dict', loaded)

    dataset_name = checkpoint_args.get('dataset', 'cub')
    backbone_type = checkpoint_args.get('backbone_type', 'timm')
    backbone_name = checkpoint_args.get('backbone_name', 'vit_base_patch14_dinov2')
    use_lora = checkpoint_args.get('use_lora', False)
    lora_r = checkpoint_args.get('lora_r', 8)
    lora_alpha = checkpoint_args.get('lora_alpha', 16.0)
    use_group_broadcasting = checkpoint_args.get('use_group_broadcasting', False)
    use_cosine_attention = checkpoint_args.get('use_cosine_attention', False)
    latent_concepts = checkpoint_args.get('latent_concepts', 0)
    num_classes = checkpoint_args.get('num_classes', 200)

    print(f"🧬 Model Specs: dataset={dataset_name.upper()} | backbone={backbone_name} | use_lora={use_lora}")

    if dataset_name == 'derm7pt':
        dataset_class = Derm7PtDataset
        csv_path = checkpoint_args.get('csv_path', 'data/derm7pt/meta/meta.csv')
        image_dir = checkpoint_args.get('image_dir', 'data/derm7pt/images')
        concept_config_path = checkpoint_args.get('concept_config_path', 'data/derm7pt/concept_config.json')
    else:
        dataset_class = CUB2011Dataset
        csv_path = checkpoint_args.get('csv_path', 'data/CUB_200_2011/images.txt')
        image_dir = checkpoint_args.get('image_dir', 'data/CUB_200_2011/images')
        concept_config_path = checkpoint_args.get('concept_config_path', 'data/CUB_200_2011/concept_config.json')

    dataset_config = dataset_class.get_default_config()
    dataset_config["concept_config_path"] = concept_config_path

    print("📊 Loading validation split for concept analysis...")
    val_dataset = dataset_class(
        csv_path=csv_path,
        image_dir=image_dir,
        split='val',
        config=dataset_config,
        cache_in_memory=False
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    num_supervised_concepts = val_dataset.config["num_concepts"]

    # Parse concept groups
    with open(concept_config_path, 'r', encoding='utf-8') as f:
        concept_json = json.load(f)

    concept_groups = []
    total_dims = 0
    for name, info in concept_json.items():
        ctype = info.get("type", "numerical")
        if ctype == "categorical":
            classes = info.get("classes", [])
            num_feats = len(classes)
            group = {
                "name": name,
                "flat_indices": list(range(total_dims, total_dims + num_feats))
            }
            total_dims += num_feats
        else:
            group = {
                "name": name,
                "flat_indices": [total_dims]
            }
            total_dims += 1
        concept_groups.append(group)

    concept_groups_info = []
    for g in concept_groups:
        concept_groups_info.append((g["flat_indices"][0], len(g["flat_indices"])))

    group_mapping = None
    num_groups = len(concept_groups)
    if use_group_broadcasting:
        group_mapping = []
        for group_idx, g in enumerate(concept_groups):
            num_in_group = len(g["flat_indices"])
            group_mapping.extend([group_idx] * num_in_group)
        concept_groups_info_param = None
    else:
        concept_groups_info_param = concept_groups_info

    model = UniversalFlexibleCBM(
        backbone_type=backbone_type,
        backbone_name=backbone_name,
        num_supervised_concepts=num_supervised_concepts,
        num_classes=num_classes,
        num_latent_concepts=latent_concepts,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        concept_groups_info=concept_groups_info_param,
        use_cosine_attention=use_cosine_attention,
        use_group_broadcasting=use_group_broadcasting,
        num_groups=num_groups,
        group_mapping=group_mapping
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    all_val_logits = []
    all_val_targets = []

    print("🏃 Running validation inference pass...")
    with torch.no_grad():
        for images, concepts, _ in val_loader:
            images = images.to(device)
            _, concept_logits, _ = model(images)
            all_val_logits.append(concept_logits[:, :num_supervised_concepts].cpu())
            all_val_targets.append(concepts.cpu())

    all_val_logits = torch.cat(all_val_logits, dim=0)
    all_val_targets = torch.cat(all_val_targets, dim=0)

    concept_metrics = calculate_concept_metrics(
        all_val_logits,
        all_val_targets,
        concept_groups_info=concept_groups_info if not use_group_broadcasting else None,
        threshold=model.concept_thresholds.cpu()
    )

    concepts_list = val_dataset.config.get("concepts_flat", val_dataset.config.get("concepts", []))

    data = []
    num_samples = all_val_targets.size(0)
    for c in range(num_supervised_concepts):
        name = concepts_list[c] if c < len(concepts_list) else f"Concept_{c}"
        gt_positives = (all_val_targets[:, c] > 0.5).sum().item()
        frequency = gt_positives / num_samples
        bal_acc = concept_metrics["individual_balanced_acc"][c].item()
        
        preds_bin = (all_val_logits[:, c] > model.concept_thresholds[c].cpu()).float()
        targets_bin = (all_val_targets[:, c] > 0.5).float()
        tp = (preds_bin * targets_bin).sum().item()
        tn = ((1 - preds_bin) * (1 - targets_bin)).sum().item()
        fp = (preds_bin * (1 - targets_bin)).sum().item()
        fn = ((1 - preds_bin) * targets_bin).sum().item()
        
        tpr = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 1.0
        tnr = tn / (tn + fp + 1e-8) if (tn + fp) > 0 else 1.0

        data.append({
            "index": c,
            "concept_name": name,
            "gt_positives": int(gt_positives),
            "frequency_pct": frequency * 100,
            "balanced_acc_pct": bal_acc * 100,
            "tpr_pct": tpr * 100,
            "tnr_pct": tnr * 100
        })

    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by="frequency_pct", ascending=True)

    # Save to JSON for downstream code/plotting
    os.makedirs("scratch", exist_ok=True)
    json_path = "scratch/concept_analysis.json"
    df_sorted.to_json(json_path, orient="records", indent=2)
    
    # Generate Markdown Table Report
    md_content = "# 🧬 CUB Concept Frequency vs Performance Analysis\n\n"
    md_content += f"**Analyzed Checkpoint**: `{checkpoint_path}`\n\n"
    md_content += "This report summarizes the frequency of positive annotations vs the balanced accuracy of the CBM concept head, sorted from rarest to most frequent to diagnose sparse concepts learning.\n\n"
    
    md_content += "## 📉 30 Rarest Concepts (Highly Sparse)\n\n"
    md_content += "| Index | Concept Name | GT Positives | Frequency (%) | Balanced Acc (%) | TPR (%) | TNR (%) |\n"
    md_content += "|---|---|---|---|---|---|---|\n"
    for _, r in df_sorted.head(30).iterrows():
        md_content += f"| {r['index']} | {r['concept_name']} | {r['gt_positives']} | {r['frequency_pct']:.2f}% | {r['balanced_acc_pct']:.2f}% | {r['tpr_pct']:.2f}% | {r['tnr_pct']:.2f}% |\n"

    md_content += "\n## 📈 10 Most Frequent Concepts\n\n"
    md_content += "| Index | Concept Name | GT Positives | Frequency (%) | Balanced Acc (%) | TPR (%) | TNR (%) |\n"
    md_content += "|---|---|---|---|---|---|---|\n"
    for _, r in df_sorted.tail(10).iterrows():
        md_content += f"| {r['index']} | {r['concept_name']} | {r['gt_positives']} | {r['frequency_pct']:.2f}% | {r['balanced_acc_pct']:.2f}% | {r['tpr_pct']:.2f}% | {r['tnr_pct']:.2f}% |\n"

    md_path = "scratch/concept_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "="*60)
    print("  ✅ Concept Analysis Complete!")
    print(f"  💾 Full JSON statistics saved to: {json_path}")
    print(f"  📄 Readable Markdown Report saved to: {md_path}")
    print("="*60 + "\n")

    print("📊 Quick view - Top 5 Rarest Concepts:")
    for _, r in df_sorted.head(5).iterrows():
        print(f"   └─ {r['concept_name']}: Freq={r['frequency_pct']:.2f}%, GT_Pos={r['gt_positives']}, Acc={r['balanced_acc_pct']:.2f}%")

if __name__ == "__main__":
    main()
