import os
import pandas as pd
from typing import Tuple, List, Optional
import torch
from torchvision import transforms
from PIL import Image
from src.data.base_dataset import BaseDataset

class MILK10KDataset(BaseDataset):
    # Ground truth disease categories (one-hot columns in GroundTruth CSV)
    GT_LABEL_COLS = [
        'AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF',
        'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC'
    ]

    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "num_concepts": 7,
            "num_classes": len(cls.GT_LABEL_COLS),
            "concepts": [
                'MONET_ulceration_crust', 'MONET_hair', 'MONET_vasculature_vessels', 
                'MONET_erythema', 'MONET_pigmented', 'MONET_gel_water_drop_fluid_dermoscopy_liquid', 
                'MONET_skin_markings_pen_ink_purple_pen'
            ],
            "target_col": 'diagnosis_idx',
            "target_classes": cls.GT_LABEL_COLS,
            "default_csv_path": 'data/MILK10K/MILK10k_Training_Metadata.csv',
            "default_image_dir": 'data/MILK10K/MILK10k_Training_Input/MILK10k_Training_Input'
        }

    def __init__(
        self,
        csv_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        split: str = 'train',
        config: Optional[dict] = None,
        transform: Optional[transforms.Compose] = None,
        cache_in_memory: bool = False,
        max_cache_size_gb: float = 10.0
    ):
        """
        Args:
            csv_path: Path to the metadata CSV file. If None, loaded from default config.
            image_dir: Directory containing the images. If None, loaded from default config.
            split: Dataset split to return ('train', 'val', 'test').
            config: Configuration dictionary for the dataset.
            transform: torchvision transforms to apply to the images.
        """
        super().__init__()
        self.split = split.lower()
        if self.split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of 'train', 'val', 'test'")

        # Initialize config
        self.config = config or self.get_default_config()
        
        # Determine paths
        resolved_csv = csv_path or self.config.get("default_csv_path") or "data/MILK10K/MILK10k_Training_Metadata.csv"
        self.image_dir = image_dir or self.config.get("default_image_dir") or "data/MILK10K/MILK10k_Training_Input/MILK10k_Training_Input"

        self.dummy_mode = not os.path.exists(resolved_csv)
        if self.dummy_mode:
            print(f"Warning: CSV file not found at {resolved_csv}. Running in dummy mode ({self.split} split).")
            self.df = pd.DataFrame()
        else:
            full_df = pd.read_csv(resolved_csv)
            
            # Auto-merge ground truth labels if they exist
            gt_csv = os.path.join(os.path.dirname(resolved_csv), "MILK10k_Training_GroundTruth.csv")
            if os.path.exists(gt_csv):
                gt_df = pd.read_csv(gt_csv)
                if 'lesion_id' in full_df.columns and 'lesion_id' in gt_df.columns:
                    full_df = pd.merge(full_df, gt_df, on='lesion_id', how='left')
                    
                    # Build multi-class label: one-hot → integer class index
                    gt_label_cols = self.GT_LABEL_COLS
                    if all(col in full_df.columns for col in gt_label_cols):
                        full_df['diagnosis_idx'] = full_df[gt_label_cols].values.argmax(axis=1)
                        self.config["target_classes"] = gt_label_cols
                        self.config["num_classes"] = len(gt_label_cols)
                        print(f"Merged ground truth from {gt_csv} → {len(gt_label_cols)}-class classification (diagnosis_idx)")
                    
                    # Also keep legacy binary Malignancy column for backward compat
                    malignant_cols = ['AKIEC', 'BCC', 'MEL', 'SCCKA', 'MAL_OTH']
                    if all(col in full_df.columns for col in malignant_cols):
                        full_df['Malignancy'] = full_df[malignant_cols].sum(axis=1).clip(upper=1.0)
            
            # Deterministic split: 70% train, 15% val, 15% test
            shuffled_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            n = len(shuffled_df)
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            
            if self.split == 'train':
                self.df = shuffled_df.iloc[:train_end].reset_index(drop=True)
            elif self.split == 'val':
                self.df = shuffled_df.iloc[train_end:val_end].reset_index(drop=True)
            else:  # test
                self.df = shuffled_df.iloc[val_end:].reset_index(drop=True)

        # Load concept config if path is provided in config dict
        concept_config_path = self.config.get("concept_config_path")
        self.concept_config = None
        self.concept_features_info = None

        if concept_config_path and os.path.exists(concept_config_path):
            import json
            with open(concept_config_path, 'r', encoding='utf-8') as f:
                self.concept_config = json.load(f)
            print(f"Loaded structured concept configuration file from: {concept_config_path}")
            
            self.concept_cols = list(self.concept_config.keys())
            self.concept_features_info = []
            self.concepts_flat = []
            total_dims = 0
            
            for name, info in self.concept_config.items():
                ctype = info.get("type", "numerical")
                if ctype == "categorical":
                    classes = info.get("classes", [])
                    num_feats = len(classes)
                    self.concept_features_info.append({
                        "name": name,
                        "type": "categorical",
                        "classes": classes,
                        "start_idx": total_dims,
                        "num_feats": num_feats
                    })
                    for cls_val in classes:
                        self.concepts_flat.append(f"{name}_{cls_val}")
                    total_dims += num_feats
                else:
                    self.concept_features_info.append({
                        "name": name,
                        "type": "numerical",
                        "min": float(info.get("min", 0.0)),
                        "max": float(info.get("max", 1.0)),
                        "start_idx": total_dims,
                        "num_feats": 1
                    })
                    self.concepts_flat.append(name)
                    total_dims += 1
            
            # Dynamically set num_concepts to the exact bottleneck size (categories + numerical)
            self.config["num_concepts"] = total_dims
            self.config["concepts"] = self.concept_cols
            self.config["concepts_flat"] = self.concepts_flat
        else:
            # Auto-detect MONET columns if CSV is present and concepts is not specified
            if not self.config.get("concepts") and not self.dummy_mode:
                detected_monet = [col for col in self.df.columns if col.startswith('MONET_')]
                if detected_monet:
                    self.config["concepts"] = detected_monet
                    self.config["num_concepts"] = len(detected_monet)
            
            # Fallback to defaults from get_default_config if still empty
            if not self.config.get("concepts"):
                self.config["concepts"] = self.get_default_config()["concepts"]
                self.config["num_concepts"] = len(self.config["concepts"])

            self.concept_cols = self.config["concepts"]
            self.config["concepts_flat"] = self.concept_cols

        self.target_col = self.config.get("target_col", "Malignancy")
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # In-memory caching
        self.cache_in_memory = cache_in_memory
        self._cache = None
        self._cache_populated = False
        self._try_populate_cache(max_cache_size_gb=max_cache_size_gb)

    def __len__(self) -> int:
        if self.dummy_mode:
            if self.split == 'train':
                return 70
            elif self.split == 'val':
                return 15
            else:
                return 15
        return len(self.df)

    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dummy_mode:
            img_tensor = torch.rand(3, 224, 224)
            img_tensor = self.transform(transforms.ToPILImage()(img_tensor)) if self.transform else img_tensor
            concepts = torch.rand(self.config.get("num_concepts", len(self.concept_cols)))
            target = torch.randint(0, 2, (1,)).float()
            return img_tensor, concepts, target
            
        row = self.df.iloc[idx]
        
        # Determine image path dynamically with fallbacks
        img_name = row.get('image_path')
        isic_id = row.get('isic_id')
        lesion_id = row.get('lesion_id')
        
        img_path = None
        if pd.notna(lesion_id) and pd.notna(isic_id):
            # Nested: image_dir/lesion_id/isic_id.jpg
            nested_path = os.path.join(self.image_dir, str(lesion_id), f"{isic_id}.jpg")
            if os.path.exists(nested_path):
                img_path = nested_path
                
        if img_path is None and pd.notna(isic_id):
            # Flat: image_dir/isic_id.jpg
            flat_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
            if os.path.exists(flat_path):
                img_path = flat_path
                
        if img_path is None and pd.notna(img_name):
            # Standard row path: image_dir/image_path
            custom_path = os.path.join(self.image_dir, str(img_name))
            if os.path.exists(custom_path):
                img_path = custom_path
                
        if img_path is None:
            # Fallback placeholder path
            img_path = os.path.join(self.image_dir, f"image_{idx}.jpg")
            
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, TypeError):
            if not hasattr(self, '_warned_missing'):
                self._warned_missing = set()
            if img_path not in self._warned_missing:
                print(f"Warning: Image file not found at {img_path}. Using a dummy black image.")
                self._warned_missing.add(img_path)
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        # Process concept variables dynamically
        if self.concept_features_info is not None:
            concept_vals = []
            for info in self.concept_features_info:
                name = info["name"]
                val = row.get(name)
                
                if info["type"] == "categorical":
                    classes = info["classes"]
                    one_hot = [0.0] * len(classes)
                    if pd.notna(val):
                        try:
                            if len(classes) > 0:
                                target_type = type(classes[0])
                                val_typed = target_type(val)
                                if val_typed in classes:
                                    val_idx = classes.index(val_typed)
                                    one_hot[val_idx] = 1.0
                        except (ValueError, TypeError):
                            pass
                    concept_vals.extend(one_hot)
                else:
                    min_val = info["min"]
                    max_val = info["max"]
                    if pd.isna(val):
                        scaled_val = 0.5
                    else:
                        try:
                            val_float = float(val)
                            denom = max_val - min_val
                            if denom == 0:
                                scaled_val = 0.0
                            else:
                                scaled_val = (val_float - min_val) / denom
                                scaled_val = max(0.0, min(1.0, scaled_val))
                        except (ValueError, TypeError):
                            scaled_val = 0.5
                    concept_vals.append(scaled_val)
            concept_tensor = torch.tensor(concept_vals, dtype=torch.float32)
        else:
            concept_vals = [float(row.get(col, 0.0)) for col in self.concept_cols]
            concept_tensor = torch.tensor(concept_vals, dtype=torch.float32)
        
        # Extract target value
        num_classes = self.config.get("num_classes", 1)
        if num_classes > 1:
            # Multi-class: return integer class index
            target_idx = int(row.get(self.target_col, 0))
            target_tensor = torch.tensor([target_idx], dtype=torch.long)
        else:
            # Binary: return float
            target_val = float(row.get(self.target_col, 0.0))
            target_tensor = torch.tensor([target_val], dtype=torch.float32)
        
        return image, concept_tensor, target_tensor
