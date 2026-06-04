import os
import pandas as pd
from typing import Tuple, List, Optional
import torch
from torchvision import transforms
from PIL import Image
from src.data.base_dataset import BaseDataset

class Derm7PtDataset(BaseDataset):
    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "num_concepts": 7,
            "num_classes": 3,
            "concepts": [
                "pigment_network", "streaks", "pigmentation", "regression_structures", 
                "dots_and_globules", "blue_whitish_veil", "vascular_structures"
            ],
            "target_col": 'diagnosis',
            "default_csv_path": 'data/derm7pt/meta/meta.csv',
            "default_image_dir": 'data/derm7pt/images',
            "use_multimodal": False
        }

    def __init__(
        self,
        csv_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        split: str = 'train',
        image_type: str = 'derm',  # 'derm' or 'clinic'
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
            image_type: Type of skin image to load ('derm' or 'clinic').
            config: Configuration dictionary for the dataset.
            transform: torchvision transforms to apply to the images.
        """
        super().__init__()
        self.split = split.lower()
        if self.split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of 'train', 'val', 'test'")
        
        self.image_type = image_type.lower()
        if self.image_type not in ['derm', 'clinic']:
            raise ValueError("image_type must be 'derm' or 'clinic'")

        # Initialize config
        self.config = config or self.get_default_config()
        self.use_multimodal = self.config.get("use_multimodal", False)
        
        # Determine paths
        resolved_csv = csv_path or self.config.get("default_csv_path") or "data/derm7pt/meta/meta.csv"
        self.image_dir = image_dir or self.config.get("default_image_dir") or "data/derm7pt/images"

        self.dummy_mode = not os.path.exists(resolved_csv)
        if self.dummy_mode:
            print(f"Warning: CSV file not found at {resolved_csv}. Running in dummy mode ({self.split} split).")
            self.df = pd.DataFrame()
        else:
            full_df = pd.read_csv(resolved_csv)
            
            # Apply diagnosis mapping to consolidate 20 classes into 5 classes
            self.diagnosis_mapping = {
                'clark nevus': 'Nevus',
                'reed or spitz nevus': 'Nevus',
                'dermal nevus': 'Nevus',
                'blue nevus': 'Nevus',
                'congenital nevus': 'Nevus',
                'combined nevus': 'Nevus',
                'recurrent nevus': 'Nevus',
                'melanoma (less than 0.76 mm)': 'Melanoma',
                'melanoma (in situ)': 'Melanoma',
                'melanoma (0.76 to 1.5 mm)': 'Melanoma',
                'melanoma (more than 1.5 mm)': 'Melanoma',
                'melanoma metastasis': 'Melanoma',
                'melanoma': 'Melanoma',
                'basal cell carcinoma': 'BCC',
                'seborrheic keratosis': 'SK',
                'vascular lesion': 'MISC',
                'lentigo': 'MISC',
                'dermatofibroma': 'MISC',
                'melanosis': 'MISC',
                'miscellaneous': 'MISC'
            }
            target_col = self.config.get("target_col", "diagnosis")
            if target_col in full_df.columns:
                full_df[target_col] = full_df[target_col].map(self.diagnosis_mapping)

            
            # Load suggested indexes for the given split
            meta_dir = os.path.dirname(resolved_csv)
            if self.split == 'train':
                idx_file = os.path.join(meta_dir, 'train_indexes.csv')
            elif self.split == 'val':
                idx_file = os.path.join(meta_dir, 'valid_indexes.csv')
            else:
                idx_file = os.path.join(meta_dir, 'test_indexes.csv')
                
            if os.path.exists(idx_file):
                idx_df = pd.read_csv(idx_file)
                self.df = full_df.iloc[idx_df['indexes']].reset_index(drop=True)
                print(f"Successfully loaded {self.split} split indexes for derm7pt from: {idx_file} (size: {len(self.df)})")
            else:
                print(f"Warning: Split index file {idx_file} not found. Splitting full CSV deterministically (70/15/15).")
                shuffled_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                n = len(shuffled_df)
                train_end = int(n * 0.7)
                val_end = int(n * 0.85)
                
                if self.split == 'train':
                    self.df = shuffled_df.iloc[:train_end].reset_index(drop=True)
                elif self.split == 'val':
                    self.df = shuffled_df.iloc[train_end:val_end].reset_index(drop=True)
                else:
                    self.df = shuffled_df.iloc[val_end:].reset_index(drop=True)

            # Filter out any rows where clinic or derm filenames contain 'bis'
            if 'clinic' in self.df.columns and 'derm' in self.df.columns:
                before_len = len(self.df)
                self.df = self.df[~(self.df['clinic'].str.contains('bis', case=False, na=False) | self.df['derm'].str.contains('bis', case=False, na=False))].reset_index(drop=True)
                print(f"Filtered out 'bis' images for {self.split} split. Rows before: {before_len}, Rows after: {len(self.df)}")


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
            if not self.config.get("concepts"):
                self.config["concepts"] = self.get_default_config()["concepts"]
            self.config["num_concepts"] = len(self.config["concepts"])
            self.concept_cols = self.config["concepts"]
            self.config["concepts_flat"] = self.concept_cols

        self.target_col = self.config.get("target_col", "diagnosis")
        
        # Build target class mapping dynamically from full CSV (to ensure uniform mapping across splits)
        if not self.dummy_mode:
            unique_targets = sorted(full_df[self.target_col].dropna().unique())
            self.target_to_idx = {name: i for i, name in enumerate(unique_targets)}
            self.config["num_classes"] = len(unique_targets)
        else:
            self.target_to_idx = {}
            self.config["num_classes"] = self.config.get("num_classes", 1)

        if transform is not None:
            self.transform = transform
        else:
            if self.split == 'train':
                # Advanced spatial data augmentations for training to prevent overfitting
                # Strictly NO color jittering, since lesion color is a clinical diagnostic feature.
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Static, deterministic transforms for validation/testing
                self.transform = transforms.Compose([
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
            if getattr(self, 'use_multimodal', False):
                img_t1 = torch.rand(3, 224, 224)
                img_t2 = torch.rand(3, 224, 224)
                if self.transform:
                    img_t1 = self.transform(transforms.ToPILImage()(img_t1))
                    img_t2 = self.transform(transforms.ToPILImage()(img_t2))
                image = torch.stack([img_t1, img_t2], dim=0)
            else:
                img_tensor = torch.rand(3, 224, 224)
                image = self.transform(transforms.ToPILImage()(img_tensor)) if self.transform else img_tensor
                
            concepts = torch.rand(self.config.get("num_concepts", len(self.concept_cols)))
            num_classes = self.config.get("num_classes", 1)
            if num_classes == 1:
                target = torch.randint(0, 2, (1,)).float()
            else:
                target = torch.randint(0, num_classes, (1,)).float()
            return image, concepts, target
            
        row = self.df.iloc[idx]
        
        if getattr(self, 'use_multimodal', False):
            # Load clinical image
            clinic_name = row.get('clinic')
            clinic_path = os.path.join(self.image_dir, str(clinic_name)) if pd.notna(clinic_name) else ""
            try:
                img_clinic = Image.open(clinic_path).convert('RGB') if clinic_path and os.path.exists(clinic_path) else Image.new('RGB', (224, 224), color=(0, 0, 0))
            except Exception:
                img_clinic = Image.new('RGB', (224, 224), color=(0, 0, 0))
                
            # Load dermoscopic image
            derm_name = row.get('derm')
            derm_path = os.path.join(self.image_dir, str(derm_name)) if pd.notna(derm_name) else ""
            try:
                img_derm = Image.open(derm_path).convert('RGB') if derm_path and os.path.exists(derm_path) else Image.new('RGB', (224, 224), color=(0, 0, 0))
            except Exception:
                img_derm = Image.new('RGB', (224, 224), color=(0, 0, 0))
                
            if self.transform:
                img_clinic = self.transform(img_clinic)
                img_derm = self.transform(img_derm)
            else:
                transform_to_tensor = transforms.ToTensor()
                img_clinic = transform_to_tensor(img_clinic)
                img_derm = transform_to_tensor(img_derm)
                
            image = torch.stack([img_clinic, img_derm], dim=0) # shape: [2, 3, H, W]
        else:
            # Determine image path
            img_name = row.get(self.image_type, row.get('derm'))
            img_path = os.path.join(self.image_dir, str(img_name))
            
            try:
                image = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
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
        target_val_raw = row.get(self.target_col)
        num_classes = self.config.get("num_classes", 1)
        if num_classes == 1:
            try:
                target_val = float(target_val_raw)
            except (ValueError, TypeError):
                target_val = 0.0
            target_tensor = torch.tensor([target_val], dtype=torch.float32)
        else:
            target_idx = self.target_to_idx.get(target_val_raw, 0)
            target_tensor = torch.tensor([target_idx], dtype=torch.float32)
            
        return image, concept_tensor, target_tensor
