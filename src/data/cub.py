import os
import pandas as pd
from typing import Tuple, List, Optional
import torch
from torchvision import transforms
from PIL import Image
from src.data.base_dataset import BaseDataset

class CUB2011Dataset(BaseDataset):
    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "num_concepts": 312,
            "num_classes": 200,
            "concepts": [],
            "target_col": 'class_id',
            "default_csv_path": 'data/CUB_200_2011/images.txt',
            "default_image_dir": 'data/CUB_200_2011/images'
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
            csv_path: Path to the metadata images file. If None, loaded from default config.
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
        resolved_csv = csv_path or self.config.get("default_csv_path") or "data/CUB_200_2011/images.txt"
        self.image_dir = image_dir or self.config.get("default_image_dir") or "data/CUB_200_2011/images"
        data_root = os.path.dirname(resolved_csv)

        self.dummy_mode = not os.path.exists(resolved_csv)
        if self.dummy_mode:
            print(f"Warning: metadata not found at {resolved_csv}. Running in dummy mode ({self.split} split).")
            self.df = pd.DataFrame()
            self.concept_matrix = None
        else:
            # 1. Load CUB-200-2011 base mapping files
            images_df = pd.read_csv(resolved_csv, sep=r'\s+', header=None, names=['image_id', 'image_path'])
            
            labels_file = os.path.join(data_root, 'image_class_labels.txt')
            labels_df = pd.read_csv(labels_file, sep=r'\s+', header=None, names=['image_id', 'class_id'])
            
            split_file = os.path.join(data_root, 'train_test_split.txt')
            split_df = pd.read_csv(split_file, sep=r'\s+', header=None, names=['image_id', 'is_train'])
            
            # Merge base frames
            merged = images_df.merge(labels_df, on='image_id').merge(split_df, on='image_id')
            
            # 2. Slice deterministic train, val, test splits
            train_raw = merged[merged['is_train'] == 1].reset_index(drop=True)
            test_val_raw = merged[merged['is_train'] == 0].reset_index(drop=True)
            
            # Deterministic split of test set into 50% val and 50% test
            shuffled_test_val = test_val_raw.sample(frac=1.0, random_state=42).reset_index(drop=True)
            n_test_val = len(shuffled_test_val)
            val_end = n_test_val // 2
            
            if self.split == 'train':
                self.df = train_raw
            elif self.split == 'val':
                self.df = shuffled_test_val.iloc[:val_end].reset_index(drop=True)
            else:  # test
                self.df = shuffled_test_val.iloc[val_end:].reset_index(drop=True)
                
            self.df['image_idx'] = self.df['image_id'] - 1
            print(f"Successfully loaded CUB split [{self.split}] (size: {len(self.df)})")

            # 3. Load Attribute concept presence annotations [11788, 312]
            attr_file = os.path.join(data_root, 'attributes', 'image_attribute_labels.txt')
            if os.path.exists(attr_file):
                print(f"Loading image-level attribute annotations from: {attr_file}")
                attr_df = pd.read_csv(
                    attr_file, sep=r'\s+', header=None, usecols=[0, 1, 2],
                    names=['image_id', 'attribute_id', 'is_present']
                )
                # Reshape attributes perfectly into concept presence matrix
                self.concept_matrix = attr_df['is_present'].values.reshape(11788, 312)
            else:
                print(f"Warning: attributes file not found at {attr_file}. Using dummy concepts.")
                import numpy as np
                self.concept_matrix = np.zeros((11788, 312))

        # 4. Load concept configuration for dynamic formatting
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
            
            self.config["num_concepts"] = total_dims
            self.config["concepts"] = self.concept_cols
            self.config["concepts_flat"] = self.concepts_flat
        else:
            self.config["num_concepts"] = self.config.get("num_concepts", 312)
            self.concept_cols = [f"Attribute_{i}" for i in range(1, 313)]
            self.config["concepts"] = self.concept_cols
            self.config["concepts_flat"] = self.concept_cols

        # Target classification parameters
        self.target_col = self.config.get("target_col", "class_id")
        self.config["num_classes"] = 200

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Caching
        self.cache_in_memory = cache_in_memory
        self._cache = None
        self._cache_populated = False
        self._try_populate_cache(max_cache_size_gb=max_cache_size_gb)

    def __len__(self) -> int:
        if self.dummy_mode:
            if self.split == 'train':
                return 100
            elif self.split == 'val':
                return 20
            else:
                return 20
        return len(self.df)

    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dummy_mode:
            img_tensor = torch.rand(3, 224, 224)
            img_tensor = self.transform(transforms.ToPILImage()(img_tensor)) if self.transform else img_tensor
            concepts = torch.rand(self.config.get("num_concepts", 312))
            target = torch.randint(0, 200, (1,)).long()
            return img_tensor, concepts, target
            
        row = self.df.iloc[idx]
        image_idx = int(row['image_idx'])
        
        # Load image
        img_name = row['image_path']
        img_path = os.path.join(self.image_dir, str(img_name))
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        # Extract concepts from matrix
        concept_vals = self.concept_matrix[image_idx]
        concept_tensor = torch.tensor(concept_vals, dtype=torch.float32)
        
        # Extract class_id (1-indexed -> convert to 0-indexed)
        target_idx = int(row[self.target_col]) - 1
        target_tensor = torch.tensor([target_idx], dtype=torch.long)
        
        return image, concept_tensor, target_tensor
