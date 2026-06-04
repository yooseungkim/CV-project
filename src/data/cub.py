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
            "default_image_dir": 'data/CUB_200_2011/images',
            "filter_rare_concepts": False,
            "use_paper_preprocessing": False
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
        self.filter_rare_concepts = self.config.get("filter_rare_concepts", False)
        self.use_paper_preprocessing = self.config.get("use_paper_preprocessing", False)
        
        # Determine paths
        resolved_csv = csv_path or self.config.get("default_csv_path") or "data/CUB_200_2011/images.txt"
        self.image_dir = image_dir or self.config.get("default_image_dir") or "data/CUB_200_2011/images"
        data_root = os.path.dirname(resolved_csv)

        self.dummy_mode = not os.path.exists(resolved_csv)
        if self.dummy_mode:
            print(f"Warning: metadata not found at {resolved_csv}. Running in dummy mode ({self.split} split).")
            self.df = pd.DataFrame()
            self.concept_matrix = None
            self.bbox_df = None
        else:
            # 1. Load CUB-200-2011 base mapping files
            images_df = pd.read_csv(resolved_csv, sep=r'\s+', header=None, names=['image_id', 'image_path'])
            
            labels_file = os.path.join(data_root, 'image_class_labels.txt')
            labels_df = pd.read_csv(labels_file, sep=r'\s+', header=None, names=['image_id', 'class_id'])
            
            split_file = os.path.join(data_root, 'train_test_split.txt')
            split_df = pd.read_csv(split_file, sep=r'\s+', header=None, names=['image_id', 'is_train'])
            
            # Merge base frames
            merged = images_df.merge(labels_df, on='image_id').merge(split_df, on='image_id')
            
            # Load bounding boxes for regional bird cropping
            bbox_file = os.path.join(data_root, 'bounding_boxes.txt')
            if os.path.exists(bbox_file):
                print(f"Loading bounding boxes from: {bbox_file}")
                self.bbox_df = pd.read_csv(bbox_file, sep=r'\s+', header=None, names=['image_id', 'x', 'y', 'w', 'h'])
                self.bbox_df.set_index('image_id', inplace=True)
            else:
                print(f"Warning: Bounding boxes file not found at {bbox_file}. Running without bounding boxes.")
                self.bbox_df = None
                
            # 2. Slice deterministic train, val, test splits
            train_raw = merged[merged['is_train'] == 1].reset_index(drop=True)
            test_val_raw = merged[merged['is_train'] == 0].reset_index(drop=True)
            
            if self.use_paper_preprocessing:
                # Randomly split 20% of the official train set to make a validation set
                # Use random_state=42 for determinism
                shuffled_train = train_raw.sample(frac=1.0, random_state=42).reset_index(drop=True)
                n_train = len(shuffled_train)
                val_size = int(n_train * 0.2)
                
                if self.split == 'train':
                    self.df = shuffled_train.iloc[val_size:].reset_index(drop=True)
                elif self.split == 'val':
                    self.df = shuffled_train.iloc[:val_size].reset_index(drop=True)
                else:  # test
                    self.df = test_val_raw
            else:
                # Deterministic split of test set into 50% val and 50% test (legacy/default)
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
                self.concept_matrix = attr_df['is_present'].values.reshape(11788, 312).copy()
            else:
                print(f"Warning: attributes file not found at {attr_file}. Using dummy concepts.")
                import numpy as np
                self.concept_matrix = np.zeros((11788, 312))

            # Filter rare concepts (< 1% global frequency) or run CBM Paper Preprocessing
            if self.use_paper_preprocessing:
                import numpy as np
                # Class-level majority voting
                # self.concept_matrix has shape [11788, 312]
                class_ids = labels_df['class_id'].values - 1  # 0-indexed
                
                # Step 1: Compute class-level concepts (majority voting: mean >= 0.5)
                class_concepts = np.zeros((200, 312))
                for class_id in range(200):
                    class_mask = (class_ids == class_id)
                    if class_mask.sum() > 0:
                        class_concepts[class_id] = (self.concept_matrix[class_mask].mean(axis=0) >= 0.5).astype(float)
                
                # Step 2: Overwrite instance-level concepts with majority-voted class-level concepts
                for class_id in range(200):
                    class_mask = (class_ids == class_id)
                    if class_mask.sum() > 0:
                        self.concept_matrix[class_mask] = class_concepts[class_id]
                
                # Step 3: Keep concepts present in at least 10 classes after majority voting
                valid_concepts_mask = (class_concepts.sum(axis=0) >= 10)
                self.valid_indices = np.where(valid_concepts_mask)[0]
                print(f"CBM Paper Preprocessing: keeping {len(self.valid_indices)} out of 312 concepts (present in >= 10 classes after majority voting).")
                self.concept_matrix = self.concept_matrix[:, self.valid_indices]
            elif self.filter_rare_concepts:
                import numpy as np
                freqs = np.mean(self.concept_matrix, axis=0)
                self.valid_indices = np.where(freqs >= 0.01)[0]
                print(f"Filtering concepts with frequency < 1%: keeping {len(self.valid_indices)} out of 312 concepts.")
                self.concept_matrix = self.concept_matrix[:, self.valid_indices]
            else:
                self.valid_indices = None

        # 4. Load concept configuration for dynamic formatting
        concept_config_path = self.config.get("concept_config_path")
        self.concept_config = None
        self.concept_features_info = None

        if concept_config_path and os.path.exists(concept_config_path):
            import json
            with open(concept_config_path, 'r', encoding='utf-8') as f:
                self.concept_config = json.load(f)
            print(f"Loaded structured concept configuration file from: {concept_config_path}")
            
            # If the config file is already filtered, we bypass filtering on config loading
            # but we still keep self.concept_matrix sliced using self.valid_indices.
            is_already_filtered = "filtered" in os.path.basename(concept_config_path)
            
            self.concept_features_info = []
            self.concepts_flat = []
            new_total_dims = 0
            original_idx = 0
            
            # Reconstruct filtered concept configuration dictionary
            filtered_config = {}
            
            for name, info in self.concept_config.items():
                ctype = info.get("type", "numerical")
                if ctype == "categorical":
                    classes = info.get("classes", [])
                    
                    # Find which classes are valid
                    valid_classes_in_group = []
                    for cls_val in classes:
                        if is_already_filtered or self.valid_indices is None or original_idx in self.valid_indices:
                            valid_classes_in_group.append(cls_val)
                        original_idx += 1
                        
                    if len(valid_classes_in_group) > 0:
                        num_feats = len(valid_classes_in_group)
                        self.concept_features_info.append({
                            "name": name,
                            "type": "categorical",
                            "classes": valid_classes_in_group,
                            "start_idx": new_total_dims,
                            "num_feats": num_feats
                        })
                        for cls_val in valid_classes_in_group:
                            self.concepts_flat.append(f"{name}_{cls_val}")
                        new_total_dims += num_feats
                        
                        filtered_config[name] = {
                            "type": "categorical",
                            "classes": valid_classes_in_group
                        }
                else:
                    # Numerical group
                    is_valid = (is_already_filtered or self.valid_indices is None or original_idx in self.valid_indices)
                    if is_valid:
                        self.concept_features_info.append({
                            "name": name,
                            "type": "numerical",
                            "min": float(info.get("min", 0.0)),
                            "max": float(info.get("max", 1.0)),
                            "start_idx": new_total_dims,
                            "num_feats": 1
                        })
                        self.concepts_flat.append(name)
                        new_total_dims += 1
                        
                        filtered_config[name] = info.copy()
                    original_idx += 1
            
            self.concept_cols = [info["name"] for info in self.concept_features_info]
            self.config["num_concepts"] = new_total_dims
            self.config["concepts"] = self.concept_cols
            self.config["concepts_flat"] = self.concepts_flat

            # Save the filtered concept config to disk for Gradio and eval compatibility
            if (self.filter_rare_concepts or self.use_paper_preprocessing) and not self.dummy_mode and not is_already_filtered:
                filtered_path = concept_config_path.replace(".json", "_filtered.json")
                try:
                    with open(filtered_path, 'w', encoding='utf-8') as f:
                        json.dump(filtered_config, f, indent=2, ensure_ascii=False)
                    print(f"[Config] Saved filtered concept configuration to: {filtered_path}")
                except Exception as e:
                    print(f"Warning: Failed to save filtered concept configuration to {filtered_path}: {e}")
        else:
            self.config["num_concepts"] = self.config.get("num_concepts", 312)
            self.concept_cols = [f"Attribute_{i}" for i in range(1, 313)]
            self.config["concepts"] = self.concept_cols
            self.config["concepts_flat"] = self.concept_cols

        # Target classification parameters
        self.target_col = self.config.get("target_col", "class_id")
        self.config["num_classes"] = 200

        if transform is not None:
            self.transform = transform
        else:
            if self.split == 'train':
                # Advanced dynamic data augmentation for training to prevent overfitting
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
        image_id = int(row['image_id'])
        
        # Load image
        img_name = row['image_path']
        img_path = os.path.join(self.image_dir, str(img_name))
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
        # Crop image using bounding box if available
        if getattr(self, 'bbox_df', None) is not None and image_id in self.bbox_df.index:
            bbox = self.bbox_df.loc[image_id]
            img_w, img_h = image.size
            x = max(0.0, float(bbox['x']))
            y = max(0.0, float(bbox['y']))
            w = max(1.0, float(bbox['w']))
            h = max(1.0, float(bbox['h']))
            x1 = min(img_w - 1.0, x)
            y1 = min(img_h - 1.0, y)
            x2 = min(img_w, x1 + w)
            y2 = min(img_h, y1 + h)
            if x2 > x1 and y2 > y1:
                image = image.crop((x1, y1, x2, y2))
                
        # Do NOT apply transform here; return raw PIL Image so we can cache it
        # and apply dynamic augmentations on the fly during __getitem__
        
        # Extract concepts from matrix
        concept_vals = self.concept_matrix[image_idx]
        concept_tensor = torch.tensor(concept_vals, dtype=torch.float32)
        
        # Extract class_id (1-indexed -> convert to 0-indexed)
        target_idx = int(row[self.target_col]) - 1
        target_tensor = torch.tensor([target_idx], dtype=torch.long)
        
        return image, concept_tensor, target_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cache_populated and self._cache is not None:
            image, concept_tensor, target_tensor = self._cache[idx]
        else:
            image, concept_tensor, target_tensor = self._load_sample(idx)
            
        # Apply the transform dynamically on the fly
        if self.transform:
            if isinstance(image, Image.Image):
                image = self.transform(image)
                
        return image, concept_tensor, target_tensor
