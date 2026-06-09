import os
import pandas as pd
from typing import Tuple, List, Optional
import torch
from torchvision import transforms
from PIL import Image
from src.data.base_dataset import BaseDataset

class CUB2011Dataset(BaseDataset):
    # Canonical CUB attribute indices used by the original Concept Bottleneck
    # paper/codebase after class-level denoising and the >=10-class filter.
    # These are zero-based indices from CUB attributes.txt.
    PAPER_ATTRIBUTE_INDICES = [
        1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38,
        40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72,
        75, 80, 84, 90, 91, 93, 99, 101, 106, 110, 111, 116, 117,
        119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157,
        158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193,
        194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220,
        221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253,
        254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294,
        298, 299, 304, 305, 308, 309, 310, 311
    ]

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
            "use_paper_preprocessing": False,
            "allow_legacy_filtered_concept_config": False
        }

    @staticmethod
    def is_filtered_concept_config_path(concept_config_path: Optional[str]) -> bool:
        if not concept_config_path:
            return False
        root, _ = os.path.splitext(os.path.basename(concept_config_path))
        return root.endswith(("_filtered", "_paper_filtered", "_rare_filtered"))

    @staticmethod
    def infer_unfiltered_concept_config_path(concept_config_path: Optional[str]) -> Optional[str]:
        if not concept_config_path:
            return None

        dirname = os.path.dirname(concept_config_path)
        root, ext = os.path.splitext(os.path.basename(concept_config_path))
        for suffix in ("_paper_filtered", "_rare_filtered", "_filtered"):
            if root.endswith(suffix):
                return os.path.join(dirname, root[:-len(suffix)] + ext)
        return None

    @staticmethod
    def flatten_concept_config_keys(concept_config: dict) -> List[str]:
        flat_keys = []
        for name, info in concept_config.items():
            if info.get("type", "numerical") == "categorical":
                flat_keys.extend(f"{name}::{class_name}" for class_name in info.get("classes", []))
            else:
                flat_keys.append(name)
        return flat_keys

    def _load_legacy_valid_indices_from_filtered_config(self, concept_config_path: str):
        import json
        import numpy as np

        unfiltered_path = self.infer_unfiltered_concept_config_path(concept_config_path)
        if not unfiltered_path or not os.path.exists(unfiltered_path):
            raise FileNotFoundError(
                "Legacy filtered CUB concept config requires the matching unfiltered "
                f"concept_config.json, but it was not found for: {concept_config_path}"
            )

        with open(concept_config_path, "r", encoding="utf-8") as f:
            filtered_config = json.load(f)
        with open(unfiltered_path, "r", encoding="utf-8") as f:
            unfiltered_config = json.load(f)

        unfiltered_flat = self.flatten_concept_config_keys(unfiltered_config)
        filtered_flat = self.flatten_concept_config_keys(filtered_config)
        index_by_key = {}
        for idx, key in enumerate(unfiltered_flat):
            if key in index_by_key:
                raise ValueError(f"Duplicate CUB concept key in unfiltered config: {key}")
            index_by_key[key] = idx

        missing = [key for key in filtered_flat if key not in index_by_key]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Filtered CUB concept config {concept_config_path} contains concepts "
                f"that are not present in {unfiltered_path}: {preview}"
            )

        self._legacy_filtered_concept_config = filtered_config
        return np.asarray([index_by_key[key] for key in filtered_flat], dtype=int)

    def _fit_class_level_concepts(self, concept_preprocessing_df, labels_df, certainty_matrix=None):
        import numpy as np

        train_image_indices = concept_preprocessing_df['image_id'].to_numpy(dtype=int) - 1
        train_class_ids = concept_preprocessing_df['class_id'].to_numpy(dtype=int) - 1
        all_class_ids = labels_df.sort_values('image_id')['class_id'].to_numpy(dtype=int) - 1
        self.concept_preprocessing_train_image_ids = concept_preprocessing_df['image_id'].astype(int).tolist()

        if certainty_matrix is not None:
            class_attr_count = np.zeros((200, 312, 2), dtype=int)
            seen_class_ids = set()
            for image_idx, class_id in zip(train_image_indices, train_class_ids):
                seen_class_ids.add(int(class_id))
                certainties = certainty_matrix[image_idx]
                for attr_idx, attr_label in enumerate(self.concept_matrix[image_idx]):
                    attr_label = int(attr_label)
                    if attr_label == 0 and int(certainties[attr_idx]) == 1:
                        continue
                    class_attr_count[class_id, attr_idx, attr_label] += 1

            class_attr_min_label = np.argmin(class_attr_count, axis=2)
            class_concepts = np.argmax(class_attr_count, axis=2)
            equal_count = np.where(class_attr_min_label == class_concepts)
            class_concepts[equal_count] = 1

            missing_train_classes = [
                class_id + 1
                for class_id in range(200)
                if class_id not in seen_class_ids
            ]
            if missing_train_classes:
                print(
                    "Warning: No training samples found for CUB classes "
                    f"{missing_train_classes}. Their class-level concepts remain all-zero."
                )
                class_concepts[np.asarray(missing_train_classes, dtype=int) - 1] = 0
        else:
            class_concepts = np.zeros((200, 312))
            missing_train_classes = []
            for class_id in range(200):
                class_image_indices = train_image_indices[train_class_ids == class_id]
                if class_image_indices.size > 0:
                    class_concepts[class_id] = (
                        self.concept_matrix[class_image_indices].mean(axis=0) >= 0.5
                    ).astype(float)
                else:
                    missing_train_classes.append(class_id + 1)

            if missing_train_classes:
                print(
                    "Warning: No training samples found for CUB classes "
                    f"{missing_train_classes}. Their class-level concepts remain all-zero."
                )

        for class_id in range(200):
            class_mask = (all_class_ids == class_id)
            if class_mask.sum() > 0:
                self.concept_matrix[class_mask] = class_concepts[class_id]

        self.class_concepts = class_concepts.copy()
        return class_concepts

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
        self.allow_legacy_filtered_concept_config = self.config.get("allow_legacy_filtered_concept_config", False)
        self.concept_metadata = self.config.get("concept_metadata") or self.config.get("precomputed_concept_metadata") or {}
        self.valid_indices = None
        self.class_concepts = None
        self.effective_concept_config = None
        self.concept_preprocessing_train_image_ids = []
        self._legacy_filtered_concept_config = None
        
        # Determine paths
        resolved_csv = csv_path or self.config.get("default_csv_path") or "data/CUB_200_2011/images.txt"
        self.image_dir = image_dir or self.config.get("default_image_dir") or "data/CUB_200_2011/images"
        data_root = os.path.dirname(resolved_csv)

        self.dummy_mode = not os.path.exists(resolved_csv)
        if self.dummy_mode:
            print(f"Warning: metadata not found at {resolved_csv}. Running in dummy mode ({self.split} split).")
            self.df = pd.DataFrame()
            self.concept_matrix = None
            self.concept_certainty_matrix = None
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
                train_split_raw = shuffled_train.iloc[val_size:].reset_index(drop=True)
                val_split_raw = shuffled_train.iloc[:val_size].reset_index(drop=True)
                
                if self.split == 'train':
                    self.df = train_split_raw
                elif self.split == 'val':
                    self.df = val_split_raw
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
                
                train_split_raw = train_raw

            concept_preprocessing_df = train_split_raw.reset_index(drop=True)
            self.df['image_idx'] = self.df['image_id'] - 1
            print(f"Successfully loaded CUB split [{self.split}] (size: {len(self.df)})")

            # 3. Load Attribute concept presence annotations [11788, 312]
            attr_file = os.path.join(data_root, 'attributes', 'image_attribute_labels.txt')
            if os.path.exists(attr_file):
                print(f"Loading image-level attribute annotations from: {attr_file}")
                attr_df = pd.read_csv(
                    attr_file, sep=r'\s+', header=None, usecols=[0, 1, 2, 3],
                    names=['image_id', 'attribute_id', 'is_present', 'certainty']
                )
                # Reshape attributes perfectly into concept presence matrix
                self.concept_matrix = attr_df['is_present'].values.reshape(11788, 312).copy()
                self.concept_certainty_matrix = attr_df['certainty'].values.reshape(11788, 312).copy()
            else:
                print(f"Warning: attributes file not found at {attr_file}. Using dummy concepts.")
                import numpy as np
                self.concept_matrix = np.zeros((11788, 312))
                self.concept_certainty_matrix = np.zeros((11788, 312))

            # Filter rare concepts (< 1% global frequency) or run CBM Paper Preprocessing
            precomputed_valid_indices = self.concept_metadata.get("valid_indices")
            legacy_valid_indices = None
            concept_config_path = self.config.get("concept_config_path")
            if (
                precomputed_valid_indices is None
                and self.allow_legacy_filtered_concept_config
                and concept_config_path
                and self.is_filtered_concept_config_path(concept_config_path)
            ):
                legacy_valid_indices = self._load_legacy_valid_indices_from_filtered_config(concept_config_path)

            if precomputed_valid_indices is not None or legacy_valid_indices is not None:
                import numpy as np
                if precomputed_valid_indices is not None:
                    self.valid_indices = np.asarray(precomputed_valid_indices, dtype=int)
                else:
                    self.valid_indices = np.asarray(legacy_valid_indices, dtype=int)
                precomputed_class_concepts = self.concept_metadata.get("class_concepts")
                if precomputed_class_concepts is not None:
                    self.class_concepts = np.asarray(precomputed_class_concepts, dtype=float)
                    if self.class_concepts.shape != (200, 312):
                        raise ValueError(
                            "CUB concept_metadata.class_concepts must have shape [200, 312], "
                            f"got {self.class_concepts.shape}."
                        )
                    all_class_ids = labels_df.sort_values('image_id')['class_id'].to_numpy(dtype=int) - 1
                    for class_id in range(200):
                        class_mask = (all_class_ids == class_id)
                        if class_mask.sum() > 0:
                            self.concept_matrix[class_mask] = self.class_concepts[class_id]
                elif self.use_paper_preprocessing:
                    if legacy_valid_indices is None:
                        raise ValueError(
                            "CUB paper preprocessing metadata requires class_concepts to avoid "
                            "recomputing class-level concepts during evaluation."
                        )
                    legacy_preprocessing_df = merged[['image_id', 'class_id']].reset_index(drop=True)
                    self._fit_class_level_concepts(legacy_preprocessing_df, labels_df)

                if legacy_valid_indices is not None:
                    print(
                        "Warning: Loading metadata-free legacy CUB concept mask from "
                        f"{concept_config_path}; keeping {len(self.valid_indices)} concepts. "
                        "Reproducing historical all-image class-level concepts for old "
                        "checkpoint evaluation only."
                    )
                else:
                    print(
                        "CUB Concept Metadata: using checkpoint-selected "
                        f"{len(self.valid_indices)} concepts; skipping concept filter recomputation."
                    )
                self.concept_matrix = self.concept_matrix[:, self.valid_indices]
            elif self.use_paper_preprocessing:
                import numpy as np
                self._fit_class_level_concepts(
                    concept_preprocessing_df,
                    labels_df,
                    certainty_matrix=self.concept_certainty_matrix
                )
                
                # Match the original ConceptBottleneck CUB pipeline exactly:
                # use the canonical 112 paper attributes after class-level
                # denoising, rather than recomputing a split-dependent mask.
                self.valid_indices = np.asarray(self.PAPER_ATTRIBUTE_INDICES, dtype=int)
                print(
                    "CBM Paper Preprocessing: fitted on "
                    f"{len(concept_preprocessing_df)} train images; keeping {len(self.valid_indices)} "
                    "canonical paper concepts out of 312."
                )
                self.concept_matrix = self.concept_matrix[:, self.valid_indices]
            elif self.filter_rare_concepts:
                import numpy as np
                train_image_indices = concept_preprocessing_df['image_id'].to_numpy(dtype=int) - 1
                self.concept_preprocessing_train_image_ids = concept_preprocessing_df['image_id'].astype(int).tolist()
                freqs = np.mean(self.concept_matrix[train_image_indices], axis=0)
                self.valid_indices = np.where(freqs >= 0.01)[0]
                print(
                    "Filtering concepts with frequency < 1% using "
                    f"{len(concept_preprocessing_df)} train images: keeping {len(self.valid_indices)} out of 312 concepts."
                )
                self.concept_matrix = self.concept_matrix[:, self.valid_indices]
            else:
                self.valid_indices = None

        # 4. Load concept configuration for dynamic formatting
        concept_config_path = self.config.get("concept_config_path")
        self.concept_config = None
        self.concept_features_info = None
        precomputed_concept_config = self.concept_metadata.get("concept_config")
        if (
            concept_config_path
            and self.is_filtered_concept_config_path(concept_config_path)
            and not precomputed_concept_config
            and not self.allow_legacy_filtered_concept_config
        ):
            raise ValueError(
                f"Refusing to load filtered CUB concept config file: {concept_config_path}. "
                "Filtered concept configs can be stale relative to a checkpoint. Use the "
                "unfiltered concept_config.json plus checkpoint concept_metadata instead."
            )

        if precomputed_concept_config:
            self.concept_config = precomputed_concept_config
            print("Loaded structured concept configuration from checkpoint metadata")
            is_already_filtered = True
        elif self._legacy_filtered_concept_config is not None:
            self.concept_config = self._legacy_filtered_concept_config
            print(f"Loaded legacy filtered concept configuration file from: {concept_config_path}")
            is_already_filtered = True
        elif concept_config_path and os.path.exists(concept_config_path):
            import json
            with open(concept_config_path, 'r', encoding='utf-8') as f:
                self.concept_config = json.load(f)
            print(f"Loaded structured concept configuration file from: {concept_config_path}")
            
            # If the config file is already filtered, we bypass filtering on config loading
            # but we still keep self.concept_matrix sliced using self.valid_indices.
            is_already_filtered = "filtered" in os.path.basename(concept_config_path)
        else:
            is_already_filtered = False

        if self.concept_config is not None:
            if is_already_filtered and self.valid_indices is not None:
                loaded_flat_dims = 0
                for info in self.concept_config.values():
                    if info.get("type", "numerical") == "categorical":
                        loaded_flat_dims += len(info.get("classes", []))
                    else:
                        loaded_flat_dims += 1
                if loaded_flat_dims != len(self.valid_indices):
                    raise ValueError(
                        f"Filtered CUB concept config at {concept_config_path} has "
                        f"{loaded_flat_dims} flattened concepts, but preprocessing selected "
                        f"{len(self.valid_indices)}. Pass the unfiltered concept_config.json or regenerate "
                        "the filtered config with the current preprocessing code."
                    )
            
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
            self.effective_concept_config = filtered_config

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

    def get_concept_metadata(self) -> dict:
        valid_indices = None
        if self.valid_indices is not None:
            valid_indices = [int(idx) for idx in self.valid_indices.tolist()]

        class_concepts = None
        if self.class_concepts is not None:
            class_concepts = self.class_concepts.astype(int).tolist()

        if self.use_paper_preprocessing:
            preprocessing_mode = "paper"
        elif self.filter_rare_concepts:
            preprocessing_mode = "rare"
        else:
            preprocessing_mode = "none"

        return {
            "version": 1,
            "dataset": "cub",
            "preprocessing_mode": preprocessing_mode,
            "filter_rare_concepts": bool(self.filter_rare_concepts),
            "use_paper_preprocessing": bool(self.use_paper_preprocessing),
            "concept_config_path": self.config.get("concept_config_path"),
            "num_concepts": int(self.config.get("num_concepts", 0)),
            "valid_indices": valid_indices,
            "class_concepts": class_concepts,
            "concept_config": self.effective_concept_config,
            "concepts": list(self.config.get("concepts", [])),
            "concepts_flat": list(self.config.get("concepts_flat", [])),
            "concept_features_info": self.concept_features_info,
            "preprocessing_train_image_ids": list(self.concept_preprocessing_train_image_ids),
        }

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
