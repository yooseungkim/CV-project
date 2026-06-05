import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from src.data.base_dataset import BaseDataset

class CheXpertDataset(BaseDataset):
    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "num_concepts": 9,
            "num_classes": 5,
            "concepts": [
                'No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 
                'Lung Lesion', 'Pneumonia', 'Pneumothorax', 'Pleural Other', 
                'Fracture', 'Support Devices'
            ],
            "target_col": ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
            "default_csv_path": "data/CheXpert/train.csv",
            "default_image_dir": "data/CheXpert/",
            "concept_config_path": "data/CheXpert/concept_config.json",
            "policy": "u-ones",
            "subset_ratio": None
        }

    def __init__(
        self,
        csv_file=None,
        root_dir=None,
        transform=None,
        policy=None,
        subset_ratio=None,
        csv_path=None,
        image_dir=None,
        split='train',
        config=None,
        cache_in_memory=False,
        max_cache_size_gb=10.0
    ):
        super().__init__()
        self.split = split.lower()
        
        # Initialize config
        self.config = config or self.get_default_config()
        
        # Support both standalone signature and main.py config convention
        resolved_csv = csv_file or csv_path or self.config.get("default_csv_path")
        
        # Automatically route to the validation or test CSV if main.py passed train.csv
        if resolved_csv and resolved_csv.endswith("train.csv"):
            if self.split in ['val', 'valid']:
                resolved_csv = resolved_csv.replace("train.csv", "valid.csv")
            elif self.split == 'test':
                resolved_csv = resolved_csv.replace("train.csv", "test.csv")
                
        self.root_dir = root_dir or image_dir or self.config.get("default_image_dir")
        self.image_dir = self.root_dir  # BaseDataset expects self.image_dir for size estimation
        
        self.policy = policy or self.config.get("policy", "u-ones")
        if self.policy not in ["u-ones", "u-zeros"]:
            raise ValueError("policy must be 'u-ones' or 'u-zeros'")
            
        self.subset_ratio = subset_ratio if subset_ratio is not None else self.config.get("subset_ratio", None)
        
        self.concepts_cols = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 
            'Lung Lesion', 'Pneumonia', 'Pneumothorax', 'Pleural Other', 
            'Fracture', 'Support Devices'
        ]
        self.concept_cols = self.concepts_cols
        self.concept_features_info = None
        self.targets_cols = [
            'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'
        ]
        self.all_cols = self.concepts_cols + self.targets_cols
        
        self.dummy_mode = not os.path.exists(resolved_csv) if resolved_csv else True
        if self.dummy_mode:
            print(f"Warning: CSV file not found at {resolved_csv}. Running in dummy mode ({self.split} split).")
            self.df = pd.DataFrame()
            self.concepts_data = np.zeros((10, len(self.concepts_cols)), dtype=np.float32)
            self.targets_data = np.zeros((10, len(self.targets_cols)), dtype=np.float32)
        else:
            self.df = pd.read_csv(resolved_csv)
            
            # Filter for frontal images only
            if 'Frontal/Lateral' in self.df.columns:
                self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)
            
            # Optionally sample a subset of the dataset (only for training split)
            if self.split == 'train' and self.subset_ratio is not None:
                if not (0.0 < self.subset_ratio <= 1.0):
                    raise ValueError("subset_ratio must be between 0.0 and 1.0")
                self.df = self.df.sample(frac=self.subset_ratio, random_state=42).reset_index(drop=True)
                
            # Preprocess all 14 label columns
            for col in self.all_cols:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0.0)
                if self.policy == "u-ones":
                    self.df[col] = self.df[col].replace(-1.0, 1.0)
                elif self.policy == "u-zeros":
                    self.df[col] = self.df[col].replace(-1.0, 0.0)
                    
            # Store preprocessed labels as numpy arrays for efficient access
            self.concepts_data = self.df[self.concepts_cols].to_numpy(dtype=np.float32)
            self.targets_data = self.df[self.targets_cols].to_numpy(dtype=np.float32)
        
        # Set transform (only default if config is provided, otherwise keep None as requested)
        self.transform = transform
        if self.transform is None and config is not None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10, expand=False, fill=0),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        # In-memory caching setup
        self.cache_in_memory = cache_in_memory
        self._cache = None
        self._cache_populated = False
        self._try_populate_cache(max_cache_size_gb=max_cache_size_gb)

    def __len__(self):
        if self.dummy_mode:
            return 10
        return len(self.df)

    def _load_sample(self, idx):
        if self.dummy_mode:
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            if self.transform:
                image = self.transform(image)
            concepts = torch.tensor(self.concepts_data[idx], dtype=torch.float32)
            targets = torch.tensor(self.targets_data[idx], dtype=torch.float32)
            return image, concepts, targets
            
        row = self.df.iloc[idx]
        path_str = str(row['Path'])
        if path_str.startswith("CheXpert-v1.0-small/"):
            path_str = path_str[len("CheXpert-v1.0-small/"):]
        img_path = os.path.join(self.root_dir, path_str)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        concepts = torch.tensor(self.concepts_data[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets_data[idx], dtype=torch.float32)
        
        return image, concepts, targets

    def _estimate_dataset_size_gb(self) -> float:
        if self.dummy_mode or self.df.empty:
            return 0.0
        
        sample_df = self.df.sample(n=min(100, len(self.df)), random_state=42)
        total_sample_bytes = 0
        valid_samples = 0
        for _, row in sample_df.iterrows():
            path_str = str(row['Path'])
            if path_str.startswith("CheXpert-v1.0-small/"):
                path_str = path_str[len("CheXpert-v1.0-small/"):]
            img_path = os.path.join(self.root_dir, path_str)
            if os.path.exists(img_path):
                total_sample_bytes += os.path.getsize(img_path)
                valid_samples += 1
                
        if valid_samples == 0:
            return 0.0
            
        avg_size = total_sample_bytes / valid_samples
        estimated_total_bytes = avg_size * len(self.df)
        return estimated_total_bytes / (1024 ** 3)
