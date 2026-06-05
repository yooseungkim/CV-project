import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, policy="u-ones", subset_ratio=None):
        if policy not in ["u-ones", "u-zeros"]:
            raise ValueError("policy must be 'u-ones' or 'u-zeros'")
            
        self.root_dir = root_dir
        self.transform = transform
        self.policy = policy
        
        self.concepts_cols = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 
            'Lung Lesion', 'Pneumonia', 'Pneumothorax', 'Pleural Other', 
            'Fracture', 'Support Devices'
        ]
        self.targets_cols = [
            'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'
        ]
        self.all_cols = self.concepts_cols + self.targets_cols
        
        self.df = pd.read_csv(csv_file)
        
        # Optionally sample a subset of the dataset
        if subset_ratio is not None:
            if not (0.0 < subset_ratio <= 1.0):
                raise ValueError("subset_ratio must be between 0.0 and 1.0")
            self.df = self.df.sample(frac=subset_ratio, random_state=42).reset_index(drop=True)
        
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, str(row['Path']))
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        concepts = torch.tensor(self.concepts_data[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets_data[idx], dtype=torch.float32)
        
        return image, concepts, targets
