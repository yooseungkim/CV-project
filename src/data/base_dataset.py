from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    """Abstract base class for CBM datasets."""
    
    @classmethod
    def get_default_config(cls) -> dict:
        """Returns the default dataset configuration dictionary containing:
        - num_concepts: int
        - num_classes: int
        - concepts: List[str]
        - target_col: str
        """
        return {
            "num_concepts": 0,
            "num_classes": 1,
            "concepts": [],
            "target_col": ""
        }

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - image_tensor: The input image tensor.
            - concept_labels_tensor: The ground truth concept labels.
            - target_label_tensor: The ground truth target label(s).
        """
        pass
