import os
import glob
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseDataset(Dataset, ABC):
    """Abstract base class for CBM datasets with optional in-memory caching.
    
    When `cache_in_memory=True` and the estimated dataset size is below 
    `max_cache_size_gb` (default: 10 GB), all samples are preloaded into RAM
    after __init__. This eliminates repeated disk I/O and transform computation
    during training.
    """
    
    # Cache storage — initialized as None until populated
    _cache: Optional[list] = None
    _cache_populated: bool = False
    
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
            "target_col": "",
            "target_classes": []
        }

    def _estimate_dataset_size_gb(self) -> float:
        """Estimates the on-disk size of the image directory in GB.
        
        Scans common image extensions (jpg, jpeg, png, bmp, tiff) in self.image_dir.
        Returns 0.0 if the directory doesn't exist.
        """
        image_dir = getattr(self, 'image_dir', None)
        if not image_dir or not os.path.isdir(image_dir):
            return 0.0
        
        total_bytes = 0
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
        for ext in extensions:
            for filepath in glob.iglob(os.path.join(image_dir, '**', ext), recursive=True):
                total_bytes += os.path.getsize(filepath)
        
        return total_bytes / (1024 ** 3)

    def _try_populate_cache(self, max_cache_size_gb: float = 10.0):
        """Attempts to populate the in-memory cache.
        
        Should be called at the END of the subclass __init__ after all attributes
        (self.df, self.image_dir, self.transform, etc.) are fully initialized.
        
        Args:
            max_cache_size_gb: Maximum dataset size (on-disk) to allow caching.
        """
        cache_enabled = getattr(self, 'cache_in_memory', False)
        if not cache_enabled:
            return
        
        estimated_gb = self._estimate_dataset_size_gb()
        
        if estimated_gb > max_cache_size_gb:
            tqdm.write(
                f"  [Cache] Warning: Dataset too large for caching ({estimated_gb:.2f} GB > {max_cache_size_gb} GB). "
                f"Falling back to on-the-fly loading."
            )
            self.cache_in_memory = False
            return
        
        tqdm.write(
            f"  [Cache] Caching {len(self)} samples into memory "
            f"(estimated disk size: {estimated_gb:.2f} GB)..."
        )
        
        self._cache = [None] * len(self)
        # Temporarily disable cache reads while populating
        self._cache_populated = False
        
        for idx in tqdm(range(len(self)), desc=f"  Caching [{self.split}]", leave=False):
            self._cache[idx] = self._load_sample(idx)
        
        self._cache_populated = True
        tqdm.write(f"  [Cache] Cache ready for [{self.split}] split ({len(self)} samples)")

    @abstractmethod
    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Loads a single sample from disk. Must be implemented by subclasses.
        
        This is the actual data loading logic (previously in __getitem__).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - image_tensor: The input image tensor.
            - concept_labels_tensor: The ground truth concept labels.
            - target_label_tensor: The ground truth target label(s).
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a sample, from cache if available, otherwise loads from disk."""
        if self._cache_populated and self._cache is not None:
            sample = self._cache[idx]
        else:
            sample = self._load_sample(idx)
            
        if len(sample) == 3:
            dummy_tabular = torch.zeros(3, dtype=torch.float32)
            return sample + (dummy_tabular,)
        return sample
