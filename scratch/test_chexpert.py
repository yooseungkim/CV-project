import os
import tempfile
import pandas as pd
import numpy as np
import torch
from PIL import Image
from src.data.chexpert import CheXpertDataset

def run_test():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create a dummy image
        img_filename = "dummy_image.png"
        img_path = os.path.join(tmpdir, img_filename)
        Image.new('RGB', (100, 100), color='red').save(img_path)
        
        # 2. Create a dummy CSV with 4 rows (3 Frontal, 1 Lateral)
        data = {
            'Path': [img_filename, img_filename, img_filename, img_filename],
            'Frontal/Lateral': ['Frontal', 'Frontal', 'Lateral', 'Frontal'],
            'No Finding': [1.0, -1.0, 0.0, 1.0],
            'Enlarged Cardiomediastinum': [0.0, np.nan, 1.0, 0.0],
            'Lung Opacity': [-1.0, "", 0.0, -1.0],
            'Lung Lesion': [0.0, 1.0, 1.0, 0.0],
            'Pneumonia': [1.0, 0.0, 0.0, 1.0],
            'Pneumothorax': [np.nan, -1.0, 0.0, 0.0],
            'Pleural Other': [" ", 1.0, 1.0, 0.0],
            'Fracture': [0.0, 0.0, 0.0, 0.0],
            'Support Devices': [1.0, 1.0, 1.0, 1.0],
            # Targets:
            'Cardiomegaly': [-1.0, np.nan, 1.0, 0.0],
            'Edema': ["", 1.0, 0.0, 0.0],
            'Consolidation': [1.0, 0.0, 1.0, 0.0],
            'Atelectasis': [0.0, -1.0, 0.0, 0.0],
            'Pleural Effusion': [1.0, " ", 0.0, 1.0]
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(tmpdir, "metadata.csv")
        df.to_csv(csv_path, index=False)
        
        # 3. Test with policy = "u-ones"
        # Total rows = 4, but 1 is Lateral, so only 3 Frontal rows should remain!
        dataset_u_ones = CheXpertDataset(csv_file=csv_path, root_dir=tmpdir, policy="u-ones")
        assert len(dataset_u_ones) == 3
        
        img, concepts, targets = dataset_u_ones[0]
        
        # Assert type and shape
        assert isinstance(img, Image.Image)
        assert img.size == (100, 100)
        assert isinstance(concepts, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert concepts.dtype == torch.float32
        assert targets.dtype == torch.float32
        assert concepts.ndim == 1 and concepts.shape[0] == 9
        assert targets.ndim == 1 and targets.shape[0] == 5
        
        expected_concepts_0 = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        expected_targets_0 = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
        assert torch.allclose(concepts, expected_concepts_0)
        assert torch.allclose(targets, expected_targets_0)
        
        # 4. Test with policy = "u-zeros"
        dataset_u_zeros = CheXpertDataset(csv_file=csv_path, root_dir=tmpdir, policy="u-zeros")
        img_z, concepts_z, targets_z = dataset_u_zeros[0]
        
        expected_concepts_z0 = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        expected_targets_z0 = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
        assert torch.allclose(concepts_z, expected_concepts_z0)
        assert torch.allclose(targets_z, expected_targets_z0)
        
        # 5. Test subset ratio (e.g. subset_ratio=0.66 -> should be ~2 out of 3 frontal rows)
        dataset_subset = CheXpertDataset(csv_file=csv_path, root_dir=tmpdir, policy="u-ones", subset_ratio=0.66)
        assert len(dataset_subset) == 2
        
        # 6. Test invalid subset ratio
        try:
            CheXpertDataset(csv_file=csv_path, root_dir=tmpdir, subset_ratio=-0.1)
            assert False, "Should raise ValueError for negative subset ratio"
        except ValueError:
            pass
            
        try:
            CheXpertDataset(csv_file=csv_path, root_dir=tmpdir, subset_ratio=1.5)
            assert False, "Should raise ValueError for ratio > 1.0"
        except ValueError:
            pass
        
        # 7. Test transform
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset_transform = CheXpertDataset(csv_file=csv_path, root_dir=tmpdir, transform=transform, policy="u-ones")
        img_t, _, _ = dataset_transform[0]
        assert isinstance(img_t, torch.Tensor)
        assert img_t.shape == (3, 100, 100)
        
        print("ALL TESTS (INCLUDING FRONTAL FILTER & SUBSET RATIO) PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_test()
