# Modular Concept Bottleneck Model (CBM) Pipeline

This project provides a highly modular, dataset-agnostic Concept Bottleneck Model (CBM) pipeline designed for medical and fine-grained visual categorization. 

## Features
- **Universal Flexible Backbone:** Easily swap between `timm` and `open_clip` backbones. The model dynamically infers feature dimensions, requiring no hardcoded values.
- **Unified Data Pipeline:** Abstract `BaseDataset` ensures all loaders produce a standardized tuple `(image_tensor, concept_labels_tensor, target_label_tensor)`.
- **Object-Oriented Design:** Clear separation of concerns between data loading, model architecture, training loop, and metrics.
- **Sequential Training Support:** Built-in arguments to independently freeze the vision backbone or the classifier head.

## Directory Structure
```
project_root/
├── data/                   # (Ignored in git) Store MILK10K and other data here
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base_dataset.py # Abstract base class for datasets
│   │   └── milk10k.py      # MILK10K dataset implementation
│   ├── models/
│   │   ├── __init__.py
│   │   └── cbm_factory.py  # UniversalFlexibleCBM implementation
│   └── utils/
│       ├── __init__.py
│       └── metrics.py      # Utilities for evaluating concept and target accuracy
├── main.py                 # Entry point for training and evaluation
├── requirements.txt        # Dependencies exported from uv
└── README.md
```

## Setup Environment

This project utilizes `uv` for fast dependency management.

```bash
# Initialize uv if not already done
uv init --python 3.12

# Dependencies are managed in pyproject.toml / uv.lock.
# You can install the environment with:
uv sync
```

*(Alternatively, use `pip install -r requirements.txt` if not using `uv`)*

## Usage Examples

Run the training pipeline using the `main.py` entry point. The script will automatically adapt the internal CBM layer structures based on the provided `--num_concepts` and `--num_classes`.

### 1. Training with a `timm` Backbone (ResNet50)
```bash
uv run python main.py \
    --dataset milk10k \
    --backbone_type timm \
    --backbone_name resnet50 \
    --num_concepts 7 \
    --num_classes 1 \
    --epochs 5 \
    --batch_size 16
```

### 2. Training with an `open_clip` Backbone (BioCLIP)
```bash
uv run python main.py \
    --dataset milk10k \
    --backbone_type clip \
    --backbone_name hf-hub:imageomics/bioclip \
    --num_concepts 7 \
    --num_classes 1 \
    --epochs 5 \
    --batch_size 16
```

### 3. Sequential Training (Freezing Backbone)
Freeze the vision backbone and only train the projection layers and classifier:
```bash
uv run python main.py \
    --dataset milk10k \
    --backbone_type timm \
    --backbone_name convnext_base \
    --num_concepts 7 \
    --freeze_backbone
```

### 4. Weights & Biases Logging
Weights & Biases (wandb) logging is enabled by default. To disable wandb logging (for instance during debug runs), pass `--use_wandb False`:
```bash
uv run python main.py \
    --dataset milk10k \
    --backbone_type timm \
    --backbone_name resnet50 \
    --num_concepts 7 \
    --use_wandb False
```

## Adding a New Dataset
1. Create a new dataset class in `src/data/` that inherits from `BaseDataset`.
2. Implement the `__len__` and `__getitem__` methods.
3. Import and initialize it in `main.py` when the corresponding `--dataset` argument is passed. No changes to the model architecture are required!
