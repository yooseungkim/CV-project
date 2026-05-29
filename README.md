# Attention-Based Modular Concept Bottleneck Model (Attention-CBM) Pipeline

This project provides a highly modular, dataset-agnostic Attention-Based Concept Bottleneck Model (Attention-CBM) pipeline designed for medical and fine-grained visual categorization. 

## Features
- **Spatial Concept Grounding:** Uses a Cross-Attention mechanism (`nn.MultiheadAttention`) with learnable concept queries over 2D spatial feature maps, completely replacing Global Average Pooling (GAP) for pixel-level explainability.
- **Universal Flexible Backbone:** Built primarily for `timm` backbones (which dynamically preserve spatial dimensions via `global_pool=''`). The model dynamically infers feature dimensions, requiring no hardcoded values.
- **Unified Data Pipeline:** Abstract `BaseDataset` ensures all loaders produce a standardized tuple `(image_tensor, concept_labels_tensor, target_label_tensor)` and support unified `dataset_config` parameterization.
- **Dynamic Config Inference:** Standalone utility script to auto-generate CBM concept configs by analyzing dataset metadata CSV profiles with intelligent type-inference heuristics.
- **Hybrid Loss Training:** Built-in support for target classification loss and concept bottleneck loss, weighted by a `--lambda_c` hyperparameter.
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
├── generate_concept_config.py # Standalone script to auto-generate CBM concept configs
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

Run the training pipeline using the `main.py` entry point. The script will automatically adapt the internal CBM layer structures based on the provided `--concept_config_path` and `--num_classes`.

> **Note:** The `open_clip` backbone is currently not supported for the Attention-CBM architecture due to the requirement for raw 2D spatial feature maps, which are uniformly extracted from `timm` models via `global_pool=''`.

### 1. Training with a `timm` Backbone (ResNet50)
```bash
uv run python main.py \
    --dataset milk10k \
    --concept_config_path data/MILK10K/concept_config.json \
    --backbone_type timm \
    --backbone_name resnet50 \
    --lambda_c 1.0 \
    --num_classes 1 \
    --epochs 5 \
    --batch_size 16
```

### 2. Sequential Training (Freezing Backbone)
Freeze the vision backbone and only train the attention queries, projection layers, and classifier:
```bash
uv run python main.py \
    --dataset milk10k \
    --concept_config_path data/MILK10K/concept_config.json \
    --backbone_type timm \
    --backbone_name convnext_base \
    --freeze_backbone
```

### 3. Weights & Biases Logging & Backbone Identification
Weights & Biases (wandb) logging is enabled by default. The run names are dynamically generated using the backbone name and current timestamp:
- Example: `resnet50-cbm-20260530_010616`

All CLI arguments (such as `backbone_name`, `backbone_type`, and `lambda_c`) are automatically logged under the wandb `config` dictionary, allowing you to easily filter, group, and compare runs by backbone in the wandb dashboard.

To disable wandb logging (for instance during debug runs), pass `--use_wandb False`:
```bash
uv run python main.py \
    --dataset milk10k \
    --concept_config_path data/MILK10K/concept_config.json \
    --backbone_type timm \
    --backbone_name resnet50 \
    --use_wandb False
```

---

## Model Weight Saving

After training completes, the model weights (`state_dict`) are automatically saved to your configured `--save_dir` grouped into backbone-specific directories. The saved weights filename contains the run timestamp and the backbone training mode:

- **Default Root Directory:** `checkpoints/{backbone_name}/`
- **Filename Pattern:** `{YYYYMMDD_HHMMSS}_cbm_{mode}.pth` (e.g. `20260530_010630_cbm_full.pth` or `20260530_010630_cbm_frozen_backbone.pth`)
- **Override Path:** Customize the root save folder using `--save_dir [PATH]` (default: `checkpoints`)

---

## Auto-Generating Concept Configuration Files

This project includes a standalone script `generate_concept_config.py` that fully automates the creation of a structured Concept Configuration file by parsing your dataset's metadata CSV.

### Heuristics & Rules
- **Type Inference:** Auto-classifies each metadata column (excluding ignored ones) into `categorical` (if `object`, `bool`, `string`, or numeric with `< 15` unique values) or `numerical` (if numeric with `>= 15` unique values).
- **Metadata Extraction:**
  - For **Categorical** concepts: Compiles all unique values as a sorted `classes` list (ignoring NaNs).
  - For **Numerical** concepts: Auto-calculates `min` and `max` scaling bounds (ignoring NaNs).
- **Type Safety:** Auto-converts numpy datatypes into native Python formats to guarantee serializable outputs.

### CLI Usage Example
Generate a `concept_config.json` profile for your dataset by ignoring non-concept identifiers:
```bash
uv run python generate_concept_config.py \
    --csv_path data/MILK10K/MILK10k_Test_Metadata.csv \
    --ignore_cols lesion_id,image_type,isic_id,attribution,copyright_license,image_manipulation \
    --output_path concept_config.json
```

> [!TIP]
> **Manual Refinements:** The generated configuration is a standard JSON/YAML file that serves as a highly robust starting point. It is fully encouraged to manually inspect, edit, and adjust this file after generation. For example, you can manually reclassify a concept from `numerical` to `categorical`, adjust the min-max boundaries, correct category order, or prune unwanted concepts.

## Adding a New Dataset
1. Create a new dataset class in `src/data/` that inherits from `BaseDataset`.
2. Implement the `__len__` and `__getitem__` methods.
3. Import and initialize it in `main.py` when the corresponding `--dataset` argument is passed. No changes to the model architecture are required!
