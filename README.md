# Attention-Based Modular Concept Bottleneck Model (Attention-CBM) Pipeline

This project provides a modular, dataset-agnostic Attention-Based Concept Bottleneck Model (Attention-CBM) pipeline for medical and fine-grained visual categorization.

---

## Key Features

- **Attention-Based Concept Prediction:** CNN backbones keep spatial feature maps (`global_pool=''`), while ViT/DINOv2/ConvNeXt backbones keep patch or token features. Concept heads compute logits and attention maps from those features instead of using a GAP-pooled image vector for concept prediction.
- **Config-Driven Datasets and Concepts:** YAML files and `concept_config.json` define datasets, target classes, categorical concept groups, numeric concept ranges, and training defaults. The project includes configs for MILK10K, CUB-200-2011, Derm7pt, and CheXpert.
- **Backbone Training Modes:** `--backbone_train_mode {frozen,lora,full}` controls whether the backbone is frozen, adapted with LoRA, or fully fine-tuned. LoRA is supported for ViT/DINOv2-style backbones, and checkpoints are loaded with compatible mode detection.
- **Training Workflow Support:** Multi-stage training uses `--resume_checkpoint` and `--save_filename`, phase-specific early stopping, inverse-frequency class weights for imbalanced targets, optional dropout on bottleneck activations, and optional post-hoc concept bias calibration.
- **Evaluation and TTI:** `eval_cbm.py` reports concept, target, and **Classification (GT Concept)** metrics with Accuracy, Macro-F1, and Macro-F2. Categorical concept groups are evaluated with argmax one-hot predictions, singleton concepts use `logit > 0`, and TTI benchmarks include group-level, concept-level, uncertainty-based, and CooP policies.
- **Interactive Gradio Explorer:** The app loads checkpoints, shows grouped per-concept attention maps, provides dropdowns for categorical interventions and physical-scale sliders for numeric concepts, and can open uncertain categorical controls based on the top-1/top-2 probability margin.

---

## Directory Structure
```
project_root/
├── checkpoints/            # Saved weights organized by backbone model name
├── configs/
│   ├── train_config.yaml   # Model, optimizer, scheduler, and early-stop settings
│   ├── cub_train_config.yaml # CUB LoRA, calibration, and CooP evaluation settings
│   └── ...                 # Dataset-specific configs for MILK10K, Derm7pt, and CheXpert
├── data/                   # (Ignored in git) Store MILK10K, CUB, and other raw data here
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base_dataset.py # Abstract base class for datasets
│   │   └── milk10k.py      # MILK10K multi-class dataset implementation
│   ├── models/
│   │   ├── __init__.py
│   │   └── cbm_factory.py  # UniversalFlexibleCBM implementation
│   ├── tti/
│   │   ├── common.py       # Shared TTI metrics/logit helpers
│   │   └── coop.py         # CooP scoring and validation fitting
│   └── utils/
│       ├── __init__.py
│       ├── concept_bias.py # Calibration split and concept-bias learning
│       └── metrics.py      # Utilities for evaluating concept and target accuracy
├── app.py                  # Gradio checkpoint explorer web application
├── download_CUB-200-2011.py # CUB download, verification, and extraction script
├── eval_cbm.py             # Evaluation and TTI benchmark entry point
├── generate_concept_config.py # Utility script to auto-generate CBM concept configs
├── main.py                 # Entry point for training and evaluation
├── requirements.txt        # Dependencies exported from uv
└── README.md
```

---

## Setup Environment

This project utilizes `uv` for fast dependency management.

```bash
# Initialize uv if not already done
uv init --python 3.12

# Install dependencies and sync the lockfile
uv sync
```

*(Alternatively, use `pip install -r requirements.txt` if not using `uv`)*

---

## Usage Examples

Run the training pipeline with the `main.py` entry point. The model, dataset, concept, and training settings are read from the YAML file passed through `--config_path`.

### 1. Basic CBM Training (Using Config Defaults)
Train a CBM with the defaults in `configs/train_config.yaml`:
```bash
uv run python main.py --config_path configs/train_config.yaml
```

### 2. Basic CBM Training with Parameter Overrides
Override selected training parameters from the command line:
```bash
uv run python main.py \
    --config_path configs/train_config.yaml \
    --epochs 15 \
    --batch_size 32 \
    --lambda_c 3.0 \
    --lr 0.0005
```

### 3. Backbone Training Modes
Control backbone trainability with `--backbone_train_mode`. Use `frozen` to preserve pre-trained features, `lora` for parameter-efficient ViT/DINOv2 adaptation, or `full` for full backbone fine-tuning:
```bash
uv run python main.py --config_path configs/train_config.yaml --backbone_train_mode frozen
uv run python main.py --config_path configs/cub_train_config.yaml --backbone_train_mode lora
uv run python main.py --config_path configs/train_config.yaml --backbone_train_mode full
```

### 4. Freezing the Classifier Head
Freeze the classification head while updating the backbone and concept projection layers:
```bash
uv run python main.py --config_path configs/train_config.yaml --freeze_head
```

### 5. Multi-Stage Training
Chain multiple training stages together in sequence using `--save_filename` and `--resume_checkpoint`:
- **Stage 1:** Train for 10 epochs with a higher concept penalty (`lambda_c = 5.0`) and save the result as `phase1_cbm.pth`.
- **Stage 2:** Load Phase 1 weights, reduce the concept penalty (`lambda_c = 0.5`), and train for another 10 epochs before saving `phase2_cbm.pth`.

```bash
uv run python main.py --config_path configs/train_config.yaml --epochs 10 --lambda_c 5.0 --save_filename phase1_cbm.pth && \
uv run python main.py --config_path configs/train_config.yaml --epochs 10 --lambda_c 0.5 --resume_checkpoint checkpoints/resnet50/phase1_cbm.pth --save_filename phase2_cbm.pth
```

### 6. CUB-200-2011 Download and Training
Download and verify the CUB archive, then generate the project concept config:
```bash
uv run python download_CUB-200-2011.py --data-dir data
uv run python scratch/convert_cub_attributes.py
```

Train with the CUB configuration, which enables LoRA backbone training, concept bias calibration, and evaluation defaults:
```bash
uv run python main.py --config_path configs/cub_train_config.yaml
```

### 7. Concept Bias Calibration
Enable calibration from YAML by adding `learn_concept_bias` to `calibration.for_what`. Learned concept-bias buffers are saved in the checkpoint:
```yaml
calibration:
  source_split: train
  ratio: 0.10
  seed: 42
  for_what:
    - learn_concept_bias

learn_concept_bias:
  objective:
    metric: target_nll
  parameterization: singleton_only
  temperature: 1.1
  l2_lambda: 0.003
```

### 8. Evaluation and TTI Benchmark
Run standard CBM evaluation plus TTI benchmarks using `eval_cbm.py`:
```bash
uv run python eval_cbm.py \
    --checkpoint PATH_TO_CHECKPOINT \
    --config_path configs/cub_train_config.yaml
```

Common evaluation switches:
```bash
uv run python eval_cbm.py --checkpoint PATH_TO_CHECKPOINT --without-tti
uv run python eval_cbm.py --checkpoint PATH_TO_CHECKPOINT --without-coop-fit --coop-score-mode all
uv run python eval_cbm.py --checkpoint PATH_TO_CHECKPOINT --without-coop-tti
uv run python eval_cbm.py --checkpoint PATH_TO_CHECKPOINT --ignore-bias
```

### 9. Gradio Web Application (Interactive Explorer & Intervention)
Once the model is trained, launch the Gradio explorer using your checkpoint:
```bash
uv run python app.py --checkpoint checkpoints/resnet50/phase2_cbm.pth --num_classes 11 --port 7860
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your web browser to:
1. Drag and drop dermoscopy images to predict the **Top-3 highest-probability classes**.
2. View **filtered attention maps** showing only the selected category for categorical attributes.
3. Manually edit clinical parameters using **compact dropdowns** (e.g. sex, site) and **physical sliders** (e.g. age) to inspect how interventions change predictions.

---

## Auto-Generating Concept Configuration Files

If working with custom datasets, use the `generate_concept_config.py` script to inspect your metadata and generate CBM-ready configuration files:

```bash
uv run python generate_concept_config.py \
    --csv_path data/MILK10K/MILK10k_Test_Metadata.csv \
    --ignore_cols lesion_id,image_type,isic_id,attribution,copyright_license,image_manipulation \
    --output_path concept_config.json
```
