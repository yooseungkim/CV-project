# Attention-Based Modular Concept Bottleneck Model (Attention-CBM) Pipeline

This project provides a highly modular, dataset-agnostic Attention-Based Concept Bottleneck Model (Attention-CBM) pipeline designed for medical and fine-grained visual categorization. 

---

## 🚀 Key Premium Features

- **Spatial Concept Grounding:** Uses a Cross-Attention mechanism (`nn.MultiheadAttention`) with learnable concept queries over 2D spatial feature maps, completely replacing Global Average Pooling (GAP) for pixel-level explainability.
- **Multi-Class Disease Classification:** Transitioned from binary labels to full 11-class multi-class disease classification (`AKIEC`, `BCC`, `BEN_OTH`, `BKL`, `DF`, `INF`, `MAL_OTH`, `MEL`, `NV`, `SCCKA`, `VASC`) integrated across dataset loading, metrics, loss functions, and inference.
- **Class-Imbalance Mitigation:** Dynamically computes **Inverse-Frequency Class Weights** from the training dataset distribution and integrates them into `nn.CrossEntropyLoss` to handle severe class imbalances (e.g., BCC: 2522 vs. MAL_OTH: 9) during multi-class training.
- **Model Regularization (Dropout):** Integrates `nn.Dropout(p=0.2)` on the concept bottleneck activations to eliminate noise and improve target prediction robustness while preserving strict checkpoint compatibility.
- **Advanced Gradio Web Application & Human-in-the-Loop:**
  - **Filtered Attention Maps:** Categorical sub-classes are logically grouped under their parent concepts (e.g., `site_foot`, `site_genital` -> `site`). The visualization **only shows the single highest-probability (argmax) choice** for each categorical concept, keeping the UI clean (exactly 11 active heatmaps instead of 22 cluttered ones).
  - **Categorical Dropdowns:** Replaced cluttered individual $[0.0, 1.0]$ sliders for categorical choices with a **single, unified Dropdown component** per concept (e.g. `site` selection dropdown). Selecting an option automatically one-hot encodes it behind the scenes.
  - **Scaled Numerical Sliders:** Sliders for numerical concepts are automatically bounded by their real physical min/max values defined in `concept_config.json` (e.g., `age_approx` is $5$ to $85$ years).
    - *Forward Prediction:* Scales $[0, 1]$ model sigmoid values to real physical scales.
    - *Intervention:* Normalizes user's physical adjustments back to $[0, 1]$ before passing them to the model classifier.
  - **Ultra-Compact Multi-Column Layout:** Employs a beautiful 2-column layout panel inside a custom-scrollable box with optimized CSS to fit all controllers on a single screen.
- **Advanced Multi-Stage Resuming & Transfer Learning:**
  - Exposes the `--resume_checkpoint` argument (and YAML parameter) in `main.py`.
  - Employs a **hybrid checkpoint loading strategy**: tries strict loading (`strict=True`) first, and automatically falls back to non-strict loading (`strict=False`) with a warning. This allows seamless transfer of backbone and concept attention weights even when modifying task settings (such as binarization or changing class counts).
- **Flexible Weight Saving:** Introduced `--save_filename` to allow specifying static output filenames (e.g., `phase1_cbm.pth`), making multi-stage chained training highly scriptable.
- **Per-Concept Validation Tracking:** 
  - Computes and logs the individual validation accuracy of **all 22 flattened concepts** to Weights & Biases (`val_concept_acc/{concept_name}`).
  - Automatically prints the **Top-3 struggling concepts** to the console at the end of each validation epoch for instant developer feedback.
- **Ultra-Fast Memory Caching:** Support for full dataset RAM-caching (`cache_in_memory: true`) completely eliminates disk I/O bottlenecks during training.

---

## Directory Structure
```
project_root/
├── checkpoints/            # Saved weights organized by backbone model name
├── configs/
│   └── train_config.yaml   # Comprehensive model, optimizer, scheduler, & early-stop settings
├── data/                   # (Ignored in git) Store MILK10K and other raw data here
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base_dataset.py # Abstract base class for datasets
│   │   └── milk10k.py      # MILK10K multi-class dataset implementation
│   ├── models/
│   │   ├── __init__.py
│   │   └── cbm_factory.py  # UniversalFlexibleCBM implementation
│   └── utils/
│       ├── __init__.py
│       └── metrics.py      # Utilities for evaluating concept and target accuracy
├── app.py                  # High-performance Gradio explorer web application
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

Run the training pipeline using the `main.py` entry point. The script will automatically adapt the internal CBM layer structures based on the provided `--config_path` YAML specifications.

### 1. Basic CBM Training (Using Config defaults)
Standard multi-class training of the full CBM (backbone + bottleneck attention queries + classification head) using optimal settings defined in `configs/train_config.yaml`:
```bash
uv run python main.py --config_path configs/train_config.yaml
```

### 2. Basic CBM Training with Parameter Overrides
Override specific training parameters on the fly via the command line (e.g. increase epochs, adjust learning rate, or set concept loss weight):
```bash
uv run python main.py \
    --config_path configs/train_config.yaml \
    --epochs 15 \
    --batch_size 32 \
    --lambda_c 3.0 \
    --lr 0.0005
```

### 3. Sequential Training: Freezing the Vision Backbone
Freeze thevision backbone to train **only** the spatial attention layers and the classification head (extremely useful to preserve pre-trained backbone features or speed up training):
```bash
uv run python main.py --config_path configs/train_config.yaml --freeze_backbone
```

### 4. Sequential Training: Freezing the Classifier Head
Freeze the classification head parameters to optimize **only** the backbone fine-tuning and bottleneck concept projection:
```bash
uv run python main.py --config_path configs/train_config.yaml --freeze_head
```

### 5. Multi-Stage Chained Training (Sequential optimization)
Chain multiple training stages together in sequence using `--save_filename` and `--resume_checkpoint`:
- **Stage 1:** Train for 10 epochs with a strong concept penalty (`lambda_c = 5.0`) to force the model to ground the concept bottlenecks properly, and save it as `phase1_cbm.pth`.
- **Stage 2:** Load Phase 1 weights, loosen the concept penalty (`lambda_c = 0.5`), and train for another 10 epochs to maximize target accuracy, saving it as `phase2_cbm.pth`.

```bash
uv run python main.py --config_path configs/train_config.yaml --epochs 10 --lambda_c 5.0 --save_filename phase1_cbm.pth && \
uv run python main.py --config_path configs/train_config.yaml --epochs 10 --lambda_c 0.5 --resume_checkpoint checkpoints/resnet50/phase1_cbm.pth --save_filename phase2_cbm.pth
```

### 6. Gradio Web Application (Interactive Explorer & Intervention)
Once the model is trained, launch the premium interactive Gradio explorer using your checkpoint:
```bash
uv run python app.py --checkpoint checkpoints/resnet50/phase2_cbm.pth --num_classes 11 --port 7860
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your web browser to:
1. Drag and drop dermoscopy images to predict the **Top-3 highest-probability classes**.
2. View **filtered attention maps** showing only the selected category for categorical attributes.
3. Manually edit clinical parameters using **compact dropdowns** (e.g. sex, site) and **physical sliders** (e.g. age) to see how the model dynamically updates predictions (Human-in-the-Loop intervention).

---

## Auto-Generating Concept Configuration Files

If working with custom datasets, use the `generate_concept_config.py` script to inspect your metadata and generate CBM-ready configuration files:

```bash
uv run python generate_concept_config.py \
    --csv_path data/MILK10K/MILK10k_Test_Metadata.csv \
    --ignore_cols lesion_id,image_type,isic_id,attribution,copyright_license,image_manipulation \
    --output_path concept_config.json
```
