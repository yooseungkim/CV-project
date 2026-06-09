# Attention-Based Modular Concept Bottleneck Model (Attention-CBM) Pipeline

This project classifies birds with a Concept Bottleneck Model to build an interpretable classification model.

Instead of predicting labels directly from images with a black-box model, a CBM inserts a concept bottleneck in the middle of the pipeline. The model first predicts human-recognizable attributes, then uses those attributes to predict the final label.

This project uses the CUB-200-2011 dataset. The dataset contains 11,788 images representing 200 bird species. Each image includes not only a species label but also 312 attributes. In this project, 112 of those 312 attributes are filtered and used as concepts.

---

## Directory Structure
```
project_root/
├── checkpoints/            # Storage for trained model weights
├── configs/
│   ├── train_config.yaml   # YAML settings for model, optimizer, scheduler, and early stopping
│   ├── cub_train_config.yaml # CUB LoRA, calibration, and CooP evaluation settings
│   └── ...
├── data/                   # (Excluded from git) Storage for raw CUB data
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── base_dataset.py # Abstract base dataset class
│   ├── models/
│   │   ├── __init__.py
│   │   └── cbm_factory.py  # UniversalFlexibleCBM layout builder
│   ├── tti/
│   │   ├── common.py       # Shared TTI metric/logit helpers
│   │   └── coop.py         # CooP scoring and validation fitting
│   └── utils/
│       ├── __init__.py
│       ├── concept_bias.py # Calibration split and concept-bias training
│       └── metrics.py      # Utilities for computing accuracy and evaluation metrics
├── app.py                  # Gradio checkpoint explorer web application
├── download_CUB-200-2011.py # CUB download, verification, and extraction script
├── eval_cbm.py             # Evaluation and TTI benchmark entry point
├── generate_concept_config.py # Metadata-based concept setting extractor
├── main.py                 # Integrated training and evaluation entry point script
├── requirements.txt        # Dependency file
└── README_KR.md
```

---

## Environment Setup

This project uses `uv` for dependency management and builds.

```bash
# Synchronize dependency packages and install the virtual environment
uv sync
```

*(If you are not using `uv`, run `pip install -r requirements.txt` instead.)*

The Python version and major library versions are included in [pyproject.toml](pyproject.toml).

```text
requires-python = ">=3.12"
dependencies = [
    "gradio>=6.15.2",
    "huggingface-hub>=1.17.0",
    "matplotlib>=3.10.9",
    "numpy>=2.4.6",
    "open-clip-torch>=3.3.0",
    "pandas>=3.0.3",
    "pyyaml>=6.0.3",
    "timm>=1.0.27",
    "torch>=2.12.0",
    "torchvision>=0.27.0",
    "tqdm>=4.67.3",
    "transformers>=5.9.0",
    "wandb>=0.27.0",
]
```

---

## Training and Evaluation

Run the training pipeline through the `main.py` entry point. Model, dataset, concept, and training settings are read from the YAML file passed through `--config_path`.

### 1) Download CUB-200-2011 and Prepare Training

Download and verify the CUB archive, then generate the project concept config:

```bash
# Download from Kaggle; manual extraction is required
curl -L -o data/cub2002011.zip \
  https://www.kaggle.com/api/v1/datasets/download/wenewone/cub2002011

# Or use the script (Caltech server)
uv run python download_CUB-200-2011.py --data-dir data

# Generate concept indices
uv run python scratch/convert_cub_attributes.py
```

### 2) Basic CBM Training (Using Config Defaults)

Train the CBM with the default settings in `configs/cub_train_config.yaml`.
The following command saves outputs under `checkpoints/` and, by default, also runs evaluation.

```bash
uv run python main.py --config_path configs/cub_train_config.yaml --use_wandb false
```

The evaluation results (logs) should look similar to [eval_example.txt](eval_example.txt).

### 3) Evaluation and TTI Benchmark

You can manually run standard CBM evaluation and TTI benchmarks with `eval_cbm.py`.

```bash
uv run python eval_cbm.py \
    --checkpoint PATH_TO_CHECKPOINT

# Save evaluation results to a separate file
TQDM_DISABLE=1 uv run bash -c "python eval_cbm.py --checkpoint PATH_TO_CHECKPOINT 2>&1 | tee eval.txt"
```

## Reproducibility

After downloading the training dataset, you can reproduce the results by running training and evaluation.
On an RTX 4090 environment, training took about 30 minutes and evaluation took about 1 minute.

## AI Usage

The CODEX coding agent was used to write training and evaluation code, create the dataset download script, and translate documentation.

# References

- Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. **Concept Bottleneck Models**. *Proceedings of the 37th International Conference on Machine Learning (ICML)*, PMLR 119:5338-5348, 2020. [Link](https://proceedings.mlr.press/v119/koh20a.html)
- Kushal Chauhan, Rishabh Tiwari, Jan Freyberg, Pradeep Shenoy, and Krishnamurthy Dvijotham. **Interactive Concept Bottleneck Models**. *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(5):5948-5955, 2023. DOI: [10.1609/aaai.v37i5.25736](https://doi.org/10.1609/aaai.v37i5.25736). [Link](https://ojs.aaai.org/index.php/AAAI/article/view/25736)
- Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. **The Caltech-UCSD Birds-200-2011 Dataset**. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. [Link](https://authors.library.caltech.edu/records/cvm3y-5hh21)
