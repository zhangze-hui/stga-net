# STGA-Net Training Guide

## Preparation Before Running

1. Set up a Python virtual environment (Python 3.11 recommended)
2. Install dependencies
3. Prepare the cached dataset directory (`.pt` files)

> **Note**: The `CACHED_DATASET_DIR` in the main script is currently an absolute path. Please modify it according to your machine's path.

## Quick Run (PowerShell)

# STGA-Net Training Guide

This repository contains a clean training pipeline for STGA-Net on UI-PRMD, including:

1. **Dataset preprocessing** (`cache_dataset.py`) from original segmented motion files to cached `.pt` tensors.
2. **Model training/evaluation** (`train.py`) with cross-validation and final hold-out testing.

## Directory Structure

- `cache_dataset.py`: dataset preprocessing and cache generation.
- `train.py`: main training entry point.
- `com_verify_cached_dataset.py`: cached dataset reader.
- `STGA_Net.py`: STGA-Net and related model definitions.
- `tgcn.py`, `graph.py`: graph convolution base modules.
- `explainability_plot_utils.py`: Grad-CAM/Attention explainability plotting utilities.
- `requirements-vscode-cu121.txt`: dependency list (CUDA 12.1 environment).
- `requirements.txt`: standard dependency entry file.

## Step 1: Environment Setup

1. Create a Python virtual environment (Python 3.11 recommended).
2. Install dependencies.

```powershell
python -m pip install -r requirements.txt
```

## Step 2: Dataset Preprocessing

Run preprocessing to convert raw segmented UI-PRMD files into cached tensors (`.pt`).

```powershell
python .\cache_dataset.py
```

Before running, please edit paths in `cache_dataset.py` (`__main__` section):

- `original_data_root_dirs["correct"]`
- `original_data_root_dirs["incorrect"]`
- `cached_dataset_dir`

The script will:

- smooth raw position/angle trajectories,
- reconstruct skeleton coordinates,
- perform temporal length normalization (default: `resample` to 200 frames),
- save each sample as a cached `.pt` file.

## Step 3: Training

After cache generation is complete, run training:

```powershell
python .\train.py
```

Before training, check `CACHED_DATASET_DIR` in `train.py` and point it to your cached dataset directory.

## Outputs

Training results are saved to:

- `stg-net_cross_val_experiment_results_YYYYMMDD_HHMMSS/`

including per-fold best checkpoints, confusion matrices, final hold-out results, and explainability outputs.
