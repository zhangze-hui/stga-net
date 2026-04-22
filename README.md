# STGA-Net Training Guide

## Directory Structure

- `train.py`: Main training entry point
- `com_verify_cached_dataset.py`: Cached dataset reader
- `STGA-Net.py`: STGA-Net and related model definitions
- `tgcn.py`, `graph.py`: Graph convolution base modules
- `explainability_plot_utils.py`: Grad-CAM/Attention explainability plotting utilities
- `requirements-vscode-cu121.txt`: Dependency list (CUDA 12.1 environment)
- `requirements.txt`: Standard dependency entry file

## Preparation Before Running

1. Set up a Python virtual environment (Python 3.11 recommended)
2. Install dependencies
3. Prepare the cached dataset directory (`.pt` files)

> **Note**: The `CACHED_DATASET_DIR` in the main script is currently an absolute path. Please modify it according to your machine's path.

## Quick Run (PowerShell)

```powershell
python -m pip install -r requirements.txt
python ".\train.py"