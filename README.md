# Self-Supervised Learning Model (MoCo v2)

This repository contains code for training and evaluating self-supervised learning models using MoCo v2 (Momentum Contrast) framework.

## Project Structure

```
ssl_model/
├── run_model.py          # Main training script
├── eval_linear.py        # Evaluation script (linear probe & k-NN)
├── train_model.sh        # Training script wrapper
├── config.json           # Configuration file for training parameters
├── requirements.txt      # Python dependencies
└── external_repos/       # External repository code for dataset preparation
```

## Features

- **MoCo v2 Training**: Self-supervised learning with momentum contrast
- **Model Architectures**: Supports ResNet-50/101/152 and ConvNeXt-Base
- **Evaluation Methods**: 
  - Linear probe with early stopping
  - k-NN classifier
- **Device Support**: CUDA, MPS (Apple Silicon), and CPU
- **Config Management**: Centralized configuration via JSON file

## Installation

1. Create a virtual environment (Python 3.10 recommended for `learn2learn`):
```bash
python3.10 -m venv venv_py310
source venv_py310/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: For dataset preparation with `learn2learn`, Python 3.10 is required to avoid compilation errors.

## Usage

### Training

1. Configure parameters in `config.json` or override via command line:
```bash
bash train_model.sh
```

Or directly:
```bash
python run_model.py --config config.json
```

### Evaluation

Evaluate a trained model and generate submission file:
```bash
python eval_linear.py \
    --checkpoint path/to/checkpoint.pt \
    --data_dir path/to/kaggle_data \
    --output submission.csv \
    --batch_size 128
```

The evaluation script will:
- Extract features from train/val/test sets
- Train a linear probe with early stopping (saves best model)
- Evaluate with k-NN for comparison
- Generate submission CSV using the best linear probe model

## Configuration

The `config.json` file contains all training parameters organized into sections:
- `training`: epochs, batch size, learning rate, optimizer settings
- `model`: architecture, feature dimensions, MoCo parameters
- `data`: dataset paths and settings
- `augmentation`: data augmentation parameters
- `training_config`: output directories, checkpoint paths

## Key Features of Evaluation Script

- **Best Model Selection**: Automatically saves and uses the linear probe model with highest validation accuracy
- **Early Stopping**: Prevents overfitting with configurable patience
- **Device Optimization**: Automatically selects best available device (CUDA > MPS > CPU)
- **macOS Compatibility**: Automatically sets `num_workers=0` on macOS to prevent multiprocessing issues

## Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- scikit-learn
- pandas
- numpy
- tqdm

## Notes

- Large data directories and model checkpoints are excluded from git (see `.gitignore`)
- Virtual environments should not be committed
- Submission CSV files are generated outputs and excluded from version control

