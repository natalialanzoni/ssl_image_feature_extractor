# Mini-ImageNet SSL Competition - Getting Started

## Quick Start

### Step 1: Prepare Dataset
```bash
python prepare_miniimagenet_for_kaggle.py --download_dir ./raw_data --output_dir ./data
```

This will:
- Download Mini-ImageNet dataset (~3 GB)
- Create train/val/test splits
- Generate CSV files with labels

**Note**: You'll need to download Mini-ImageNet pkl files first:
- Option 1: Install `learn2learn` and it will auto-download: `pip install learn2learn`
- Option 2: Download manually from https://github.com/yaoyao-liu/mini-imagenet-tools

### Step 2: Create Submission (Baseline)
```bash
python create_submission_knn.py --data_dir ./data --output submission.csv --resolution 96 --k 5
```

This example uses pretrained WebSSL + KNN. **For the competition, you must train your own model from scratch!** Also you don't have to follow this evaluation, you can tune KNN or Linear Probing. Please remember to freeze your trained encoder.

### Step 3: Upload to Kaggle
Upload `submission.csv` to the competition page.

---

## Dataset Structure After Step 1

```
data/
├── train/              # Training images (with labels)
├── val/                # Validation images (with labels)
├── test/               # Test images (NO labels)
├── train_labels.csv    
├── val_labels.csv      
├── test_images.csv     
└── sample_submission.csv
```

---

Good luck! 🚀

