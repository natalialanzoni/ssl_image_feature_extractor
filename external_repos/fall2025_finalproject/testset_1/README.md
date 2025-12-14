# CUB-200 SSL Competition - Getting Started

## Quick Start

### Step 1: Prepare Dataset
```bash
python prepare_cub200_for_kaggle.py --download_dir ./raw_data --output_dir ./data
```

This will:
- Download CUB-200-2011 dataset (~1.1 GB)
- Create train/val/test splits
- Generate CSV files with labels

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

