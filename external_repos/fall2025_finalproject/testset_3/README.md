# SUN397 SSL Competition - Getting Started

## Quick Start

### Step 1: Prepare Dataset
```bash
pip install datasets

python prepare_sun397_for_kaggle.py --download_dir ./raw_data --output_dir ./data
```

This will:
- Download SUN397 dataset from HuggingFace (~37 GB original, resized to 96x96)
- Create train/val/test splits
- 397 scene categories (abbey, airport, bedroom, beach, forest, etc.)
- Generate CSV files with labels

### Step 2: Create Submission
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
├── class_mapping.csv   # Scene class ID to name mapping
└── sample_submission.csv
```

---

## About SUN397

- **Task**: Scene recognition (not objects!)
- **Classes**: 397 scene categories
- **Examples**: abbey, airport_terminal, bedroom, beach, forest, kitchen, mountain, office, restaurant, street, etc.
- **Challenge**: Understanding context and environment, not just objects

Good luck! 🚀

