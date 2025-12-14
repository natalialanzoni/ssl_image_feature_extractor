"""
Create submission using ResNet MoCo v2 encoder
==============================================

This script loads a ResNet-50 encoder trained with MoCo v2 style contrastive
loss (resnet_moco_addbirdsv2.py) and extracts features for the Kaggle
evaluation pipeline.

Supports both k-NN and linear probe evaluation.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import argparse
import platform
import copy


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def build_resnet50_backbone(model_name = 'resnet50'):
    if model_name == 'resnet50':
        resnet = models.resnet50(weights=None)
    elif model_name == 'resnet101':
        resnet = models.resnet101(weights=None)
    elif model_name == 'resnet152':
        resnet = models.resnet152(weights=None)
    elif model_name == 'convnext_base':
        resnet = models.convnext_base(weights=None)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    modules = list(resnet.children())[:-1]
    backbone = torch.nn.Sequential(*modules)
    if model_name == 'convnext_base':
        feat_dim = 1024
    else:
        feat_dim = resnet.fc.in_features
    return backbone, feat_dim


class ProjectionHead(torch.nn.Module):
    """
    MoCo v2 style projection head.
    Supports both 2-layer (standard MoCo v2) and 3-layer architectures.
    """
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=128, use_3layer=False):
        super().__init__()
        if use_3layer:
            # 3-layer projection head with BatchNorm
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, out_dim)
            )
        else:
            # 2-layer (MoCo v2 standard)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, out_dim)
            )

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x, dim=1)
        return x


def load_moco_checkpoint(checkpoint_path, backbone, projector, device, use_momentum=True):
    """
    Load MoCo checkpoint. By default uses momentum encoder (encoder_k) for evaluation,
    which is the standard practice in MoCo (similar to DINO's teacher model).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if use_momentum:
        # Use momentum encoder (standard for evaluation)
        backbone.load_state_dict(ckpt["encoder_k"], strict=True)
        projector.load_state_dict(ckpt["projector_k"], strict=True)
        print(f"✓ Loaded checkpoint from {checkpoint_path} (using momentum encoder)")
    else:
        # Fallback to query encoder if needed
        backbone.load_state_dict(ckpt["encoder_q"], strict=True)
        projector.load_state_dict(ckpt["projector_q"], strict=True)
        print(f"✓ Loaded checkpoint from {checkpoint_path} (using query encoder)")


class FeatureExtractor:
    def __init__(self, checkpoint_path, device="cuda", use_projected=True, 
                 feature_dim=128, use_3layer_proj=False, use_momentum=True, model_name='resnet50'):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_projected = use_projected

        self.backbone, feat_dim = build_resnet50_backbone(model_name=model_name)
        self.projector = ProjectionHead(
            in_dim=feat_dim, 
            hidden_dim=feat_dim, 
            out_dim=feature_dim,
            use_3layer=use_3layer_proj
        )

        self.backbone.to(self.device)
        self.projector.to(self.device)

        load_moco_checkpoint(checkpoint_path, self.backbone, self.projector, self.device, use_momentum=use_momentum)

        self.backbone.eval()
        self.projector.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize(96, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _encode(self, img_tensor):
        # Use inference_mode for faster inference (faster than no_grad)
        with torch.inference_mode():
            feat = self.backbone(img_tensor)
            # Handle different output shapes
            if feat.ndim == 4:  # (B, C, H, W) - ConvNext might output this
                # Global average pooling
                feat = feat.mean(dim=(-2, -1))  # (B, C)
            elif feat.ndim > 2:
                feat = torch.flatten(feat, 1)  # Flatten spatial dimensions
            # If already (B, C), keep as is
            if self.use_projected:
                feat = self.projector(feat)
            else:
                feat = F.normalize(feat, dim=1)
        return feat

    def extract_features(self, image):
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        feat = self._encode(tensor)
        return feat.cpu().numpy()[0]

    def extract_batch_features(self, images):
        tensor = torch.stack([self.transform(img) for img in images]).to(self.device)
        feat = self._encode(tensor)
        return feat.cpu().numpy()


# ---------------------------------------------------------------------------
# Dataset + dataloader helpers (reuse existing structure)
# ---------------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, image_dir, image_list, labels=None, resolution=96):
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.resolution = resolution

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx]
        path = self.image_dir / name
        image = Image.open(path).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        if self.labels is not None:
            return image, self.labels[idx], name
        return image, name


def collate_fn(batch):
    if len(batch[0]) == 3:
        images = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        filenames = [b[2] for b in batch]
        return images, labels, filenames
    images = [b[0] for b in batch]
    filenames = [b[1] for b in batch]
    return images, filenames


def extract_features_from_dataloader(feature_extractor, dataloader, split_name):
    feats = []
    labels = []
    filenames = []
    print(f"\nExtracting {split_name} features...")
    for batch in tqdm(dataloader, desc=f"{split_name} features"):
        if len(batch) == 3:
            images, batch_labels, batch_files = batch
            labels.extend(batch_labels)
        else:
            images, batch_files = batch
        features = feature_extractor.extract_batch_features(images)
        feats.append(features)
        filenames.extend(batch_files)
    feats = np.concatenate(feats, axis=0)
    labels = labels if labels else None
    print(f"  {split_name}: {feats.shape}")
    return feats, labels, filenames


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def print_feature_stats(features, split_name, expected_dim):
    print(f"\n[Diagnostic] {split_name} feature stats")
    print(f"  Shape: {features.shape}")
    if features.shape[1] != expected_dim:
        print(f"  ⚠️ Expected dim {expected_dim}, got {features.shape[1]}")
    norms = np.linalg.norm(features, axis=1)
    print(f"  Norm mean {norms.mean():.4f}, std {norms.std():.4f}")
    var = features.var(axis=0)
    print(f"  Var per dim mean {var.mean():.6f}, min {var.min():.6f}, max {var.max():.6f}")


def run_linear_probe(train_features, train_labels, val_features, val_labels, device="cuda", max_samples=None, 
                     use_early_stopping=True, patience=150, min_epochs=300, weight_decay=1e-4, test_features=None):
    """
    Train a PyTorch linear probe on frozen features.
    
    Args:
        train_features: numpy array (N_train, feat_dim)
        train_labels: list or numpy array (N_train,)
        val_features: numpy array (N_val, feat_dim)
        val_labels: list or numpy array (N_val,)
        device: torch device
        max_samples: if not None, subsample training data (for speed)
        test_features: optional test features for prediction
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Convert to tensors
    train_features = torch.FloatTensor(train_features).to(device)
    train_labels = torch.LongTensor(np.array(train_labels)).to(device)
    val_features = torch.FloatTensor(val_features).to(device)
    val_labels = torch.LongTensor(np.array(val_labels)).to(device)
    
    # Subsample if needed
    if max_samples is not None and len(train_features) > max_samples:
        idx = torch.randperm(len(train_features))[:max_samples]
        train_features = train_features[idx]
        train_labels = train_labels[idx]
        print(f"  Subsampled train from {len(train_features) + len(idx) - max_samples} to {max_samples}")
    
    feat_dim = train_features.shape[1]
    num_classes = len(torch.unique(train_labels))
    
    # Create linear classifier
    linear = torch.nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=50, min_lr=1e-5
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop with early stopping
    batch_size = 128
    num_epochs = 1600
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    
    linear.train()
    for epoch in range(num_epochs):
        # Shuffle training data
        perm = torch.randperm(len(train_features))
        train_features_shuffled = train_features[perm]
        train_labels_shuffled = train_labels[perm]
        
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_features_shuffled), batch_size):
            batch_features = train_features_shuffled[i:i+batch_size]
            batch_labels = train_labels_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = linear(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Evaluate validation accuracy every epoch
        linear.eval()
        with torch.no_grad():
            val_outputs = linear(val_features)
            val_preds = val_outputs.argmax(dim=1)
            current_val_acc = (val_preds == val_labels).float().mean().item()
        
        # Check for improvement BEFORE switching back to train mode
        # This ensures we save the exact state that achieved best_val_acc
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            # Save best model state immediately while still in eval mode
            # Use deep copy to ensure it's not modified by subsequent operations
            best_model_state = copy.deepcopy(linear.state_dict())
        else:
            epochs_without_improvement += 1
        
        # Switch back to train mode for next epoch
        linear.train()
        
        # Update learning rate scheduler
        scheduler.step(current_val_acc)
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / num_batches
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{num_epochs}, avg loss: {avg_loss:.4f}, val acc: {current_val_acc:.4f} ({current_val_acc*100:.2f}%), lr: {current_lr:.6f}")
        
        # Early stopping (only after minimum epochs and if enabled)
        if use_early_stopping and epoch + 1 >= min_epochs and epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f} at epoch {best_epoch})")
            break
    
    # Always use best model for final evaluation and predictions
    # (best_model_state is always defined since we initialize best_val_acc = 0.0)
    linear.eval()  # Set to eval mode before loading
    linear.load_state_dict(best_model_state, strict=True)
    print(f"  Using best model from epoch {best_epoch} (val acc: {best_val_acc:.4f})")
    
    # Verify the loaded model matches the best validation accuracy
    with torch.no_grad():
        # Re-evaluate validation to verify model state
        val_outputs_check = linear(val_features)
        val_preds_check = val_outputs_check.argmax(dim=1)
        val_acc_check = (val_preds_check == val_labels).float().mean().item()
        
        # Train accuracy
        train_outputs = linear(train_features)
        train_preds = train_outputs.argmax(dim=1)
        train_acc = (train_preds == train_labels).float().mean().item()
        
        # Val accuracy (should match best_val_acc exactly)
        val_outputs = linear(val_features)
        val_preds = val_outputs.argmax(dim=1)
        val_acc = (val_preds == val_labels).float().mean().item()
    
    print(f"  Linear probe train acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Linear probe val acc:   {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # Verify we're using the best model
    if abs(val_acc - best_val_acc) > 0.0001:  # Use smaller tolerance for float comparison
        print(f"  ⚠️  Warning: Final val acc ({val_acc:.4f}) doesn't match best val acc ({best_val_acc:.4f})")
        print(f"     Difference: {abs(val_acc - best_val_acc):.6f}")
        print(f"     This might indicate an issue with model state restoration.")
    
    # Generate test predictions if test_features provided (using best model)
    test_predictions = None
    if test_features is not None:
        test_features_tensor = torch.FloatTensor(test_features).to(device)
        linear.eval()  # Ensure eval mode
        with torch.no_grad():
            test_outputs = linear(test_features_tensor)
            test_predictions = test_outputs.argmax(dim=1).cpu().numpy()
    
    return linear, train_acc, val_acc, test_predictions



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Create submission using ResNet MoCo features")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="submission_resnet_moco.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use: 'cuda', 'cpu', or 'mps' (for Apple Silicon)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--use_projected", action="store_true", default=True,
                       help="Use projected features (default) or pooled features")
    parser.add_argument("--no_projected", dest="use_projected", action="store_false",
                       help="Use pooled features instead of projected")
    parser.add_argument("--feature_dim", type=int, default=128,
                        help="Feature dimension (must match training, default: 128 for MoCo v2)")
    parser.add_argument("--use_3layer_proj", action="store_true",
                        help="Use 3-layer projection head (default: False, 2-layer for MoCo v2)")
    parser.add_argument("--use_momentum", action="store_true", default=True,
                        help="Use momentum encoder for evaluation (default: True, recommended)")
    parser.add_argument("--no_momentum", dest="use_momentum", action="store_false",
                        help="Use query encoder instead of momentum encoder")
    parser.add_argument("--run_linear_probe", action="store_true", default=True,
                        help="Run linear probe evaluation (default: True)")
    parser.add_argument("--no_linear_probe", dest="run_linear_probe", action="store_false",
                        help="Skip linear probe evaluation")
    parser.add_argument("--model_name", type=str, default='resnet50',
                        help="Model architecture: 'resnet50', 'resnet101', 'resnet152', or 'convnext_base'")
    parser.add_argument("--no_early_stopping", action="store_true",
                        help="Disable early stopping, train for full num_epochs")
    parser.add_argument("--early_stop_patience", type=int, default=350,
                        help="Patience for early stopping (default: 150)")
    parser.add_argument("--early_stop_min_epochs", type=int, default=300,
                        help="Minimum epochs before early stopping can trigger (default: 300)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for linear probe (default: 1e-4)")
    args = parser.parse_args()

    # Device selection: prioritize CUDA, then MPS (Apple Silicon), then CPU
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif args.device == "mps" or (args.device == "cuda" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        device = "mps"
        print("⚠️  Using MPS (Apple Silicon GPU) - may be slower than CUDA but faster than CPU")
    else:
        device = "cpu"
        if args.device == "cuda":
            print("⚠️  CUDA not available, using CPU (this will be slow)")
    print(f"Using device: {device}")
    print(f"Model architecture:")
    print(f"  Backbone: {args.model_name}")
    print(f"  Feature dimension: {args.feature_dim}")
    print(f"  Projection head: {'3-layer' if args.use_3layer_proj else '2-layer'}")
    print(f"  Encoder: {'momentum (encoder_k)' if args.use_momentum else 'query (encoder_q)'}")
    print(f"  Using {'projected' if args.use_projected else 'pooled'} features")

    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train_labels.csv")
    val_df = pd.read_csv(data_dir / "val_labels.csv")
    test_df = pd.read_csv(data_dir / "test_images.csv")

    train_dataset = ImageDataset(
        data_dir / "train",
        train_df["filename"].tolist(),
        train_df["class_id"].tolist(),
        resolution=args.resolution
    )
    val_dataset = ImageDataset(
        data_dir / "val",
        val_df["filename"].tolist(),
        val_df["class_id"].tolist(),
        resolution=args.resolution
    )
    test_dataset = ImageDataset(
        data_dir / "test",
        test_df["filename"].tolist(),
        labels=None,
        resolution=args.resolution
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    feature_extractor = FeatureExtractor(
        checkpoint_path=args.checkpoint,
        device=device,
        use_projected=args.use_projected,
        feature_dim=args.feature_dim,
        use_3layer_proj=args.use_3layer_proj,
        use_momentum=args.use_momentum,
        model_name=args.model_name
    )

    train_features, train_labels, _ = extract_features_from_dataloader(
        feature_extractor, train_loader, "train"
    )
    val_features, val_labels, _ = extract_features_from_dataloader(
        feature_extractor, val_loader, "val"
    )
    test_features, _, test_filenames = extract_features_from_dataloader(
        feature_extractor, test_loader, "test"
    )

    # Line 451 - fix expected_dim calculation
    if args.model_name == 'convnext_base':
        expected_dim = args.feature_dim if args.use_projected else 1024
    else:
        expected_dim = args.feature_dim if args.use_projected else 2048

    print_feature_stats(train_features, "train", expected_dim)
    print_feature_stats(val_features, "val", expected_dim)

    # Always run linear probe for evaluation
    print("\n" + "="*60)
    print("Linear Probe Evaluation")
    print("="*60)
    linear_model, train_acc, val_acc, test_predictions = run_linear_probe(
        train_features, train_labels, val_features, val_labels,
        device=device,
        use_early_stopping=not args.no_early_stopping,
        patience=args.early_stop_patience,
        min_epochs=args.early_stop_min_epochs,
        weight_decay=args.weight_decay,
        test_features=test_features
    )

    # k-NN evaluation (for comparison)
    print("\n" + "="*60)
    print("k-NN Evaluation")
    print("="*60)
    print(f"Training k-NN classifier (k={args.k})...")
    classifier = KNeighborsClassifier(
        n_neighbors=args.k,
        metric="cosine",
        weights="distance",
        n_jobs=-1
    )
    classifier.fit(train_features, train_labels)
    train_acc_knn = classifier.score(train_features, train_labels)
    val_acc_knn = classifier.score(val_features, val_labels)
    print(f"k-NN train accuracy: {train_acc_knn:.4f} ({train_acc_knn*100:.2f}%)")
    print(f"k-NN val accuracy:   {val_acc_knn:.4f} ({val_acc_knn*100:.2f}%)")

    print("\n" + "="*60)
    print("Generating Submission (using Linear Probe)")
    print("="*60)
    
    # Use linear probe predictions (best model)
    if test_predictions is not None:
        predictions = test_predictions
    else:
        # Fallback: predict with linear model if test_predictions not returned
        test_features_tensor = torch.FloatTensor(test_features).to(device)
        linear_model.eval()
        with torch.no_grad():
            test_outputs = linear_model(test_features_tensor)
            predictions = test_outputs.argmax(dim=1).cpu().numpy()
    
    # Create submission dataframe (same format as k-NN example)
    submission = pd.DataFrame({
        "id": test_filenames,
        "class_id": predictions
    })
    
    submission.to_csv(args.output, index=False)
    print(f"✓ Saved submission to {args.output} (using Linear Probe)")
    print(f"  Linear probe val acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Total predictions: {len(submission)}")
    print(f"  Class distribution:")
    print(submission["class_id"].value_counts().head(10))


if __name__ == "__main__":
    main()


