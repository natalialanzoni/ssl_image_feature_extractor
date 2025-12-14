"""
ResNet MoCo v2 Style Contrastive Self-Supervised Training
=========================================================

This script trains a ResNet-50 encoder with MoCo v2 improvements:
- BatchNorm shuffling (prevents information leakage)
- Learning rate schedule (warmup + cosine annealing)
- MoCo v2 augmentation style
- Birdsnap dataset integration
"""

import os
import math
import time
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plot

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFile
import numpy as np
from datasets import load_dataset

# Allow loading truncated images (some Birdsnap images are corrupted)
ImageFile.LOAD_TRUNCATED_IMAGES = True
from huggingface_hub import snapshot_download
import zipfile
import glob
from tqdm import tqdm
import torch.multiprocessing as mp

COLOR_JITTER = 0.4
RANDOM_GRAYSCALE_PROB = 0.2

# ---------------------------------------------------------------------------
# Utility: set deterministic behavior for reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data pipeline: HuggingFace dataset returning two augmented views
# ---------------------------------------------------------------------------
class TwoCropTransform:
    """Create two augmented views of the same image."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def build_moco_v2_augmentations(image_size=96):
    """
    MoCo v2's augmentation style (from official code).
    Similar to SimCLR but with RandomApply for ColorJitter and GaussianBlur.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomApply(
            [transforms.ColorJitter(COLOR_JITTER, COLOR_JITTER, COLOR_JITTER, 0.1)],
            p=0.8,  # not strengthened
        ),
        transforms.RandomGrayscale(p=RANDOM_GRAYSCALE_PROB),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
            p=0.5,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    return transforms.Compose(augmentation)


class ImageFolderTwoCropDataset(Dataset):
    """
    Dataset that reads images from a directory (unzipped) and returns two crops.
    """

    def __init__(self, image_paths, image_size=96):
        self.image_paths = image_paths
        self.transform = TwoCropTransform(build_moco_v2_augmentations(image_size))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        # PIL images are lazy-loaded, so we need to ensure data is loaded
        # before closing. The convert() call loads the data.
        im1, im2 = self.transform(img)
        # Note: PIL images auto-close when garbage collected, but explicit close
        # can help with "too many open files" errors
        img.close()
        return im1, im2


class HFTwoCropDataset(Dataset):
    """
    Wrap a HuggingFace dataset (providing PIL images) with two-crop augmentations.
    """

    def __init__(self, hf_dataset, image_size=96):
        self.dataset = hf_dataset
        self.transform = TwoCropTransform(build_moco_v2_augmentations(image_size))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Model: ResNet-50 backbone with projection head (query + key encoders)
# ---------------------------------------------------------------------------
def build_model_backbone(which_model='resnet50'):
    if which_model == 'resnet101':
        print(f'using resnet101')
        model = models.resnet101(weights=None)
    elif which_model == 'resnet152':
        print(f'using resnet152')
        model = models.resnet152(weights=None)
    elif which_model == 'resnet50':
        model = models.resnet50(weights=None)
    elif which_model == 'convnext':
        print(f'using convnext')
        model = models.convnext_base(weights=None)
    else:
        assert 0, f'unknown model type requested {which_model}'

    in_features = 1024 if which_model == 'convnext' else model.fc.in_features

    # Remove final FC layer
    modules = list(model.children())[:-1]  # keep up to global avg pool
    backbone = nn.Sequential(*modules)
    # was model.fc.in_features  # 2048
    return backbone, in_features


class ProjectionHead(nn.Module):
    """
    MoCo v2 style 2-layer projection head (standard MoCo v2 architecture).
    """
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=128):
        super().__init__()
        # 2-layer projection head (MoCo v2 standard)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x, dim=1)
        return x


class ResNetMoCo(nn.Module):
    """
    MoCo v2 style model with BatchNorm shuffling to prevent information leakage.
    """

    def __init__(self, feature_dim=128, queue_size=65536, momentum=0.99, temperature=0.2, which_model='resnet50'):
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature
        self.feature_dim = feature_dim
        self.queue_size = queue_size

        # Query encoder
        self.encoder_q, feat_dim = build_model_backbone(which_model)
        self.projector_q = ProjectionHead(feat_dim, feat_dim, feature_dim)

        # Key encoder
        self.encoder_k, _ = build_model_backbone(which_model)
        self.projector_k = ProjectionHead(feat_dim, feat_dim, feature_dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the queue
        self.register_buffer("queue", torch.randn(feature_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update the queue with new keys.
        """
        batch_size = keys.shape[0]
        queue_size = self.queue.shape[1]

        ptr = int(self.queue_ptr)
        space = queue_size - ptr

        if batch_size <= space:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % queue_size
        else:
            self.queue[:, ptr:] = keys[:space].T
            remaining = batch_size - space
            self.queue[:, :remaining] = keys[space:space + remaining].T
            ptr = remaining

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        MoCo v2 forward with BatchNorm shuffling to prevent information leakage.
        
        Input:
            im_q: a batch of query images
            im_k: a batch of key images

        Output:
            logits, targets
        """
        batch_size = im_q.shape[0]
        
        # Compute query features
        q = self.encoder_q(im_q)
        q = torch.flatten(q, 1)
        q = self.projector_q(q)  # normalized

        # Compute key features with BatchNorm shuffling (MoCo v2 fix)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            # Shuffle for making use of BN
            # Generate random permutation indices
            idx_shuffle = torch.randperm(batch_size).to(im_k.device)
            idx_unshuffle = torch.argsort(idx_shuffle)
            
            # Shuffle the key batch before encoder forward pass
            im_k_shuffled = im_k[idx_shuffle]
            
            # Forward pass through shuffled key encoder
            k = self.encoder_k(im_k_shuffled)
            k = torch.flatten(k, 1)
            k = self.projector_k(k)  # normalized
            
            # Unshuffle to restore original order
            k = k[idx_unshuffle]

        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        self._dequeue_and_enqueue(k)

        return logits, labels


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def train_one_epoch(model, data_loader, optimizer, scaler, device, epoch, plot_fname='resnet.txt'):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    running_loss = 0.0
    avg_loss = 0

    if plot_fname is not None:
        fid = open(plot_fname, 'a')

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (im1, im2) in pbar:
        im1 = im1.to(device, non_blocking=True)
        im2 = im2.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits, labels = model(im1, im2)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.encoder_q.parameters()) + list(model.projector_q.parameters()),
            max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        avg_loss += loss.item()

        pbar.set_description('loss={:.3f}'.format(loss))
        if False and ((step + 1) % 100 == 0):
            avg_loss = running_loss / 100
            print("Epoch [{}] Loss: {:.4f}".format(epoch, avg_loss))
            running_loss = 0.0

    avg_loss /= step
    if plot_fname is not None:
        fid.write('{} {:.4f}\n'.format(epoch, avg_loss))
        fid.close()

    return avg_loss

def save_checkpoint(state, checkpoint_dir, epoch, best=False, latest=True):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if best:
        latest_path = checkpoint_dir / "best_resnet_moco_latest.pt"
        print("✓ Saving best to checkpoint {}".format(latest_path))
    else:
        ckpt_path = checkpoint_dir / "resnet_moco_epoch_{}.pt".format(epoch)
        latest_path = checkpoint_dir / "resnet_moco_latest.pt"
        if not latest:
            torch.save(state, ckpt_path)
            print("✓ Saved checkpoint to {}".format(ckpt_path))
        else:
            print("✓ Saved latest checkpoint to {}".format(latest_path))


    torch.save(state, latest_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def prepare_dataset(args):
    """
    Download (if needed) and unzip dataset, returning list of image paths.
    """
    is_colab = 'google.colab' in sys.modules or os.path.exists('/content/drive')
    if is_colab and not os.path.exists('/content/drive/MyDrive'):
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except ImportError:
            print("⚠️ Could not mount Google Drive in Colab.")

    default_base = '/content/drive/MyDrive/dataset_cache' if is_colab else './dataset_cache'

    zip_dir = args.zip_dir if args.zip_dir else os.path.join(default_base, 'zips')
    unzipped_dir = args.unzipped_dir if args.unzipped_dir else os.path.join(default_base, 'unzipped')

    if args.zip_dir:
        print("Using provided ZIP directory: {}".format(zip_dir))
    else:
        print("{} Using default ZIP cache at {}".format('🌐' if is_colab else '💻', zip_dir))

    if args.unzipped_dir:
        print("Using provided unzipped directory: {}".format(unzipped_dir))
    else:
        print("Unzipped images will be stored at {}".format(unzipped_dir))

    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(unzipped_dir, exist_ok=True)

    dataset_repo = args.dataset_name

    existing_zips = glob.glob(os.path.join(zip_dir, "*.zip"))
    if existing_zips:
        print("✓ Found {} ZIP files in {}".format(len(existing_zips), zip_dir))
    else:
        print("Downloading dataset {} to {} via snapshot_download...".format(dataset_repo, zip_dir))
        snapshot_download(
            repo_id=dataset_repo,
            local_dir=zip_dir,
            repo_type="dataset",
            ignore_patterns=["*.md", "*.txt"]
        )
        existing_zips = glob.glob(os.path.join(zip_dir, "*.zip"))

    def collect_images():
        paths = []
        # Collect from train/ folder (competition dataset)
        train_dir = os.path.join(unzipped_dir, 'train')
        if os.path.exists(train_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                paths.extend(Path(train_dir).rglob(ext))
        # Also collect from root of unzipped_dir (if images are directly there)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            paths.extend(Path(unzipped_dir).glob(ext))
        # Note: birds/ folder is handled separately in download_birdsnap_images
        return [p for p in set(paths) if p.exists() and p.is_file()]

    existing_images = collect_images()

    if existing_images and not args.reuse_unzipped:
        print("✓ Found {:,} unzipped images in {}".format(len(existing_images), unzipped_dir))
    elif args.reuse_unzipped:
        print("Reusing unzipped directory {}".format(unzipped_dir))
        if not existing_images:
            print("⚠️ No images found in provided unzipped directory.")
    else:
        print("Unzipping {} files to {}...".format(len(existing_zips), unzipped_dir))
        for zip_path in existing_zips:
            zip_name = os.path.basename(zip_path)
            print("  Unzipping {}...".format(zip_name))
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(unzipped_dir)
        existing_images = collect_images()
        print("✓ Unzipped {:,} images".format(len(existing_images)))

    if args.dataset_subset is not None:
        existing_images = sorted(existing_images)[:args.dataset_subset]
        print("Using subset of {} images".format(len(existing_images)))
    else:
        existing_images = sorted(existing_images)

    return existing_images, unzipped_dir


def download_birdsnap_images(unzipped_dir, args):
    """
    Download Birdsnap dataset from HuggingFace and save images to unzipped_dir/birds/
    """
    birds_dir = os.path.join(unzipped_dir, 'birds')
    os.makedirs(birds_dir, exist_ok=True)

    # Check if birds directory already has images
    existing_bird_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        existing_bird_images.extend(Path(birds_dir).rglob(ext))
    existing_bird_images = [p for p in existing_bird_images if p.exists() and p.is_file()]

    if existing_bird_images:
        print("✓ Found {:,} existing Birdsnap images in {}".format(len(existing_bird_images), birds_dir))
        return existing_bird_images

    print("Downloading Birdsnap dataset snapshot from HuggingFace (sasha/birdsnap)...")

    # Use a snapshot directory next to unzipped_dir to avoid filling home directory
    if args.hf_cache_dir:
        snapshot_dir = args.hf_cache_dir
    else:
        snapshot_dir = os.path.join(os.path.dirname(unzipped_dir), 'birdsnap_snapshot')
    os.makedirs(snapshot_dir, exist_ok=True)
    print("  Using snapshot directory: {}".format(snapshot_dir))

    try:
        # Download the entire dataset repo locally (images + metadata)
        snapshot_download(
            repo_id="sasha/birdsnap",
            repo_type="dataset",
            local_dir=snapshot_dir,
            ignore_patterns=["*.md", "*.txt"]
        )

        # Collect all image files from the snapshot and link/copy them into birds_dir
        saved_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            saved_paths.extend(Path(snapshot_dir).rglob(ext))

        bird_images = []
        for i, src in enumerate(tqdm(saved_paths, desc="Linking Birdsnap images")):
            try:
                # Use symlinks when possible to avoid duplicating data; fall back to copy
                dst = Path(birds_dir) / src.name
                if not dst.exists():
                    try:
                        dst.symlink_to(src.resolve())
                    except Exception:
                        # If symlinks are not allowed, copy the file
                        import shutil
                        shutil.copy2(src, dst)
                bird_images.append(dst)
            except Exception as e:
                if i < 10:
                    print("  Warning: Skipping image {}: {}".format(src, e))
                continue

        bird_images = [p for p in set(bird_images) if p.exists() and p.is_file()]
        print("✓ Prepared {:,} Birdsnap images under {}".format(len(bird_images), birds_dir))
        return bird_images

    except Exception as e:
        print("  Error downloading Birdsnap snapshot: {}".format(e))
        print("  Continuing without Birdsnap dataset...")
        return []


def build_hf_dataset(args):
    cache_dir = args.hf_cache_dir if args.hf_cache_dir else None
    print("Loading HuggingFace dataset {} (cache_dir={})".format(args.dataset_name, cache_dir))
    hf_dataset = load_dataset(args.dataset_name, split="train", cache_dir=cache_dir)
    if args.dataset_subset is not None:
        subset_size = min(args.dataset_subset, len(hf_dataset))
        hf_dataset = hf_dataset.select(range(subset_size))
        print("Using HF subset of size {}".format(len(hf_dataset)))
    else:
        print("HF dataset size: {}".format(len(hf_dataset)))
    return HFTwoCropDataset(hf_dataset, image_size=args.image_size)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ResNet MoCo v2 Contrastive Training")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--color_jitter", type=float, default=0.4, help="Amount of Color Jitter Augmentation")
    parser.add_argument("--random_grayscale_prob", type=float, default=0.2, help="Probability of Random Grayscale Augmentation")
    parser.add_argument("--lr", type=float, default=0.03, help="Base learning rate (for SGD: 0.03, for Adam: 0.001)")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of warmup epochs for learning rate")
    parser.add_argument("--scale_lr_by_batch", action="store_true", default=True,
                        help="Scale LR by batch_size/256 (recommended for SGD, optional for Adam)")
    parser.add_argument("--no_scale_lr_by_batch", dest="scale_lr_by_batch", action="store_false",
                        help="Don't scale LR by batch size")
    parser.add_argument("--resnet101", action="store_true", default=False, help="use Resnet101 instead of Resnet50")
    parser.add_argument("--resnet152", action="store_true", default=False, help="use Resnet152 instead of Resnet50")
    parser.add_argument("--convnext", action="store_true", default=False, help="use ConvNext instead of Resnet50")
    parser.add_argument("--name", type=str, default='RESNET_V2')
    parser.add_argument("--opt", type=str, default='SGD')
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--queue_size", type=int, default=65536)
    parser.add_argument("--feature_dim", type=int, default=128, help="Feature dimension (MoCo v2 standard: 128)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_resnet_moco_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_subset", type=int, default=None,
                        help="Optional subset size for faster experiments")
    parser.add_argument("--dataset_name", type=str, default="tsbpp/fall2025_deeplearning")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--zip_dir", type=str, default=None,
                        help="Optional existing ZIP directory to reuse")
    parser.add_argument("--unzipped_dir", type=str, default=None,
                        help="Optional existing unzipped directory containing images")
    parser.add_argument("--reuse_unzipped", action="store_true",
                        help="Skip unzip and reuse provided unzipped_dir")
    parser.add_argument("--use_hf_loader", action="store_true",
                        help="Use HuggingFace dataset loader (no zip scan)")
    parser.add_argument("--hf_cache_dir", type=str, default=None,
                        help="Cache directory for HuggingFace dataset")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., ./checkpoints_resnet_moco_v2/resnet_moco_latest.pt)")
    args = parser.parse_args()

    COLOR_JITTTER = args.color_jitter
    RANDOM_GRAYSCALE_PROB = args.random_grayscale_prob

    if args.convnext:
        which_model = 'convnext'
    elif args.resnet101:
        which_model = 'resnet101'
    elif args.resnet152:
        which_model = 'resnet152'
    else:
        which_model = 'resnet50'

    print(f'using model {which_model}')

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))

    if args.use_hf_loader:
        train_dataset = build_hf_dataset(args)
        unzipped_dir = None  # Not using unzipped directory
    else:
        print("Preparing dataset via snapshot download/unzip...")
        image_paths, unzipped_dir = prepare_dataset(args)
        train_dataset = ImageFolderTwoCropDataset(image_paths, image_size=args.image_size)
    
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

    # Download Birdsnap images and combine with competition dataset
    print("\n" + "="*60)
    print("Loading Birdsnap dataset")
    print("="*60)
    
    if unzipped_dir is None:
        print("  Skipping Birdsnap (using HuggingFace loader mode)")
    else:
        try:
            # Download Birdsnap images to unzipped_dir/birds/
            bird_image_paths = download_birdsnap_images(unzipped_dir, args)
            
            if bird_image_paths:
                # Create dataset from Birdsnap images
                birdsnap_dataset = ImageFolderTwoCropDataset(bird_image_paths, image_size=args.image_size)
                
                # Combine datasets
                print("\n" + "="*60)
                print("Combining datasets")
                print("="*60)
                
                competition_size = len(train_dataset)
                birdsnap_size = len(birdsnap_dataset)
                
                combined_dataset = torch.utils.data.ConcatDataset([train_dataset, birdsnap_dataset])
                train_dataset = combined_dataset
                
                print("  Competition dataset: {:,} images".format(competition_size))
                print("  Birdsnap dataset: {:,} images".format(birdsnap_size))
                print("  Combined total: {:,} images".format(len(train_dataset)))
            else:
                print("  No Birdsnap images found, continuing with competition dataset only...")
                print("  Total images: {:,}".format(len(train_dataset)))
                
        except Exception as e:
            print("  Warning: Failed to load Birdsnap dataset: {}".format(e))
            print("  Continuing with competition dataset only...")
            print("  Total images: {:,}".format(len(train_dataset)))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # MoCo v2 standard: 2-layer projection head, 128-dim features
    model = ResNetMoCo(
        feature_dim=args.feature_dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
        which_model=which_model
    ).to(device)
    nparms = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('ResNetMoco model created with {:.2f} M parameters'.format(nparms/(1024*1024)))

    # Scale learning rate by batch size (linear scaling rule for SGD)
    base_lr = args.lr
    if args.scale_lr_by_batch:
        scaled_lr = base_lr * (args.batch_size / 256.0)
        print("Base LR: {:.6f}, Scaled LR (batch_size={}): {:.6f}".format(base_lr, args.batch_size, scaled_lr))
    else:
        scaled_lr = base_lr
        print("Using base LR: {:.6f} (no batch size scaling)".format(scaled_lr))
    
    # Only optimize query encoder + projector
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(
            list(model.encoder_q.parameters()) + list(model.projector_q.parameters()),
            lr=scaled_lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        print("Using SGD optimizer (MoCo v2 standard)")
    else:
        optimizer = torch.optim.AdamW(
            list(model.encoder_q.parameters()) + list(model.projector_q.parameters()),
            lr=scaled_lr,
            weight_decay=1e-4
        )
        print("Using AdamW optimizer (note: MoCo v2 uses SGD)")
        if scaled_lr > 0.01:
            print("  ⚠️ Warning: LR {} may be too high for Adam. Consider --lr 0.001".format(scaled_lr))

    # Learning rate schedule: warmup + cosine annealing (MoCo v2 style)
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            # Linear warmup
            return epoch / args.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = torch.amp.GradScaler()

    # Resume from checkpoint if provided
    start_epoch = 1
    best_loss = 1e10
    best_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', 1e10)
            
            model.encoder_q.load_state_dict(checkpoint['encoder_q'])
            model.projector_q.load_state_dict(checkpoint['projector_q'])
            model.encoder_k.load_state_dict(checkpoint['encoder_k'])
            model.projector_k.load_state_dict(checkpoint['projector_k'])
            model.queue = checkpoint['queue'].to(device)
            model.queue_ptr = checkpoint['queue_ptr']
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            print("Resumed from epoch {}, best loss: {:.4f}".format(checkpoint['epoch'], best_loss))
            best_epoch = checkpoint['epoch']
        else:
            print("Warning: Checkpoint '{}' not found. Starting from scratch.".format(args.resume))
    
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, plot_fname=args.name + '.txt')
        scheduler.step()
        epoch_time = time.time() - start_time
        print("Epoch {} completed in {:.2f} min, Average Loss = {:.2f}, LR = {:.6f}, min Loss = {:.2f} at epoch {:3d}".format(
            epoch, epoch_time/60, loss, current_lr, best_loss, best_epoch))

        if loss < best_loss:
            print('********* new best loss {:.2f} ***********'.format(loss))
            best_loss = loss
            best_epoch = epoch

        # Save checkpoint
        state = {
            "epoch": epoch,
            "best_loss": best_loss,
            "encoder_q": model.encoder_q.state_dict(),
            "projector_q": model.projector_q.state_dict(),
            "encoder_k": model.encoder_k.state_dict(),
            "projector_k": model.projector_k.state_dict(),
            "queue": model.queue,
            "queue_ptr": model.queue_ptr,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args)
        }
        save_checkpoint(state, args.output_dir, epoch, best=False, latest=True)
        if (epoch % 25) == 0:
            save_checkpoint(state, args.output_dir, epoch, best=False, latest=False)
        if (loss <= best_loss):
            save_checkpoint(state, args.output_dir, epoch, best=True, latest=False)


if __name__ == "__main__":
    main()

