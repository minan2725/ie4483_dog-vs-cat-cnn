"""
Shared preprocessing pipeline for this project.

This file handles image loading, resizing, normalization, data augmentation,
balanced dataset sampling, random seeds, and DataLoader creation. 
Training augmentation can be disabled for experiments.
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Dataset folder (contains train/ val/ test/), located next to this file.
DATA_DIR = Path(__file__).resolve().parent / "datasets"

SEED = 42
IMG_SIZE = 224
# Statistics of ImageNet. Required by pretrained models (ResNet, VGG, ConvNeXt...).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def rgb_loader(path):
    """Open an image and force 3 channels (some files are grayscale / RGBA)."""
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def get_transforms(img_size=IMG_SIZE, augment=True):
    """Return (train_tf, eval_tf).

    augment=False makes the training transform deterministic (same as eval),
    which is used for the "no augmentation" experiment.
    """
    resize_to = round(img_size * 256 / 224)  # 256 when img_size = 224
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    # Validation / test: NO random augmentation, only deterministic resizing.
    eval_tf = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = eval_tf
    return train_tf, eval_tf


def class_balanced_indices(dataset, per_class, seed=SEED):
    """Deterministically pick `per_class` images of each class (None/0 = all)."""
    if not per_class:
        return list(range(len(dataset)))
    rng = random.Random(seed)
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)
    chosen = []
    for label in sorted(by_class):
        idxs = sorted(by_class[label])
        rng.shuffle(idxs)
        chosen += idxs[:per_class]
    return sorted(chosen)


def get_loaders(train_per_class=5000, val_per_class=None,
                batch_size=32, num_workers=4, data_dir=DATA_DIR,
                img_size=IMG_SIZE, augment=True):
    data_dir = Path(data_dir)
    train_tf, eval_tf = get_transforms(img_size, augment)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf, loader=rgb_loader)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf, loader=rgb_loader)
    # Label convention required by the brief: 0 = cat, 1 = dog
    assert train_ds.class_to_idx == {"cat": 0, "dog": 1}, train_ds.class_to_idx

    # The SAME images are chosen regardless of the training seed, so runs are comparable.
    train_sub = Subset(train_ds, class_balanced_indices(train_ds, train_per_class))
    val_sub = Subset(val_ds, class_balanced_indices(val_ds, val_per_class))

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            persistent_workers=num_workers > 0)
    return train_loader, val_loader


class TestImages(Dataset):
    """Flat folder of unlabeled test images (no dog/cat subfolders)."""

    def __init__(self, folder, img_size=IMG_SIZE):
        exts = {".jpg", ".jpeg", ".png"}
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)
        _, self.tf = get_transforms(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        return self.tf(rgb_loader(p)), p.name