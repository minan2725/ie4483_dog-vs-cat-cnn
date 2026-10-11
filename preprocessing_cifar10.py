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
DATA_DIR = Path(__file__).resolve().parent / "cifar-10"

SEED = 42
IMG_SIZE = 32
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


def get_transforms(augment=True):
    """Return (train_tf, eval_tf).

    augment=False makes the training transform deterministic (same as eval),
    which is used for the "no augmentation" experiment.
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    # Validation / test: NO random augmentation, only deterministic resizing.
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = eval_tf
    return train_tf, eval_tf


def class_balanced_indices(dataset, per_class=None, imbalance=None, seed=SEED):
    """Select images per class using either a fixed count or class-specific ratios."""
    if imbalance is not None:
        if len(imbalance) != 10 or any(not 0 < x <= 1.0 for x in imbalance):
            raise ValueError("imbalance must contain 10 floats in (0, 1.0].")

    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    chosen = []

    for label in sorted(by_class):
        idxs = sorted(by_class[label])
        rng.shuffle(idxs)

        if imbalance is not None:
            n = max(1, int(len(idxs) * imbalance[label]))
        elif per_class:
            n = min(per_class, len(idxs))
        else:
            n = len(idxs)

        chosen.extend(idxs[:n])

    return sorted(chosen)


def get_loaders(train_per_class, val_per_class=None, imbalance=None,
                batch_size=32, num_workers=4, data_dir=DATA_DIR, augment=True):
    data_dir = Path(data_dir)
    train_tf, eval_tf = get_transforms(augment)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf, loader=rgb_loader)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf, loader=rgb_loader)

    expected_classes = {
        "airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
        "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9
    }
    assert train_ds.class_to_idx == expected_classes, train_ds.class_to_idx
    assert val_ds.class_to_idx == expected_classes, val_ds.class_to_idx

    train_sub = Subset(train_ds, class_balanced_indices(
        train_ds, train_per_class, imbalance
    ))
    val_sub = Subset(val_ds, class_balanced_indices(
        val_ds, val_per_class, imbalance
    ))

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=g, persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader


class TestImages(Dataset):
    def __init__(self, folder, imbalance=None, seed=SEED):
        exts = {".jpg", ".jpeg", ".png"}
        paths = sorted(p for p in Path(folder).rglob("*")
                       if p.is_file() and p.suffix.lower() in exts)

        self.class_names = sorted(p.name for p in Path(folder).iterdir() if p.is_dir())
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        assert self.class_to_idx == {
            "airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
            "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9
        }, self.class_to_idx

        if imbalance is not None and (
            len(imbalance) != 10 or any(not 0 < x <= 1.0 for x in imbalance)
        ):
            raise ValueError("imbalance must contain 10 values in (0, 1.0].")

        by_class = {i: [] for i in range(10)}
        for p in paths:
            by_class[self.class_to_idx[p.parent.name]].append(p)

        rng = random.Random(seed)
        self.paths, self.labels = [], []

        for label, class_paths in by_class.items():
            rng.shuffle(class_paths)
            n = len(class_paths) if imbalance is None else max(
                1, int(len(class_paths) * imbalance[label])
            )
            selected = sorted(class_paths[:n])
            self.paths.extend(selected)
            self.labels.extend([label] * len(selected))

        _, self.tf = get_transforms()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        return self.tf(rgb_loader(p)), self.labels[i], p.name