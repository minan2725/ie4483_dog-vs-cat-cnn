"""
Shared preprocessing pipeline for this project.

This file handles image loading, resizing, normalization, data augmentation,
duplicate removal, random seeds, and DataLoader creation.
Training augmentation can be disabled for experiments.

By default, training uses all images in datasets/train except those that also
appear in val/ or test/ (listed in duplicate_pairs.csv). Validation uses all
images in datasets/val. Optional balanced limits support smoke tests.
"""
import csv
import os
import random
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Dataset folder (contains train/ val/ test/), located next to this file.
DATA_DIR = Path(__file__).resolve().parent / "datasets"
# Pairs of identical images found with a perceptual hash (train vs val, train vs test).
DUPLICATES_CSV = Path(__file__).resolve().parent / "duplicate_pairs.csv"
CACHE_DIR = Path(__file__).resolve().parent / "cached_256"

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


def load_duplicate_train_files(csv_path=DUPLICATES_CSV):
    """Paths (relative to train/, e.g. 'cat/cat.3908.jpg') of the train images listed in the csv."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Keep it next to preprocessing.py.")
    dups = set()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for key in ("path_1", "path_2"):
                parts = PurePosixPath(row[key].strip().replace("\\", "/")).parts
                if parts and parts[0] == "train":  # only the train copy is removed
                    dups.add("/".join(parts[1:]))
    return dups


def clean_train_indices(train_ds, train_root, csv_path=DUPLICATES_CSV):
    """Indices of the train images that are NOT duplicates of a val/test image."""
    dups = load_duplicate_train_files(csv_path)
    keep, removed = [], set()
    for i, (path, _) in enumerate(train_ds.samples):
        rel = os.path.relpath(path, train_root).replace(os.sep, "/")
        if rel in dups:
            removed.add(rel)
        else:
            keep.append(i)
    missing = dups - removed
    if missing:  # fail loudly: a typo or a different dataset copy would silently keep the leak
        raise RuntimeError(f"{len(missing)} images listed in {Path(csv_path).name} were not found "
                           f"in train/ (for example {sorted(missing)[:3]}).")
    print(f"Removed {len(removed)} train images that are duplicates of val/test images.")
    return keep


def class_balanced_indices(dataset, eligible_indices, per_class, seed=SEED):
    """Pick a fixed, balanced subset from eligible ImageFolder indices."""
    if not per_class:
        return eligible_indices
    rng = random.Random(seed)
    by_class = {}
    for i in eligible_indices:
        by_class.setdefault(dataset.samples[i][1], []).append(i)
    chosen = []
    for label, indices in sorted(by_class.items()):
        if len(indices) < per_class:
            raise ValueError(f"Class {label} has {len(indices)} images, fewer than {per_class} requested")
        chosen.extend(rng.sample(indices, per_class))
    return sorted(chosen)



def get_loaders(
    train_per_class=None, val_per_class=None, batch_size=32,
    num_workers=4, data_dir=DATA_DIR, img_size=IMG_SIZE,
    augment=True, seed=SEED, cache_dir=CACHE_DIR,
):
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)

    # Use cache only if all required split directories exist.
    # Otherwise, fall back to the original dataset.
    if all(
        (cache_dir / split).is_dir()
        for split in ("train", "val", "test")
    ):
        active_dir = cache_dir
        print(f"Using resized image cache: {active_dir}")
    else:
        active_dir = data_dir
        print(f"Cache not found. Using original images: {active_dir}")

    train_tf, eval_tf = get_transforms(img_size, augment)

    train_ds = datasets.ImageFolder(active_dir / "train", transform=train_tf, loader=rgb_loader)
    val_ds = datasets.ImageFolder(active_dir / "val", transform=eval_tf, loader=rgb_loader)

    # Label convention required by the brief: cat=0, dog=1.
    assert train_ds.class_to_idx == {"cat": 0, "dog": 1}, (
        train_ds.class_to_idx
    )
    assert val_ds.class_to_idx == {"cat": 0, "dog": 1}, (
        val_ds.class_to_idx
    )

    # Duplicate paths remain valid because the cache preserves filenames.
    train_indices = clean_train_indices(train_ds, active_dir / "train")
    train_indices = class_balanced_indices(train_ds, train_indices, train_per_class)
    train_sub = Subset(train_ds, train_indices)

    val_data = (Subset(val_ds, class_balanced_indices(
        val_ds, list(range(len(val_ds))), val_per_class)) if val_per_class else val_ds)

    # The seed controls the shuffling order and the random augmentation of each run.
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            persistent_workers=num_workers > 0)
    return train_loader, val_loader


class TestImages(Dataset):
    """Flat folder of unlabeled test images (no dog/cat subfolders)."""

    def __init__(self, folder, img_size=IMG_SIZE):
        folder = Path(folder)

        # If the caller passes datasets/test, transparently use its cache.
        if (
            folder.resolve() == (DATA_DIR / "test").resolve()
            and (CACHE_DIR / "test").is_dir()
        ):
            folder = CACHE_DIR / "test"

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        self.tf = get_transforms(img_size)[1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        return self.tf(rgb_loader(p)), p.name
