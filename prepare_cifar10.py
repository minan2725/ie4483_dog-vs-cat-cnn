
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Directory containing data_batch_1, ..., test_batch and batches.meta.
SOURCE_DIR = Path("cifar-10-batches-py")
OUTPUT_DIR = Path("cifar-10")

VAL_RATIO = 0.2
SEED = 42


def unpickle(file):
    with open(file, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_batch(file):
    batch = unpickle(file)
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels


def save_images(images, labels, split, class_names):
    for i, (image, label) in enumerate(zip(images, labels)):
        class_name = class_names[int(label)]
        class_dir = OUTPUT_DIR / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        filename = class_dir / f"{split}_{i:05d}.png"
        Image.fromarray(image).save(filename)


def print_image_counts(class_names):
    splits = ("train", "val", "test")
    totals = {split: 0 for split in splits}

    print("\nImage counts by folder:")
    print(f"{'Class':<15}{'Train':>10}{'Val':>10}{'Test':>10}{'Total':>10}")
    print("-" * 55)

    for class_name in class_names:
        counts = []
        for split in splits:
            class_dir = OUTPUT_DIR / split / class_name
            count = sum(1 for p in class_dir.iterdir() if p.is_file()) if class_dir.exists() else 0
            counts.append(count)
            totals[split] += count

        print(f"{class_name:<15}{counts[0]:>10}{counts[1]:>10}{counts[2]:>10}{sum(counts):>10}")

    print("-" * 55)
    print(f"{'Total':<15}{totals['train']:>10}{totals['val']:>10}{totals['test']:>10}{sum(totals.values()):>10}")


def main():
    if not 0 < VAL_RATIO < 1:
        raise ValueError("VAL_RATIO must be between 0 and 1.")

    if not (SOURCE_DIR / "batches.meta").exists():
        raise FileNotFoundError(f"Metadata file not found: {SOURCE_DIR / 'batches.meta'}")

    # Read class names from metadata.
    meta = unpickle(SOURCE_DIR / "batches.meta")
    class_names = [
        name.decode("utf-8") if isinstance(name, bytes) else name
        for name in meta[b"label_names"]
    ]

    # Load the five training batches.
    all_images, all_labels = [], []
    for i in range(1, 6):
        batch_file = SOURCE_DIR / f"data_batch_{i}"
        images, labels = load_batch(batch_file)
        all_images.append(images)
        all_labels.append(labels)

    images = np.concatenate(all_images)
    labels = np.concatenate(all_labels)

    # Stratified split: preserve class proportions in train and val.
    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=labels
    )

    # Save train and validation images.
    save_images(images[train_indices], labels[train_indices], "train", class_names)
    save_images(images[val_indices], labels[val_indices], "val", class_names)

    # Save the original test batch without splitting it.
    test_images, test_labels = load_batch(SOURCE_DIR / "test_batch")
    save_images(test_images, test_labels, "test", class_names)

    print(f"Classes: {class_names}")
    print(f"Validation ratio: {VAL_RATIO:.0%}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    print_image_counts(class_names)


if __name__ == "__main__":
    main()
