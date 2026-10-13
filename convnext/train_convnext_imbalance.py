"""Train a ConvNeXt classifier on CIFAR-10 with optional imbalance handling.

Run from the repository root (the folder containing preprocessing_cifar10.py).

Examples:
    python3 convnext/train_convnext_imbalance.py --imbalance 0.01 0.01 0.01 0.01 0.01 0.1 0.01 0.01 0.01 0.01 --epochs 1
    python3 convnext/train_convnext_imbalance.py --imbalance 1 1 1 0.05 0.1 1 0.05 1 1 0.8 --lr 0.0001 --epochs 20 # cat, deer and frog are scarce
    python3 convnext/train_convnext_imbalance.py --imbalance 1 1 1 0.05 0.1 1 0.05 1 1 0.8 --imbalance-train weighted_loss --lr 0.0001 --epochs 20
    python3 convnext/train_convnext_imbalance.py --imbalance 1 1 1 0.05 0.1 1 0.05 1 1 0.8 --imbalance-train resampling --lr 0.0001 --epochs 20
    python3 convnext/train_convnext_imbalance.py --imbalance 1 1 1 0.05 0.1 1 0.05 1 1 0.8 --imbalance-train synthetic --lr 0.0001 --epochs 20

`synthetic` uses imbalanced-learn's SMOTE on 4096-dimensional ConvNeXt penultimate-layer
features. It freezes the feature extractor and trains only the final classification layer,
so the precomputed synthetic features remain valid throughout training.

Use F1-score instead of Accuracy as evaluation metric.
"""
import os
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision import models

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from preprocessing_cifar10 import DATA_DIR, TestImages, get_device, get_loaders, set_seed  # noqa: E402

RUNS_DIR = HERE / "runs-cifar10-imbalance"
IMG_SIZE = 32


def build_model(freeze_backbone, scratch):
    weights = None if scratch else models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = models.convnext_tiny(weights=weights)

    # Replace the final linear layer (index 2 in TorchVision ConvNeXt)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 10)

    # Handle backbone freezing logic
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze ONLY the new classification linear head
        for p in model.classifier[2].parameters():
            p.requires_grad = True

    return model


def get_dataset_labels(dataset):
    """Get labels without assuming a particular custom dataset implementation."""
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets, dtype=torch.long)
    if hasattr(dataset, "labels"):
        return torch.as_tensor(dataset.labels, dtype=torch.long)
    return torch.tensor([int(dataset[i][1]) for i in range(len(dataset))], dtype=torch.long)


def make_loader(dataset, batch_size, num_workers, shuffle=False, sampler=None):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

def extract_penultimate_features(model, loader, device):
    """Extract 768-D feature vectors before ConvNeXt Tiny's final linear layer."""
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            x = model.features(images)
            x = model.avgpool(x)
            x = model.classifier[0](x)  # LayerNorm2d
            x = model.classifier[1](x)  # Flatten -> (batch_size, 768)
            all_features.append(x.cpu())
            all_labels.append(torch.as_tensor(labels, dtype=torch.long).cpu())

    return torch.cat(all_features), torch.cat(all_labels)


def apply_imbalance_train(loader, strategy, model=None, device=None, seed=42):
    """Apply weighted loss, random oversampling, or feature-space SMOTE.

    Returns (possibly replaced loader, class_weights, use_features).
    """
    if strategy is None:
        return loader, None, None, False

    labels = get_dataset_labels(loader.dataset)
    counts = torch.bincount(labels)
    present = counts > 0
    print("Class counts before handling:", counts.tolist())

    if strategy == "weighted_loss" or strategy == "focal_loss" or strategy == "logit_adjust":
        weights = counts.sum() / (present.sum() * counts.clamp_min(1).float())
        weights[~present] = 0
        print("Class weights:", [round(float(x), 4) for x in weights])
        return loader, counts, weights.float(), False

    if strategy == "resampling":
        sample_weights = 1.0 / counts[labels].float()
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(labels), replacement=True
        )
        resampled_loader = make_loader(
            loader.dataset, loader.batch_size, loader.num_workers, sampler=sampler
        )
        return resampled_loader, None, None, False

    if strategy == "synthetic":
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise ImportError(
                "Feature-space SMOTE requires imbalanced-learn. Install it with: "
                "pip install imbalanced-learn"
            ) from exc

        if model is None or device is None:
            raise ValueError("model and device are required for feature-space SMOTE.")

        # SMOTE needs at least two examples in every class to be oversampled.
        class_counts = Counter(labels.tolist())
        target_count = max(class_counts.values())
        too_small = [c for c, n in class_counts.items() if n < target_count and n < 2]
        if too_small:
            raise ValueError(
                f"SMOTE needs at least 2 samples per minority class; insufficient classes: {too_small}"
            )

        feature_loader = make_loader(
            loader.dataset, loader.batch_size, loader.num_workers, shuffle=False
        )
        features, feature_labels = extract_penultimate_features(model, feature_loader, device)
        X = features.numpy()
        y = feature_labels.numpy()

        min_class_count = min(class_counts[c] for c in class_counts if class_counts[c] < target_count)
        k_neighbors = min(5, min_class_count - 1)
        smote = SMOTE(
            sampling_strategy="minority",
            k_neighbors=k_neighbors,
            random_state=seed
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)

        smote_dataset = TensorDataset(
            torch.as_tensor(X_resampled, dtype=torch.float32),
            torch.as_tensor(y_resampled, dtype=torch.long)
        )
        smote_loader = make_loader(
            smote_dataset, loader.batch_size, loader.num_workers, shuffle=True
        )
        new_counts = torch.bincount(torch.as_tensor(y_resampled, dtype=torch.long))
        print("Class counts after feature-space SMOTE:", new_counts.tolist())
        return smote_loader, None, None, True

    raise ValueError(f"Unknown imbalance training strategy: {strategy}")

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -(1 - pt).pow(self.gamma) * log_pt

        if self.weight is not None:
            self.weight = self.weight.to(targets.device)
            loss = loss * self.weight[targets]

        return loss.mean()

class LogitAdjustedLoss(nn.Module):
    def __init__(self, class_counts, tau=1.0):
        super().__init__()
        priors = class_counts.float() / class_counts.sum()
        self.register_buffer("log_priors", priors.log())
        self.tau = tau

    def forward(self, logits, targets):
        adjusted_logits = logits + self.tau * self.log_priors
        adjusted_logits = adjusted_logits
        return F.cross_entropy(adjusted_logits, targets)

def run_epoch(model, loader, criterion, device, optimizer=None, feature_input=False):
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.classifier[2](inputs) if feature_input else model(inputs)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            n += batch_n

    macro_f1 = f1_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return total_loss / n, macro_f1


@torch.no_grad()
def predict_test(model, test_dir, device, args, out_dir):
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = TestImages(test_dir, imbalance=None)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    class_names = dataset.class_names
    num_classes = len(class_names)
    correct, total, rows = [0] * num_classes, [0] * num_classes, []

    for images, labels, names in loader:
        probs = torch.softmax(model(images.to(device)), dim=1).cpu()
        preds, labels = probs.argmax(dim=1), labels.cpu()

        for name, label, pred, prob in zip(names, labels, preds, probs):
            label, pred = int(label), int(pred)
            total[label] += 1
            correct[label] += int(pred == label)
            rows.append((Path(name).stem, pred, label, prob.tolist()))

    with open(out_dir / "submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows((name, pred) for name, pred, _, _ in rows)

    with open(out_dir / "test_probs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pred", "true_label"] + [f"prob_{name}" for name in class_names])
        writer.writerows((name, pred, label, *probs) for name, pred, label, probs in rows)

    per_class = {
        name: {
            "correct": correct[i],
            "total": total[i],
            "accuracy": correct[i] / total[i] if total[i] else None
        }
        for i, name in enumerate(class_names)
    }
    total_correct, total_samples = sum(correct), sum(total)
    results = {
        "per_class_accuracy": per_class,
        "total_accuracy": total_correct / total_samples if total_samples else None,
        "total_correct": total_correct,
        "total_samples": total_samples
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Wrote {len(rows)} predictions to {out_dir / 'submission.csv'}")
    print(f"Total accuracy: {results['total_accuracy']:.4f}")
    for name, metrics in per_class.items():
        acc = metrics["accuracy"]
        print(f"{name}: {acc:.4f}" if acc is not None else f"{name}: N/A")


def default_run_name(args):
    parts = ["convnext", f"lr{args.lr:g}",
             f"e{args.epochs}", f"s{args.seed}"]
    if args.imbalance_train:
        parts.append(args.imbalance_train)
    if args.freeze_backbone:
        parts.append("frozen")
    if args.no_augment:
        parts.append("noaug")
    if args.scratch:
        parts.append("scratch")
    if args.img_size != IMG_SIZE:
        parts.append(f"img{args.img_size}")
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--imbalance", nargs="*", type=float, help="create an imbalanced dataset")
    group.add_argument("--train-per-class", type=int, help="number of training images per class")
    ap.add_argument("--val-per-class", type=int, default=0, help="0 = use all validation images")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--imbalance-train", choices=["weighted_loss", "focal_loss", "logit_adjust", "resampling", "synthetic"],
                    default=None, help="training strategy for an imbalanced dataset")
    ap.add_argument("--freeze-backbone", action="store_true", help="freeze the backbone")
    ap.add_argument("--scratch", action="store_true", help="random init instead of ImageNet weights")
    ap.add_argument("--no-augment", action="store_true", help="disable training augmentation")
    ap.add_argument("--img-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None,
                    help="default: convnext/runs-cifar10-imbalance/<auto name from settings>")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow reusing a non-empty output directory")
    args = ap.parse_args()

    if args.scratch and args.freeze_backbone:
        ap.error("--scratch and --freeze-backbone cannot be combined")
    if args.imbalance is not None and len(args.imbalance) != 10:
        ap.error("--imbalance must contain exactly 10 values")
    if args.imbalance_train == "synthetic" and args.scratch:
        ap.error("--imbalance-train synthetic requires a pretrained ConvNeXt feature extractor; do not use --scratch")
    if args.img_size < 1:
        ap.error("--img-size must be positive")

    out_dir = Path(args.out_dir) if args.out_dir else RUNS_DIR / default_run_name(args)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        ap.error(f"{out_dir} already has results. Change a setting, choose --out-dir, or pass --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Run folder:", out_dir)

    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    train_loader, val_loader = get_loaders(
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class or None,
        imbalance=args.imbalance,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment
    )
    print(f"Original train images: {len(train_loader.dataset)}, val images: {len(val_loader.dataset)}")

    model = build_model(args.freeze_backbone, args.scratch).to(device)
    class_weights = None
    feature_input = False

    if args.imbalance_train == "synthetic":
        # Feature-space SMOTE uses fixed embeddings; train only the final linear layer.
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier[-1].parameters():
            p.requires_grad = True

    train_loader, class_counts, class_weights, feature_input = apply_imbalance_train(
        train_loader, args.imbalance_train, model=model, device=device, seed=args.seed
    )
    print(f"Training samples after handling: {len(train_loader.dataset)}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    if args.imbalance_train == "focal_loss":
        criterion = FocalLoss(weight=class_weights, gamma=2.0)
    elif args.imbalance_train == "logit_adjust":
        criterion = LogitAdjustedLoss(class_counts, 1.0).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best_f1, best_epoch = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer, feature_input=feature_input
        )
        # Validate on the original, unresampled validation images through the full model.
        va_loss, va_f1 = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        secs = time.time() - t0
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_f1=tr_f1,
                            val_loss=va_loss, val_f1=va_f1, time_s=secs))
        print(f"Epoch {epoch}/{args.epochs} | train loss {tr_loss:.4f} F1-score {tr_f1:.4f} | "
              f"val loss {va_loss:.4f} F1-score {va_f1:.4f} | {secs:.0f}s")
        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch
            torch.save(model.state_dict(), out_dir / "best.pt")

    summary = dict(
        args=vars(args), device=str(device),
        train_images=len(train_loader.dataset), val_images=len(val_loader.dataset),
        trainable_params=n_trainable, best_val_f1=best_f1, best_epoch=best_epoch,
        history=history
    )
    with open(out_dir / "history.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best validation F1-score: {best_f1:.4f} (epoch {best_epoch})")

    test_dir = DATA_DIR / "test"
    if test_dir.exists():
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
        predict_test(model, test_dir, device, args, out_dir)


if __name__ == "__main__":  # required on macOS when num_workers > 0
    main()
