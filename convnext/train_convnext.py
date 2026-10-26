"""Train an ImageNet-pretrained ConvNeXt-Tiny for Dogs vs. Cats.

Run from the repository root. Each experiment has its own directory under
convnext/runs/ and uses preprocessing.py for transforms and duplicate removal.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torchvision import models

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from preprocessing import get_device, get_loaders, set_seed  # noqa: E402

RUNS_DIR = HERE / "runs"
EXPECTED_TRAIN_IMAGES = 19_981
EXPECTED_VAL_IMAGES = 5_000


def build_model(freeze_backbone=False, pretrained=True):
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    head = model.classifier[-1]
    model.classifier[-1] = nn.Linear(head.in_features, 2)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training and not any(p.requires_grad for p in model.features.parameters()):
        model.features.eval()
    total_loss = correct = count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch_size
    return total_loss / count, correct / count


def default_run_name(args):
    parts = ["convnext_tiny", f"lr{args.lr:g}", f"n{EXPECTED_TRAIN_IMAGES}",
             f"e{args.epochs}", f"s{args.seed}"]
    if args.batch_size != 32:
        parts.append(f"b{args.batch_size}")
    if args.freeze_backbone:
        parts.append("frozen")
    if args.no_augment:
        parts.append("noaug")
    return "_".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.lr) <= 0:
        parser.error("epochs, batch-size, and lr must be positive")
    if args.num_workers < 0:
        parser.error("num-workers cannot be negative")
    return args


def main():
    args = parse_args()
    out_dir = args.out_dir or RUNS_DIR / default_run_name(args)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} already has files; choose --out-dir or pass --overwrite")

    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        seed=args.seed,
    )
    if len(train_loader.dataset) != EXPECTED_TRAIN_IMAGES:
        raise SystemExit(f"Expected {EXPECTED_TRAIN_IMAGES} training images; found {len(train_loader.dataset)}")
    if len(val_loader.dataset) != EXPECTED_VAL_IMAGES:
        raise SystemExit(f"Expected {EXPECTED_VAL_IMAGES} validation images; found {len(val_loader.dataset)}")
    if val_loader.dataset.class_to_idx != {"cat": 0, "dog": 1}:
        raise SystemExit("Validation labels must be cat=0, dog=1")

    model = build_model(freeze_backbone=args.freeze_backbone).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {out_dir}")
    print(f"Device: {device}; train: {len(train_loader.dataset)}; val: {len(val_loader.dataset)}")

    history = []
    best_acc = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - start
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, time_s=elapsed))
        print(f"Epoch {epoch}/{args.epochs}: train {train_acc:.4f}/{train_loss:.4f}, "
              f"val {val_acc:.4f}/{val_loss:.4f}, {elapsed:.1f}s", flush=True)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), out_dir / "best.pt")

    summary = dict(args={**vars(args), "out_dir": str(out_dir)},
                   device=str(device), torch_version=torch.__version__,
                   train_images=len(train_loader.dataset),
                   val_images=len(val_loader.dataset),
                   trainable_params=sum(p.numel() for p in trainable),
                   best_val_acc=best_acc, best_epoch=best_epoch,
                   completed=True, history=history)
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Best validation accuracy: {best_acc:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
