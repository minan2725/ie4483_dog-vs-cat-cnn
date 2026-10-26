"""Shared Dogs vs. Cats training and optional test prediction pipeline."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from preprocessing import DATA_DIR, TestImages, get_device, get_loaders, set_seed

ROOT = Path(__file__).resolve().parent
FULL_TRAIN_IMAGES = 19_981
FULL_VAL_IMAGES = 5_000


def parse_training_args(model_name, argv=None):
    parser = argparse.ArgumentParser(description=f"Train {model_name} on Dogs vs. Cats")
    parser.add_argument("--train-per-class", type=int, default=0,
                        help="0 uses all clean training images")
    parser.add_argument("--val-per-class", type=int, default=0,
                        help="0 uses all validation images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--predict-test", action="store_true",
                        help="write test predictions after selecting the best validation checkpoint")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.train_per_class < 0 or args.val_per_class < 0 or args.num_workers < 0:
        parser.error("per-class limits and num-workers must be nonnegative")
    if min(args.epochs, args.lr, args.batch_size, args.img_size) <= 0:
        parser.error("epochs, lr, batch-size, and img-size must be positive")
    if args.scratch and args.freeze_backbone:
        parser.error("--scratch and --freeze-backbone cannot be combined")
    return args


def default_run_name(model_name, args, train_images):
    n = args.train_per_class or train_images
    parts = [model_name, f"lr{args.lr:g}", f"n{n}", f"e{args.epochs}", f"s{args.seed}"]
    if args.val_per_class:
        parts.append(f"v{args.val_per_class}")
    if args.batch_size != 32:
        parts.append(f"b{args.batch_size}")
    if args.freeze_backbone:
        parts.append("frozen")
    if args.scratch:
        parts.append("scratch")
    if args.no_augment:
        parts.append("noaug")
    if args.img_size != 224:
        parts.append(f"img{args.img_size}")
    return "_".join(parts)


def run_epoch(model, loader, criterion, device, optimizer=None, freeze_backbone=False):
    training = optimizer is not None
    model.train(training and not freeze_backbone)
    total_loss = correct = count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
    if not count:
        raise ValueError("Cannot train or validate on an empty dataset")
    return total_loss / count, correct / count


def predict_test(model, device, args, out_dir, test_dir=DATA_DIR / "test"):
    dataset = TestImages(test_dir, img_size=args.img_size)
    ids = [path.stem for path in dataset.paths]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError(f"Expected 500 distinct test image IDs; found {len(ids)} images")
    answer_path = ROOT / "test_answer.csv"
    if answer_path.exists():
        with answer_path.open(newline="", encoding="utf-8-sig") as file:
            expected = {row["id"].strip() for row in csv.DictReader(file)}
        if set(ids) != expected:
            raise ValueError("Test image IDs do not match the expected 500 IDs")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, names in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            for name, p in zip(names, probs):
                rows.append((Path(name).stem, int(p.argmax()), float(p[0]), float(p[1])))
    with (out_dir / "submission.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["id", "label"])
        writer.writerows((row[0], row[1]) for row in rows)
    with (out_dir / "test_probs.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["id", "pred", "prob_cat", "prob_dog"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {out_dir / 'submission.csv'}")


def train_model(model_name, model_factory, args, runs_dir=None):
    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader = get_loaders(
        train_per_class=args.train_per_class or None,
        val_per_class=args.val_per_class or None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        augment=not args.no_augment,
        seed=args.seed,
    )
    if not args.train_per_class and len(train_loader.dataset) != FULL_TRAIN_IMAGES:
        raise ValueError(f"Expected {FULL_TRAIN_IMAGES} clean training images; "
                         f"found {len(train_loader.dataset)}")
    if not args.val_per_class and len(val_loader.dataset) != FULL_VAL_IMAGES:
        raise ValueError(f"Expected {FULL_VAL_IMAGES} validation images; "
                         f"found {len(val_loader.dataset)}")
    val_dataset = val_loader.dataset
    if isinstance(val_dataset, Subset):
        val_dataset = val_dataset.dataset
    if val_dataset.class_to_idx != {"cat": 0, "dog": 1}:
        raise ValueError("Validation labels must be cat=0, dog=1")

    runs_dir = Path(runs_dir) if runs_dir is not None else ROOT / model_name / "runs"
    out_dir = args.out_dir or runs_dir / default_run_name(
        model_name, args, len(train_loader.dataset))
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} already has files; choose --out-dir or pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model_factory(args).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("The model has no trainable parameters")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"Run folder: {out_dir}")
    print(f"Device: {device}; train: {len(train_loader.dataset)}; val: {len(val_loader.dataset)}")

    history = []
    best_acc, best_loss, best_epoch = -1.0, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, args.freeze_backbone)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - start
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, time_s=elapsed))
        print(f"Epoch {epoch}/{args.epochs}: train {train_acc:.4f}/{train_loss:.4f}, "
              f"val {val_acc:.4f}/{val_loss:.4f}, {elapsed:.1f}s", flush=True)
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            torch.save(model.state_dict(), out_dir / "best.pt")

    summary = dict(args={**vars(args), "out_dir": str(out_dir)}, device=str(device),
                   torch_version=torch.__version__, train_images=len(train_loader.dataset),
                   val_images=len(val_loader.dataset),
                   trainable_params=sum(p.numel() for p in trainable),
                   best_val_acc=best_acc, best_epoch=best_epoch,
                   completed=True, history=history)
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Best validation accuracy: {best_acc:.4f} (epoch {best_epoch})")

    if args.predict_test:
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=True))
        predict_test(model, device, args, out_dir)
    return out_dir
