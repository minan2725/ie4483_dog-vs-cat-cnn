"""Summarize ConvNeXt validation runs without reading test labels or predictions."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
RESULTS_DIR = HERE / "results"
RESNET_SUMMARY = HERE.parent / "resnet50" / "results" / "summary.csv"
BASELINE_SEEDS = (42, 1, 2)


def baseline(args):
    return (args.get("train_per_class") == 5000 and
            args.get("val_per_class", 0) == 0 and
            args.get("epochs") == 5 and
            args.get("lr") == 1e-4 and
            args.get("batch_size") == 32 and
            not args.get("freeze_backbone") and
            not args.get("no_augment"))


def load_runs():
    runs = []
    for folder in sorted(RUNS_DIR.glob("*")):
        history_path = folder / "history.json"
        if not folder.is_dir() or folder.name.startswith("smoke") or not history_path.exists():
            continue
        data = json.loads(history_path.read_text())
        if not data.get("completed"):
            continue
        args = data["args"]
        epochs = data["history"]
        if len(epochs) != args["epochs"]:
            raise ValueError(f"Incomplete history in {folder}")
        best = epochs[data["best_epoch"] - 1]
        runs.append(dict(folder=folder.name, args=args, data=data, best=best))
    return runs


def write_csv(path, rows, fields):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in runs:
        label = run["folder"].replace("convnext_tiny_", "")
        epochs = run["data"]["history"]
        axes[0].plot([e["epoch"] for e in epochs],
                     [100 * e["val_acc"] for e in epochs], marker="o", label=label)
        axes[1].plot([e["epoch"] for e in epochs],
                     [e["val_loss"] for e in epochs], marker="o", label=label)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].set_ylabel("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks([e["epoch"] for e in runs[0]["data"]["history"]])
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(RESULTS_DIR / "fig_validation.png", dpi=150)
    plt.close(fig)


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit(f"No completed runs found in {RUNS_DIR}")
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for run in runs:
        args, data, best = run["args"], run["data"], run["best"]
        rows.append(dict(folder=run["folder"], seed=args["seed"],
                         lr=args["lr"], train_per_class=args["train_per_class"],
                         val_images=data["val_images"], epochs=args["epochs"],
                         batch_size=args["batch_size"],
                         frozen=args["freeze_backbone"],
                         augmented=not args["no_augment"],
                         best_val_acc=data["best_val_acc"],
                         best_epoch=data["best_epoch"],
                         best_val_loss=best["val_loss"],
                         final_train_acc=data["history"][-1]["train_acc"],
                         sec_per_epoch=statistics.mean(e["time_s"] for e in data["history"]),
                         trainable_params=data["trainable_params"]))
    fields = list(rows[0])
    write_csv(RESULTS_DIR / "summary.csv", rows, fields)
    make_figure(runs)

    eligible = [run for run in runs if baseline(run["args"]) and
                run["args"]["seed"] in BASELINE_SEEDS]
    by_seed = {run["args"]["seed"]: run for run in eligible}
    if len(eligible) != len(by_seed):
        raise SystemExit("Duplicate baseline seed runs; resolve them before reporting")
    if len(by_seed) >= 2:
        values = [run["data"]["best_val_acc"] for run in by_seed.values()]
        write_csv(RESULTS_DIR / "seeds.csv", [dict(
            seeds=",".join(map(str, sorted(by_seed))), mean_val_acc=statistics.mean(values),
            std_val_acc=statistics.stdev(values), min_val_acc=min(values),
            max_val_acc=max(values))],
            ["seeds", "mean_val_acc", "std_val_acc", "min_val_acc", "max_val_acc"])

    if RESNET_SUMMARY.exists():
        with RESNET_SUMMARY.open(newline="") as file:
            resnet = {row["folder"]: row for row in csv.DictReader(file)}
        comparison = []
        for seed in BASELINE_SEEDS:
            convnext = by_seed.get(seed)
            resnet_row = resnet.get(f"resnet50_lr0.0001_n5000_e5_s{seed}")
            if convnext and resnet_row:
                comparison.append(dict(seed=seed,
                                       convnext_val_acc=convnext["data"]["best_val_acc"],
                                       resnet50_val_acc=resnet_row["best_val_acc"],
                                       delta=convnext["data"]["best_val_acc"] -
                                       float(resnet_row["best_val_acc"])))
        if comparison:
            write_csv(RESULTS_DIR / "model_comparison.csv", comparison,
                      ["seed", "convnext_val_acc", "resnet50_val_acc", "delta"])

    for row in sorted(rows, key=lambda item: (-item["best_val_acc"], item["best_val_loss"])):
        print(f"{row['best_val_acc']:.4%}  epoch {row['best_epoch']}  {row['folder']}")
    selected = max(runs, key=lambda run: (run["data"]["best_val_acc"],
                                          -run["best"]["val_loss"]))
    print(f"Highest validation run: {selected['folder']}")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
