"""Compare the ResNet50 runs in resnet50/runs/ (validation only, no test labels needed).

Usage (from the repository root):
    python resnet50/compare_runs.py
    python resnet50/compare_runs.py --test-run resnet50_lr1e-05_n19981_e10_s42

Reads : resnet50/runs/*/history.json  (and, for --test-run, final_metrics.json + error_analysis.csv
        written by predict_resnet50.py --evaluate)
Writes: resnet50/results/summary.csv, seeds.csv and fig_*.png

Experiments covered
  1. learning rate        -> fig_learning_rate.png
  2. augmentation on/off  -> fig_augmentation.png
  3. seeds (mean, std)    -> seeds.csv, noise threshold 2 std
  4. scratch vs pretrained-> fig_pretrained_vs_scratch.png
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS_DIR, RESULTS_DIR = HERE / "runs", HERE / "results"
EPOCHS = 10                      # epochs of every fine-tuning run
CLASS = {0: "cat", 1: "dog"}


# ------------------------------------------------------------------ loading
def load_runs():
    """One run per distinct setting (newest folder wins). Only full-data, 224 px, unfrozen runs."""
    runs = {}
    for d in sorted(RUNS_DIR.iterdir()):
        h = d / "history.json"
        if not d.is_dir() or d.name.startswith("smoke") or not h.exists():
            continue
        data = json.loads(h.read_text())
        a = data["args"]
        if (a.get("train_per_class") or 0) or a.get("freeze_backbone") or a.get("img_size", 224) != 224 \
                or a.get("batch_size", 32) != 32 or not data.get("completed", True):
            print(f"skipped (not part of the final plan): {d.name}")
            continue
        cfg = dict(lr=float(a["lr"]), epochs=int(a["epochs"]), seed=int(a["seed"]),
                   scratch=bool(a.get("scratch")), noaug=bool(a.get("no_augment")))
        key = tuple(cfg.values())
        label = f"lr={cfg['lr']:g}" + (" scratch" if cfg["scratch"] else "") \
            + (" no-aug" if cfg["noaug"] else "") + f" seed={cfg['seed']}" \
            + (f" ep={cfg['epochs']}" if cfg["epochs"] != EPOCHS else "")
        hist = data["history"]
        best = max(hist, key=lambda r: (r["val_acc"], -r["val_loss"]))   # same rule as training
        run = dict(dir=d, cfg=cfg, label=label, hist=hist, data=data, args=a, mtime=h.stat().st_mtime,
                   best_val=best["val_acc"], best_epoch=best["epoch"], best_loss=best["val_loss"],
                   train=hist[-1]["train_acc"], val=hist[-1]["val_acc"],
                   val_loss=hist[-1]["val_loss"], gap=hist[-1]["train_acc"] - hist[-1]["val_acc"],
                   sec=statistics.mean(r["time_s"] for r in hist if "time_s" in r))
        if key not in runs or run["mtime"] > runs[key]["mtime"]:
            if key in runs:
                print(f"note: duplicate settings, keeping newest: {d.name}")
            runs[key] = run
    return list(runs.values())


def find(runs, **want):
    base = dict(lr=None, epochs=EPOCHS, seed=42, scratch=False, noaug=False)
    base.update(want)
    return next((r for r in runs if all(base[k] is None or r["cfg"][k] == base[k] for k in base)), None)


def table(header, rows):
    w = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for i, r in enumerate([header] + rows):
        print("  ".join(str(x).ljust(n) for x, n in zip(r, w)))
        if i == 0:
            print("  ".join("-" * n for n in w))


# ------------------------------------------------------------------ figures
def save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  saved results/{name}")


def curves(items, title, name):
    """Validation loss and accuracy per epoch for [(run, legend), ...]."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, yl in zip(axes, ("val_loss", "val_acc"), ("Validation loss", "Validation accuracy")):
        for r, lab in items:
            ax.plot([h["epoch"] for h in r["hist"]], [h[key] for h in r["hist"]], marker="o", label=lab)
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(title)
    save(fig, name)


def train_vs_val(items, title, name):
    """Train (dashed) vs validation (solid): shows the overfitting gap."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (r, lab) in enumerate(items):
        ep = [h["epoch"] for h in r["hist"]]
        for ax, k in zip(axes, ("acc", "loss")):
            ax.plot(ep, [h[f"val_{k}"] for h in r["hist"]], "-o", color=f"C{i}", label=f"{lab} (val)")
            ax.plot(ep, [h[f"train_{k}"] for h in r["hist"]], "--", color=f"C{i}", label=f"{lab} (train)")
    for ax, yl in zip(axes, ("Accuracy", "Loss")):
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(title)
    save(fig, name)


def test_figures(run_dir, show_images=True):
    """Confusion matrix + error samples from predict_resnet50.py --evaluate outputs."""
    mpath, epath = run_dir / "final_metrics.json", run_dir / "error_analysis.csv"
    if not mpath.exists() or not epath.exists():
        print(f"\nNo final_metrics.json / error_analysis.csv in {run_dir.name}: "
              f"run predict_resnet50.py --run {run_dir.name} --evaluate first.")
        return
    m = json.loads(mpath.read_text())
    print(f"\nTEST ({run_dir.name}): acc {100 * m['accuracy']:.2f}%, precision {100 * m['precision']:.2f}%, "
          f"recall {100 * m['recall']:.2f}%, F1 {100 * m['f1']:.2f}%  (positive class = dog)")
    print(f"  TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}  (FP = cat predicted as dog)")
    cells = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cells, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cells[i][j], ha="center", va="center", fontsize=16,
                    color="white" if cells[i][j] > m["n"] / 4 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["cat", "dog"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["cat", "dog"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Test confusion matrix")
    save(fig, "fig_confusion_matrix.png")

    with open(epath, newline="") as f:
        rows = list(csv.DictReader(f))
    wrong = [r for r in rows if r["status"] == "wrong"]
    print(f"  {len(wrong)} wrong images:")
    for r in wrong:
        print(f"    id {r['id']}: true {CLASS[int(r['true'])]}, pred {CLASS[int(r['pred'])]}, "
              f"conf {max(float(r['prob_cat']), float(r['prob_dog'])):.3f}")
    if not show_images:
        return
    items = [r for r in wrong[:8] + [r for r in rows if r["status"] != "wrong"] if (ROOT / r["image_path"]).exists()]
    if not items:
        return
    cols = 4
    fig, axes = plt.subplots(math.ceil(len(items) / cols), cols, figsize=(3 * cols, 3.4 * math.ceil(len(items) / cols)))
    axes = list(axes.flat)
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, items):
        bad = r["status"] == "wrong"
        ax.imshow(Image.open(ROOT / r["image_path"]).convert("RGB"))
        ax.set_title(f"id {r['id']} [{'WRONG' if bad else 'correct, unsure'}]\n"
                     f"true {CLASS[int(r['true'])]} / pred {CLASS[int(r['pred'])]} "
                     f"({max(float(r['prob_cat']), float(r['prob_dog'])):.2f})",
                     fontsize=8, color="red" if bad else "black")
    save(fig, "fig_error_samples.png")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-run", default=None, help="run folder for confusion matrix / error samples "
                                                      "(default: the best-validation run)")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    if not RUNS_DIR.exists():
        sys.exit(f"Folder not found: {RUNS_DIR}")
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = load_runs()
    if not runs:
        sys.exit("No usable runs found.")

    # Baseline = best pretrained, augmented, seed-42 run among the learning-rate runs.
    lr_runs = sorted([r for r in runs if r["cfg"]["seed"] == 42 and not r["cfg"]["scratch"]
                      and not r["cfg"]["noaug"] and r["cfg"]["epochs"] == EPOCHS], key=lambda r: r["cfg"]["lr"])
    base = max(lr_runs, key=lambda r: (r["best_val"], -r["val_loss"]))

    # Seed noise: seeds of the baseline configuration.
    seeds = [r for r in runs if all(r["cfg"][k] == base["cfg"][k] for k in ("lr", "epochs", "scratch", "noaug"))]
    vals = [r["best_val"] for r in seeds]
    std = statistics.stdev(vals) if len(vals) > 1 else None
    noise = std if std else 0.0014
    print(f"\n{len(runs)} runs loaded. Baseline = {base['label']} (highest val accuracy of the lr runs)\n")

    # ---- table of every run
    rows, csv_rows = [], []
    for r in sorted(runs, key=lambda r: (r is not base, r["label"])):
        d = r["best_val"] - base["best_val"]
        flag = " (~noise)" if r is not base and abs(d) <= 2 * noise else ""
        rows.append([r["label"], r["dir"].name, f"{100 * r['best_val']:.2f}", r["best_epoch"],
                     f"{100 * r['train']:.2f}", f"{100 * r['val']:.2f}", f"{100 * r['gap']:.2f}",
                     f"{r['val_loss']:.4f}", f"{r['sec']:.0f}", f"{100 * d:+.2f}{flag}"])
        a, c = r["args"], r["cfg"]
        csv_rows.append(dict(folder=r["dir"].name, seed=c["seed"], lr=c["lr"],
                             train_per_class=a.get("train_per_class", 0), val_images=r["data"].get("val_images"),
                             epochs=c["epochs"], batch_size=a.get("batch_size", 32), frozen=bool(a.get("freeze_backbone")),
                             augmented=not c["noaug"], scratch=c["scratch"], best_val_acc=r["best_val"],
                             best_epoch=r["best_epoch"], best_val_loss=r["best_loss"], final_train_acc=r["train"],
                             final_val_acc=r["val"], final_val_loss=r["val_loss"], sec_per_epoch=r["sec"],
                             trainable_params=r["data"].get("trainable_params")))
    print("ALL RUNS ('~noise' = within 2 std of the seed-to-seed spread of the baseline)")
    table(["run", "folder", "best val %", "ep", "final train %", "final val %", "gap pp", "final val loss",
           "s/epoch", "d vs base pp"], rows)
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(csv_rows)

    # ---- 1. learning rate
    print("\n1. LEARNING RATE (augmentation on, seed 42)")
    table(["lr", "val acc epoch 1 %", "best val %", "best ep", "final val loss", "final train %"],
          [[f"{r['cfg']['lr']:g}", f"{100 * r['hist'][0]['val_acc']:.2f}", f"{100 * r['best_val']:.2f}",
            r["best_epoch"], f"{r['val_loss']:.4f}", f"{100 * r['train']:.2f}"] for r in lr_runs])
    if len(lr_runs) > 1:
        curves([(r, f"lr={r['cfg']['lr']:g}") for r in lr_runs],
               "Effect of learning rate (ResNet50, pretrained)", "fig_learning_rate.png")

    # ---- 2. augmentation (every seed that exists for both settings)
    lr0 = base["cfg"]["lr"]
    noaug_runs = [r for r in runs if r["cfg"]["noaug"] and not r["cfg"]["scratch"]
                  and r["cfg"]["lr"] == lr0 and r["cfg"]["epochs"] == EPOCHS]
    common = sorted({r["cfg"]["seed"] for r in noaug_runs} & {r["cfg"]["seed"] for r in seeds})
    if common:
        print(f"\n2. AUGMENTATION at lr={lr0:g}, seeds {common}")
        rows2 = []
        for name, group in (("augmentation", seeds), ("no augmentation", noaug_runs)):
            g = sorted([r for r in group if r["cfg"]["seed"] in common], key=lambda r: r["cfg"]["seed"])
            for r in g:
                rows2.append([name, r["cfg"]["seed"], f"{100 * r['best_val']:.2f}", f"{100 * r['val']:.2f}",
                              f"{100 * r['train']:.2f}", f"{100 * r['gap']:.2f}", f"{r['val_loss']:.4f}"])
            m = lambda k: statistics.mean(r[k] for r in g)
            rows2.append([name, "mean", f"{100 * m('best_val'):.2f}", f"{100 * m('val'):.2f}",
                          f"{100 * m('train'):.2f}", f"{100 * m('gap'):.2f}", f"{m('val_loss'):.4f}"])
        table(["setting", "seed", "best val %", "final val %", "final train %", "gap pp", "final val loss"], rows2)
        pair = (find(runs, lr=lr0, seed=common[0]), find(runs, lr=lr0, seed=common[0], noaug=True))
        train_vs_val([(pair[0], "augmentation"), (pair[1], "no augmentation")],
                     f"Effect of augmentation (lr={lr0:g}, seed {common[0]})", "fig_augmentation.png")

    # ---- 3. seeds
    if len(seeds) > 1:
        print(f"\n3. SEEDS: {base['label'].rsplit(' seed', 1)[0]} with seeds "
              f"{sorted(r['cfg']['seed'] for r in seeds)}")
        finals = [r["val"] for r in seeds]
        out = [["metric", "mean %", "std %", "min %", "max %"]]
        for name, v in (("best val acc", vals), ("final val acc", finals)):
            out.append([name] + [f"{100 * x:.3f}" for x in (statistics.mean(v), statistics.stdev(v), min(v), max(v))])
        table(out[0], out[1:])
        n_val = seeds[0]["data"].get("val_images", 5000)
        p = statistics.mean(vals)
        se = math.sqrt(p * (1 - p) / n_val)
        print(f"  (sample std; a difference smaller than 2 std = {200 * noise:.2f} pp is not evidence)")
        print(f"  Binomial standard error of ONE val accuracy on {n_val} images: {100 * se:.2f} pp "
              f"(1 image = {100 / n_val:.2f} pp). Differences of a few images are weak evidence.")
        print("  'best val acc' is picked on the validation set itself, so it is slightly optimistic; "
              "'final val acc' is not.")
        with open(RESULTS_DIR / "seeds.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seeds", "mean_val_acc", "std_val_acc", "min_val_acc", "max_val_acc",
                                              "mean_final_val_acc", "std_final_val_acc"], lineterminator="\n")
            w.writeheader()
            w.writerow(dict(seeds=",".join(map(str, sorted(r["cfg"]["seed"] for r in seeds))),
                            mean_val_acc=statistics.mean(vals), std_val_acc=statistics.stdev(vals),
                            min_val_acc=min(vals), max_val_acc=max(vals),
                            mean_final_val_acc=statistics.mean(finals), std_final_val_acc=statistics.stdev(finals)))
    else:
        print("\n3. SEEDS: only one seed found, run --seed 1 and --seed 2 for the noise level.")

    # ---- 4. scratch vs pretrained
    scratch = [r for r in runs if r["cfg"]["scratch"]]
    if scratch:
        print("\n4. PRETRAINED vs SCRATCH")
        items = [(base, f"pretrained lr={base['cfg']['lr']:g} ({EPOCHS} ep)")]
        same_lr = find(runs, lr=scratch[0]["cfg"]["lr"])
        if same_lr and same_lr is not base:
            items.append((same_lr, f"pretrained lr={same_lr['cfg']['lr']:g} ({EPOCHS} ep)"))
        items += [(r, f"scratch lr={r['cfg']['lr']:g} ({r['cfg']['epochs']} ep)") for r in scratch]
        table(["run", "best val %", "final train %", "final val %"],
              [[lab, f"{100 * r['best_val']:.2f}", f"{100 * r['train']:.2f}", f"{100 * r['val']:.2f}"] for r, lab in items])
        for r in scratch:
            v = [h["val_acc"] for h in r["hist"]]
            gain = v[-1] - v[-4]
            print(f"  scratch val acc gain over its last 3 epochs: {100 * gain:+.2f} pp"
                  + (" (still improving -> the measured gap to pretrained is an upper bound of the gap at convergence)" if gain > 0.001 else ""))
        curves(items, "Pretrained vs random initialisation", "fig_pretrained_vs_scratch.png")

    # ---- test analysis of the chosen run
    chosen = next((r for r in runs if r["dir"].name == args.test_run), base) if args.test_run else base
    test_figures(chosen["dir"], not args.no_images)


if __name__ == "__main__":
    main()