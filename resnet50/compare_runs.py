"""Compare all ResNet50 training runs in resnet50/runs/ and evaluate their submissions on the test answers.

Usage (from the repository root):
    pip install matplotlib
    python3 resnet50/compare_runs.py                    # analyse everything, detailed analysis of the best-validation run
    python3 resnet50/compare_runs.py --run <folder>     # detailed analysis of a specific run folder name
    python3 resnet50/compare_runs.py --no-images        # skip the image grid of test samples

Reads : resnet50/runs/*/history.json, submission.csv, test_probs.csv, and test_answer.csv (repo root)
Writes: resnet50/results/summary.csv, seeds.csv, error_analysis.csv, *.png
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write figures to files; no window needed
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent      # resnet50/
ROOT = HERE.parent                          # repository root
RUNS_DIR = HERE / "runs"
RESULTS_DIR = HERE / "results"
ANSWER_PATH = ROOT / "test_answer.csv"
TEST_IMG_DIR = ROOT / "datasets" / "test"
SKIP_PREFIXES = ("smoke",)  # run folders starting with these names are ignored
CLASS_NAME = {0: "cat", 1: "dog"}

# The baseline configuration. Every other run differs from it in (ideally) one setting.
BASE = dict(lr=1e-4, n=5000, epochs=5, seed=42, frozen=False, scratch=False, noaug=False, img=224)


# ----------------------------------------------------------------------------- loading runs
def config_of(args):
    return dict(
        lr=float(args.get("lr", 1e-4)),
        n=int(args.get("train_per_class", 5000)),
        epochs=int(args.get("epochs", 5)),
        seed=int(args.get("seed", 42)),
        frozen=bool(args.get("freeze_backbone", False)),
        scratch=bool(args.get("scratch", False)),
        noaug=bool(args.get("no_augment", False)),
        img=int(args.get("img_size", 224)),
    )


def label_of(cfg):
    parts = []
    if cfg["scratch"]:
        parts.append("scratch")
    if cfg["frozen"]:
        parts.append("frozen")
    if cfg["noaug"]:
        parts.append("no-aug")
    if cfg["img"] != BASE["img"]:
        parts.append(f"img{cfg['img']}")
    if cfg["n"] != BASE["n"]:
        parts.append(f"n={cfg['n']}")
    if cfg["lr"] != BASE["lr"]:
        parts.append(f"lr={cfg['lr']:g}")
    if cfg["epochs"] != BASE["epochs"]:
        parts.append(f"ep={cfg['epochs']}")
    if cfg["seed"] != BASE["seed"]:
        parts.append(f"seed={cfg['seed']}")
    return " ".join(parts) if parts else "baseline"


def load_runs():
    best = {}
    if not RUNS_DIR.exists():
        sys.exit(f"Folder not found: {RUNS_DIR}")
    for d in sorted(RUNS_DIR.iterdir()):
        h = d / "history.json"
        if not d.is_dir() or d.name.startswith(SKIP_PREFIXES) or not h.exists():
            continue
        data = json.loads(h.read_text())
        cfg = config_of(data["args"])
        key = tuple(sorted(cfg.items()))
        run = dict(dir=d, data=data, cfg=cfg, label=label_of(cfg), hist=data["history"])
        score = ("time_s" in run["hist"][0], h.stat().st_mtime)
        if key in best:
            old = best[key]
            print(f"note: '{d.name}' and '{old['dir'].name}' have identical settings; keeping one of them.")
            if score <= old["score"]:
                continue
        run["score"] = score
        best[key] = run
    return list(best.values())


def find(runs, **over):
    want = dict(BASE)
    want.update(over)
    for r in runs:
        if r["cfg"] == want:
            return r
    return None


def varying(runs, *keys):
    """Runs equal to the baseline in every setting except the given ones."""
    return [r for r in runs if all(r["cfg"][k] == BASE[k] for k in BASE if k not in keys)]


def run_stats(run):
    hist = run["hist"]
    best = max(hist, key=lambda r: r["val_acc"])
    last = hist[-1]
    times = [r["time_s"] for r in hist if "time_s" in r]
    return dict(
        best_val_acc=best["val_acc"], best_epoch=best["epoch"],
        final_train_acc=last["train_acc"], final_val_acc=last["val_acc"],
        gap=last["train_acc"] - last["val_acc"], final_val_loss=last["val_loss"],
        sec_per_epoch=statistics.mean(times) if times else None,
        trainable=run["data"].get("trainable_params"),
    )


# ----------------------------------------------------------------------------- test metrics
def norm_id(x):
    s = str(x).strip()
    if s.lower().endswith((".jpg", ".jpeg", ".png")):
        s = s.rsplit(".", 1)[0]
    return str(int(s)) if s.isdigit() else s


def read_labels(path):
    """Read a csv with columns id,label -> {id: label}. Row order does not matter."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r]
    header = [h.strip().lower() for h in rows[0]]
    if "id" not in header or "label" not in header:
        sys.exit(f"{path} must have columns 'id' and 'label' (found {rows[0]})")
    i_id, i_lab = header.index("id"), header.index("label")
    return {norm_id(r[i_id]): int(float(r[i_lab])) for r in rows[1:]}


def test_metrics(truth, pred, name):
    ids = sorted(set(truth) & set(pred))
    if len(ids) != len(truth) or len(ids) != len(pred):
        print(f"warning: {name}: {len(pred)} predictions vs {len(truth)} answers, "
              f"{len(ids)} ids matched. Check the id format.")
    tp = sum(1 for i in ids if truth[i] == 1 and pred[i] == 1)
    tn = sum(1 for i in ids if truth[i] == 0 and pred[i] == 0)
    fp = sum(1 for i in ids if truth[i] == 0 and pred[i] == 1)
    fn = sum(1 for i in ids if truth[i] == 1 and pred[i] == 0)
    n = len(ids)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n=n, tp=tp, tn=tn, fp=fp, fn=fn,
                acc=(tp + tn) / n if n else 0.0, precision=prec, recall=rec, f1=f1,
                specificity=tn / (tn + fp) if tn + fp else 0.0)


# ----------------------------------------------------------------------------- printing
def pct(x):
    return "-" if x is None else f"{100 * x:.2f}"


def print_table(header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths))
    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


# ----------------------------------------------------------------------------- figures
def save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  saved results/{name}")


def curves_figure(items, title, name):
    """items: list of (run, legend label). Validation loss and accuracy per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, ylabel in zip(axes, ("val_loss", "val_acc"), ("Validation loss", "Validation accuracy")):
        for run, label in items:
            ax.plot([h["epoch"] for h in run["hist"]], [h[key] for h in run["hist"]],
                    marker="o", label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    save(fig, name)


def train_val_figure(items, title, name):
    """Training (dashed) vs validation (solid) accuracy: shows the overfitting gap."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (run, label) in enumerate(items):
        color = f"C{i}"
        ep = [h["epoch"] for h in run["hist"]]
        axes[0].plot(ep, [h["val_acc"] for h in run["hist"]], "-o", color=color, label=f"{label} (val)")
        axes[0].plot(ep, [h["train_acc"] for h in run["hist"]], "--", color=color, label=f"{label} (train)")
        axes[1].plot(ep, [h["val_loss"] for h in run["hist"]], "-o", color=color, label=f"{label} (val)")
        axes[1].plot(ep, [h["train_loss"] for h in run["hist"]], "--", color=color, label=f"{label} (train)")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    save(fig, name)


def make_figures(runs):
    print("Figures:")
    # 1. learning rate
    items = sorted([r for r in varying(runs, "lr")], key=lambda r: r["cfg"]["lr"])
    if len(items) > 1:
        curves_figure([(r, f"lr={r['cfg']['lr']:g}") for r in items],
                      "Effect of learning rate (ResNet50, pretrained)", "fig_learning_rate.png")
    # 2. frozen backbone vs full fine-tuning (same lr)
    for lr in sorted({r["cfg"]["lr"] for r in runs if r["cfg"]["frozen"]}):
        full, frozen = find(runs, lr=lr), find(runs, lr=lr, frozen=True)
        if full and frozen:
            def lab(r, text):
                st = r["stats"]
                extra = f"{st['sec_per_epoch']:.0f}s/epoch" if st["sec_per_epoch"] else ""
                if st["trainable"]:
                    extra += f", {st['trainable'] / 1e6:.1f}M trainable"
                return f"{text} ({extra})" if extra else text
            curves_figure([(full, lab(full, "full fine-tune")), (frozen, lab(frozen, "frozen backbone"))],
                          f"Training strategy at lr={lr:g}", "fig_frozen_vs_full.png")
    # 3. augmentation
    items = varying(runs, "noaug")
    if len(items) > 1:
        train_val_figure([(r, "no augmentation" if r["cfg"]["noaug"] else "augmentation") for r in items],
                         "Effect of augmentation (5000 images/class)", "fig_augmentation.png")
    small = [find(runs, n=n, noaug=na) for n in sorted({r["cfg"]["n"] for r in runs if r["cfg"]["noaug"] and r["cfg"]["n"] != BASE["n"]}) for na in (False, True)]
    small = [r for r in small if r]
    if len(small) >= 2:
        train_val_figure([(r, f"n={r['cfg']['n']} " + ("no-aug" if r["cfg"]["noaug"] else "aug")) for r in small],
                         "Augmentation with little data", "fig_augmentation_small_data.png")
    # 4. amount of training data
    items = sorted(varying(runs, "n"), key=lambda r: r["cfg"]["n"])
    if len(items) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = [r["cfg"]["n"] for r in items]
        ax.plot(xs, [100 * r["stats"]["best_val_acc"] for r in items], "-o", label="validation")
        if all(r["test"] for r in items):
            ax.plot(xs, [100 * r["test"]["acc"] for r in items], "-s", label="test")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel("Training images per class")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Learning curve: amount of training data")
        ax.grid(alpha=0.3)
        ax.legend()
        save(fig, "fig_data_size.png")
    # 5. image size
    items = sorted(varying(runs, "img"), key=lambda r: r["cfg"]["img"])
    if len(items) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        names = [f"{r['cfg']['img']}x{r['cfg']['img']}" for r in items]
        accs = [100 * r["stats"]["best_val_acc"] for r in items]
        axes[0].bar(names, accs)
        axes[0].set_ylim(max(0, min(accs) - 2), min(100, max(accs) + 1))
        axes[0].set_ylabel("Best validation accuracy (%)")
        for i, a in enumerate(accs):
            axes[0].text(i, a, f"{a:.2f}", ha="center", va="bottom")
        secs = [r["stats"]["sec_per_epoch"] for r in items]
        if all(secs):
            axes[1].bar(names, secs)
            axes[1].set_ylabel("Seconds per epoch")
        axes[0].set_title("Accuracy")
        axes[1].set_title("Training time")
        fig.suptitle("Effect of input image size")
        save(fig, "fig_image_size.png")
    # 6. pretrained vs scratch: ALL scratch runs with the same lr go into one figure,
    #    so that e.g. 10 and 20 epochs can be compared with the pretrained run.
    scratch_runs = sorted([r for r in runs if r["cfg"]["scratch"]], key=lambda r: r["cfg"]["epochs"])
    lrs = sorted({r["cfg"]["lr"] for r in scratch_runs})
    for lr in lrs:
        pre = find(runs, lr=lr)
        if not pre:
            continue
        items = [(pre, f"pretrained ({pre['cfg']['epochs']} epochs)")]
        items += [(r, f"from scratch ({r['cfg']['epochs']} epochs)")
                  for r in scratch_runs if r["cfg"]["lr"] == lr]
        fname = "fig_pretrained_vs_scratch.png" if len(lrs) == 1 else f"fig_pretrained_vs_scratch_lr{lr:g}.png"
        curves_figure(items, f"Pretrained vs random initialisation (lr={lr:g})", fname)
    # 7. overview of all runs
    ordered = sorted(runs, key=lambda r: r["stats"]["best_val_acc"])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(ordered) + 1.5))
    ys = range(len(ordered))
    ax.barh(list(ys), [100 * r["stats"]["best_val_acc"] for r in ordered], alpha=0.8, label="validation (best epoch)")
    tests = [(y, 100 * r["test"]["acc"]) for y, r in zip(ys, ordered) if r["test"]]
    if tests:
        ax.plot([t[1] for t in tests], [t[0] for t in tests], "k*", label="test")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["label"] for r in ordered])
    lo = min([100 * r["stats"]["best_val_acc"] for r in ordered] + [t[1] for t in tests])
    ax.set_xlim(max(0, lo - 3), 100)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("All runs")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    save(fig, "fig_overview.png")


def confusion_figure(run, name):
    t = run["test"]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow([[t["tn"], t["fp"]], [t["fn"], t["tp"]]], cmap="Blues")
    for i, row in enumerate([[t["tn"], t["fp"]], [t["fn"], t["tp"]]]):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                    color="white" if v > t["n"] / 4 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["cat", "dog"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["cat", "dog"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Test confusion matrix\n{run['label']}")
    save(fig, name)


# ----------------------------------------------------------------------------- error analysis
def error_analysis(run, truth, show_images):
    path = run["dir"] / "test_probs.csv"
    if not path.exists():
        print("No test_probs.csv in that run; skipping error analysis.")
        return
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            i = norm_id(r["id"])
            if i not in truth:
                continue
            pc, pdg = float(r["prob_cat"]), float(r["prob_dog"])
            rows.append(dict(id=i, true=truth[i], pred=int(r["pred"]),
                             prob_cat=pc, prob_dog=pdg, conf=max(pc, pdg)))
    wrong = sorted([r for r in rows if r["pred"] != r["true"]], key=lambda r: -r["conf"])
    unsure = sorted([r for r in rows if r["pred"] == r["true"]], key=lambda r: r["conf"])[:8]
    with open(RESULTS_DIR / "error_analysis.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "status", "true", "pred", "prob_cat", "prob_dog"])
        for status, group in (("wrong", wrong), ("correct_low_confidence", unsure)):
            for r in group:
                w.writerow([r["id"], status, CLASS_NAME[r["true"]], CLASS_NAME[r["pred"]],
                            f"{r['prob_cat']:.4f}", f"{r['prob_dog']:.4f}"])
    print(f"\nError analysis for '{run['dir'].name}': {len(wrong)} wrong out of {len(rows)} test images")
    for r in wrong:
        print(f"  id {r['id']}: true {CLASS_NAME[r['true']]}, predicted {CLASS_NAME[r['pred']]} "
              f"(confidence {r['conf']:.3f})")
    print("  saved results/error_analysis.csv")
    if not show_images or not TEST_IMG_DIR.exists():
        return
    images = {norm_id(p.stem): p for p in TEST_IMG_DIR.iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    items = [(r, True) for r in wrong[:8]] + [(r, False) for r in unsure]
    items = [(r, w_) for r, w_ in items if r["id"] in images]
    if not items:
        return
    cols = 4
    nrows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(3 * cols, 3.4 * nrows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, (r, is_wrong) in zip(axes, items):
        ax.imshow(Image.open(images[r["id"]]).convert("RGB"))
        tag = "WRONG" if is_wrong else "correct, unsure"
        ax.set_title(f"id {r['id']} [{tag}]\ntrue {CLASS_NAME[r['true']]} / pred {CLASS_NAME[r['pred']]} "
                     f"({r['conf']:.2f})", fontsize=8, color="red" if is_wrong else "black")
    save(fig, "fig_error_samples.png")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run folder name for the detailed analysis")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    runs = load_runs()
    if not runs:
        sys.exit("No runs with history.json found in resnet50/runs/")
    truth = read_labels(ANSWER_PATH) if ANSWER_PATH.exists() else None
    if truth is None:
        print(f"note: {ANSWER_PATH.name} not found, test metrics will be skipped.")
    for r in runs:
        r["stats"] = run_stats(r)
        sub = r["dir"] / "submission.csv"
        r["test"] = test_metrics(truth, read_labels(sub), r["dir"].name) if truth and sub.exists() else None
    runs.sort(key=lambda r: (r["label"] != "baseline", r["label"]))

    base = find(runs)
    seed_runs = varying(runs, "seed")
    vals = [r["stats"]["best_val_acc"] for r in seed_runs]
    std = statistics.stdev(vals) if len(vals) > 1 else None
    noise = std if std else 0.0014  # fall back to the binomial error of 5000 val images
    print(f"\nLoaded {len(runs)} runs from {RUNS_DIR}\n")

    # ---- validation table
    header = ["run", "folder", "best val %", "ep", "train %", "val %", "gap pp", "val loss",
              "s/epoch", "params M", "d vs base pp"]
    rows, csv_rows = [], []
    for r in runs:
        s = r["stats"]
        d = (s["best_val_acc"] - base["stats"]["best_val_acc"]) if base else None
        flag = ""
        if d is not None and r is not base and abs(d) <= 2 * noise:
            flag = " (~noise)"
        rows.append([r["label"], r["dir"].name, pct(s["best_val_acc"]), s["best_epoch"],
                     pct(s["final_train_acc"]), pct(s["final_val_acc"]), f"{100 * s['gap']:.2f}",
                     f"{s['final_val_loss']:.4f}",
                     "-" if s["sec_per_epoch"] is None else f"{s['sec_per_epoch']:.0f}",
                     "-" if not s["trainable"] else f"{s['trainable'] / 1e6:.1f}",
                     ("-" if d is None else f"{100 * d:+.2f}") + flag])
        t = r["test"] or {}
        csv_rows.append(dict(
            run=r["label"], folder=r["dir"].name, best_val_acc=s["best_val_acc"], best_epoch=s["best_epoch"],
            final_train_acc=s["final_train_acc"], final_val_acc=s["final_val_acc"], gap=s["gap"],
            final_val_loss=s["final_val_loss"], sec_per_epoch=s["sec_per_epoch"], trainable_params=s["trainable"],
            test_acc=t.get("acc"), test_precision=t.get("precision"), test_recall=t.get("recall"),
            test_f1=t.get("f1"), test_specificity=t.get("specificity"),
            test_tp=t.get("tp"), test_tn=t.get("tn"), test_fp=t.get("fp"), test_fn=t.get("fn")))
    print("VALIDATION (from history.json). '~noise' = difference within 2 std of the seed-to-seed spread")
    print_table(header, rows)
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)

    # ---- test table
    if truth:
        print("\nTEST SET (submission.csv vs test_answer.csv; positive class = dog)")
        trows = []
        for r in runs:
            t = r["test"]
            if t:
                trows.append([r["label"], pct(t["acc"]), pct(t["precision"]), pct(t["recall"]),
                              pct(t["f1"]), pct(t["specificity"]), t["fp"], t["fn"], t["tp"] + t["tn"]])
        print_table(["run", "acc %", "precision %", "recall %", "F1 %", "cat recall %", "FP", "FN", "correct"],
                    trows)
        print("FP = cat predicted as dog, FN = dog predicted as cat, cat recall = TN / (TN + FP)")

    # ---- seeds
    if len(seed_runs) > 1:
        print(f"\nREPRODUCIBILITY: baseline settings with {len(seed_runs)} seeds "
              f"({', '.join(str(r['cfg']['seed']) for r in sorted(seed_runs, key=lambda r: r['cfg']['seed']))})")
        out = [["metric", "mean", "std", "min", "max"]]
        series = {"val_acc": vals}
        if all(r["test"] for r in seed_runs):
            series["test_acc"] = [r["test"]["acc"] for r in seed_runs]
            series["test_f1"] = [r["test"]["f1"] for r in seed_runs]
        for k, v in series.items():
            out.append([k, 100 * statistics.mean(v), 100 * statistics.stdev(v), 100 * min(v), 100 * max(v)])
        print_table(out[0], [[r[0]] + [f"{x:.3f}" for x in r[1:]] for r in out[1:]])
        with open(RESULTS_DIR / "seeds.csv", "w", newline="") as f:
            csv.writer(f).writerows(out)
        print("(values in %; std is the sample standard deviation)")
    else:
        print("\nOnly one baseline seed found: run --seed 1 and --seed 2 to measure the noise.")

    # ---- figures
    make_figures(runs)

    # ---- detailed analysis of one run
    if args.run:
        chosen = next((r for r in runs if r["dir"].name == args.run), None)
        if chosen is None:
            sys.exit(f"Run '{args.run}' not found. Available: {[r['dir'].name for r in runs]}")
    else:
        chosen = max(runs, key=lambda r: r["stats"]["best_val_acc"])
    print(f"\nSelected run (highest validation accuracy): {chosen['dir'].name}  [{chosen['label']}]")
    print(f"To use it as the final submission (from the repo root):  "
          f"cp \"resnet50/runs/{chosen['dir'].name}/submission.csv\" submission.csv")
    if truth and chosen["test"]:
        confusion_figure(chosen, "fig_confusion_matrix.png")
        error_analysis(chosen, truth, not args.no_images)


if __name__ == "__main__":
    main()