
from pathlib import Path
from itertools import combinations
from PIL import Image
import imagehash
import csv
import matplotlib.pyplot as plt
import math

DATA_DIR = Path(__file__).resolve().parent / "datasets"
SPLITS = ["train", "val", "test"]
THRESHOLD = 5

images = []

for split in SPLITS:
    for path in (DATA_DIR / split).rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            with Image.open(path) as img:
                phash = imagehash.phash(img.convert("RGB"))
            images.append({
                "split": split,
                "label": path.parent.name,
                "path": path,
                "hash": phash
            })
        except Exception as e:
            print(f"Cannot read {path}: {e}")

matches = []
groups = {
    s: [img for img in images if img["split"] == s]
    for s in SPLITS
}

for sa, sb in combinations(SPLITS, 2):
    for a in groups[sa]:
        for b in groups[sb]:
            d = a["hash"] - b["hash"]
            if d <= THRESHOLD:
                matches.append((a, b, d))

# Save CSV
with open("duplicate_pairs.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path_1", "path_2", "label_1", "label_2", "distance"])
    for a, b, d in matches:
        writer.writerow([
            a["path"].relative_to(DATA_DIR),
            b["path"].relative_to(DATA_DIR),
            a["label"], b["label"], d
        ])

# Create contact sheet: each pair is one SubFigure (title + 2 images)
if matches:
    cols = 5
    rows = math.ceil(len(matches) / cols)

    fig = plt.figure(figsize=(cols * 5.0, rows * 3.5), facecolor="white")

    # Container leaves room at the top for the main title
    gs = fig.add_gridspec(1, 1, left=0.01, right=0.99, bottom=0.01, top=0.94)
    container = fig.add_subfigure(gs[0])

    # hspace / wspace here = gap BETWEEN pairs
    subfigs = container.subfigures(
        rows, cols, hspace=0.06, wspace=0.03, squeeze=False
    )

    for i, (a, b, d) in enumerate(matches):
        r, c = divmod(i, cols)
        sf = subfigs[r, c]

        # Light border around each pair
        sf.patch.set_facecolor("#fafafa")
        sf.patch.set_edgecolor("#cccccc")
        sf.patch.set_linewidth(1.2)

        # Pair title sits right above the image titles
        sf.suptitle(
            f"PAIR {i + 1} | Distance: {d}",
            fontsize=13, fontweight="bold", y=0.90
        )

        # wspace here = gap WITHIN a pair (between the 2 images)
        sf.subplots_adjust(left=0.03, right=0.95, top=0.70, bottom=0.04, wspace=0.08)
        axs = sf.subplots(1, 2)

        for ax, item in zip(axs, (a, b)):
            with Image.open(item["path"]) as img:
                ax.imshow(img.convert("RGB"))
            split = item["path"].relative_to(DATA_DIR).parts[0]
            label = item["label"].upper()
            path = item["path"].relative_to(DATA_DIR).as_posix()
            color = "red" if split == "train" else "darkgreen"
            ax.set_title(
                f"{split.upper()} | {label}\n{path}",
                fontsize=9, fontweight="bold", color=color, pad=6
            )
            ax.axis("off")

    plt.savefig(
        "all_duplicate_pairs.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close()

print(f"Candidate pairs: {len(matches)}")
print("Saved: all_duplicate_pairs.png")
