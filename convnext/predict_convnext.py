"""Predict the 500 test images for one validation-selected ConvNeXt run."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from preprocessing import DATA_DIR, TestImages, get_device  # noqa: E402
from train_convnext import build_model  # noqa: E402


def normalize_id(value):
    value = Path(str(value).strip()).stem
    return str(int(value)) if value.isdigit() else value


def expected_ids(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"Missing id column in {path}")
        ids = [normalize_id(row["id"]) for row in reader]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError(f"Expected 500 distinct answer IDs in {path}; found {len(ids)}")
    return set(ids)


def read_answers(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"id", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"Missing id or label column in {path}")
        answers = {normalize_id(row["id"]): int(row["label"]) for row in reader}
    if len(answers) != 500 or set(answers.values()) != {0, 1}:
        raise ValueError("Test answers must have 500 distinct IDs and labels 0/1")
    return answers


def write_csv(path, header, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def evaluate(rows, answers, image_paths, out_dir):
    tp = tn = fp = fn = 0
    wrong = []
    correct = []
    for image_id, predicted, prob_cat, prob_dog in rows:
        true = answers[normalize_id(image_id)]
        if true == 1 and predicted == 1:
            tp += 1
        elif true == 0 and predicted == 0:
            tn += 1
        elif true == 0:
            fp += 1
        else:
            fn += 1
        example = (image_id, image_paths[normalize_id(image_id)], true, predicted,
                   prob_cat, prob_dog, max(prob_cat, prob_dog))
        (correct if predicted == true else wrong).append(example)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    metrics = dict(n=len(rows), accuracy=(tp + tn) / len(rows),
                   precision=precision, recall=recall,
                   f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                   tp=tp, tn=tn, fp=fp, fn=fn)
    (out_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    wrong.sort(key=lambda item: item[-1], reverse=True)
    correct.sort(key=lambda item: item[-1])
    examples = [("wrong", row) for row in wrong] + [
        ("correct_low_confidence", row) for row in correct[:8]]
    write_csv(out_dir / "error_analysis.csv",
              ["status", "id", "image_path", "true", "pred", "prob_cat", "prob_dog"],
              [(status, *row[:-1]) for status, row in examples])
    print(f"Final test accuracy: {metrics['accuracy']:.2%} ({tp + tn}/{len(rows)})")
    print(f"Confusion counts: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Review {out_dir / 'error_analysis.csv'} for report examples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run folder name or path")
    parser.add_argument("--evaluate", action="store_true",
                        help="read test_answer.csv for final metrics and error analysis")
    args = parser.parse_args()
    run_dir = Path(args.run)
    if not run_dir.exists():
        run_dir = HERE / "runs" / args.run
    history_path = run_dir / "history.json"
    if not history_path.exists() or not (run_dir / "best.pt").exists():
        parser.error(f"Completed run and best.pt required in {run_dir}")
    history = json.loads(history_path.read_text())
    if not history.get("completed"):
        parser.error(f"Run is incomplete: {run_dir}")
    answer_path = ROOT / "test_answer.csv"
    test_dir = DATA_DIR / "test"
    if not answer_path.exists() or not test_dir.exists():
        parser.error("test_answer.csv and datasets/test are required to verify test IDs")

    dataset = TestImages(test_dir)
    if len(dataset) != 500:
        parser.error(f"Expected 500 test images; found {len(dataset)}")
    ids = [normalize_id(path.stem) for path in dataset.paths]
    if len(set(ids)) != 500 or set(ids) != expected_ids(answer_path):
        parser.error("Test filenames do not match the 500 expected IDs")
    image_paths = {normalize_id(path.stem): str(path.relative_to(ROOT)) for path in dataset.paths}

    device = get_device()
    model = build_model(pretrained=False).to(device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    config = history["args"]
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"])
    rows = []
    with torch.inference_mode():
        for images, names in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
            for name, probs in zip(names, probabilities):
                rows.append((Path(name).stem, int(probs.argmax()),
                             float(probs[0]), float(probs[1])))
    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 predictions; got {len(rows)}")
    write_csv(run_dir / "submission.csv", ["id", "label"],
              [(image_id, predicted) for image_id, predicted, _, _ in rows])
    write_csv(run_dir / "test_probs.csv", ["id", "pred", "prob_cat", "prob_dog"], rows)
    print(f"Wrote 500 predictions to {run_dir / 'submission.csv'}")
    if args.evaluate:
        evaluate(rows, read_answers(answer_path), image_paths, run_dir)


if __name__ == "__main__":
    main()
