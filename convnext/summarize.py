from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs-cifar10"
RESULTS_DIR = HERE / "results-cifar10"
GROUND_TRUTH = HERE.parent / "cifar_ground_truth.csv"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
CONVNEXT_RESULT = RESULTS_DIR / "test_results.csv"
CONVNEXT_SUMMARY = RESULTS_DIR / "test_summary.csv"

if not CONFUSION_DIR.exists():
    CONFUSION_DIR.mkdir(parents=True)

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Read ground truth
    gt = pd.read_csv(GROUND_TRUTH)
    gt["filename"] = gt["filename"].astype(str)
    gt["label"] = gt["label"].astype(int)

    # Store filename, ground-truth label, and model predictions
    results = gt[["filename", "label"]].rename(columns={"label": "ground_truth"})
    summary = []
    class_labels = sorted(gt["ground_truth"].unique()) if "ground_truth" in gt else sorted(gt["label"].unique())

    # Read predictions from every subfolder
    for submission_path in sorted(RUNS_DIR.glob("*/submission.csv")):
        model_name = submission_path.parent.name
        submission = pd.read_csv(submission_path)

        if not {"id", "label"}.issubset(submission.columns):
            print(f"Skipping {model_name}: missing 'id' or 'label' column")
            continue

        submission["id"] = submission["id"].astype(str)
        submission["label"] = submission["label"].astype(int)

        # Join submission id with ground-truth filename
        predictions = gt[["filename"]].merge(
            submission[["id", "label"]],
            left_on="filename",
            right_on="id",
            how="left",
            validate="one_to_one"
        )

        if predictions["label"].isna().any():
            missing = predictions["label"].isna().sum()
            print(f"Warning: {model_name} has {missing} missing predictions")

        # Add model predictions as a new column
        results[model_name] = predictions["label"].values

        # Evaluate only samples with valid predictions
        valid = predictions["label"].notna()
        y_true = gt.loc[valid, "label"].to_numpy()
        y_pred = predictions.loc[valid, "label"].astype(int).to_numpy()

        if len(y_true) == 0:
            continue

        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Per-class accuracy: correct predictions / ground-truth samples per class
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        class_accuracies = [
            correct / total if total else 0.0
            for correct, total in zip(cm.diagonal(), cm.sum(axis=1))
        ]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_labels
        )
        display.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Ground Truth Label")
        fig.tight_layout()

        fig.savefig(CONFUSION_DIR / f"{model_name}_confusion_matrix.png", dpi=300)
        plt.close(fig)

        row = {
            "model": model_name,
            "total_accuracy": accuracy,
            "macro_f1": macro_f1,
            "num_samples": len(y_true),
        }
        row.update({
            f"class_{label}_accuracy": class_acc
            for label, class_acc in zip(class_labels, class_accuracies)
        })
        summary.append(row)

    # Save combined predictions and evaluation summary
    results.to_csv(CONVNEXT_RESULT, index=False)
    pd.DataFrame(summary).to_csv(CONVNEXT_SUMMARY, index=False)

    print(f"Predictions saved to: {CONVNEXT_RESULT}")
    print(f"Summary saved to: {CONVNEXT_SUMMARY}")


if __name__ == "__main__":
    main()