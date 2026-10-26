import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dogcat_training import default_run_name, parse_training_args, predict_test, train_model
from preprocessing import class_balanced_indices


class SmallDataset(Dataset):
    class_to_idx = {"cat": 0, "dog": 1}

    def __init__(self):
        self.images = torch.randn(4, 3, 4, 4)
        self.labels = torch.tensor([0, 0, 1, 1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class SmallTestDataset(Dataset):
    paths = [Path(f"{i}.jpg") for i in range(1, 501)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return torch.zeros(3, 4, 4), self.paths[index].name


class TrainingPipelineTests(unittest.TestCase):
    def test_fixed_balanced_subset_and_full_default(self):
        class Folder:
            samples = [(str(i), i // 4) for i in range(8)]

        eligible = [0, 1, 2, 4, 5, 6]
        first = class_balanced_indices(Folder(), eligible, per_class=2)
        self.assertEqual(first, class_balanced_indices(Folder(), eligible, per_class=2))
        self.assertEqual([Folder.samples[i][1] for i in first], [0, 0, 1, 1])
        self.assertEqual(class_balanced_indices(Folder(), eligible, per_class=0), eligible)
        with self.assertRaises(ValueError):
            class_balanced_indices(Folder(), eligible, per_class=4)

    def test_common_defaults_and_run_name(self):
        args = parse_training_args("convnext_tiny", [])
        self.assertEqual((args.train_per_class, args.val_per_class, args.seed), (0, 0, 42))
        self.assertEqual(default_run_name("convnext_tiny", args, 19981),
                         "convnext_tiny_lr0.0001_n19981_e5_s42")

    def test_train_writes_reloadable_checkpoint_and_history(self):
        with tempfile.TemporaryDirectory() as temp:
            args = parse_training_args("tiny", ["--epochs", "1", "--batch-size", "2",
                                                "--num-workers", "0", "--train-per-class", "2",
                                                "--val-per-class", "2", "--out-dir", temp])
            dataset = SmallDataset()
            loader = DataLoader(dataset, batch_size=2)
            with patch("dogcat_training.get_loaders", return_value=(loader, loader)), \
                 patch("dogcat_training.get_device", return_value=torch.device("cpu")):
                out_dir = train_model(
                    "tiny", lambda _: nn.Sequential(nn.Flatten(), nn.Linear(48, 2)), args)
            history = json.loads((out_dir / "history.json").read_text())
            self.assertTrue(history["completed"])
            self.assertEqual((history["train_images"], history["val_images"]), (4, 4))
            self.assertEqual(history["best_epoch"], 1)
            self.assertEqual(len(history["history"]), 1)
            model = nn.Sequential(nn.Flatten(), nn.Linear(48, 2))
            model.load_state_dict(torch.load(Path(temp) / "best.pt", weights_only=True))
            self.assertEqual(tuple(model(dataset[0][0].unsqueeze(0)).shape), (1, 2))

    def test_prediction_writes_500_unique_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            args = parse_training_args("tiny", ["--batch-size", "50", "--num-workers", "0"])
            model = nn.Sequential(nn.Flatten(), nn.Linear(48, 2))
            with patch("dogcat_training.TestImages", return_value=SmallTestDataset()):
                predict_test(model, torch.device("cpu"), args, Path(temp))
            with (Path(temp) / "submission.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 500)
            self.assertEqual(len({row["id"] for row in rows}), 500)
            self.assertEqual(set(rows[0]), {"id", "label"})

    def test_validation_loss_breaks_accuracy_tie(self):
        with tempfile.TemporaryDirectory() as temp:
            args = parse_training_args("tiny", ["--epochs", "2", "--train-per-class", "2",
                                                "--val-per-class", "2", "--num-workers", "0",
                                                "--out-dir", temp])
            loader = DataLoader(SmallDataset(), batch_size=2)
            scores = [(0.5, 0.5), (0.2, 0.75), (0.4, 0.5), (0.1, 0.75)]
            with patch("dogcat_training.get_loaders", return_value=(loader, loader)), \
                 patch("dogcat_training.get_device", return_value=torch.device("cpu")), \
                 patch("dogcat_training.torch.optim.lr_scheduler.CosineAnnealingLR"), \
                 patch("dogcat_training.run_epoch", side_effect=scores):
                out_dir = train_model(
                    "tiny", lambda _: nn.Sequential(nn.Flatten(), nn.Linear(48, 2)), args)
            history = json.loads((out_dir / "history.json").read_text())
            self.assertEqual(history["best_epoch"], 2)
            self.assertEqual(history["best_val_acc"], 0.75)


if __name__ == "__main__":
    unittest.main()
