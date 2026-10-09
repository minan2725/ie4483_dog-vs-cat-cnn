# Dogs vs. Cats image classification (CNN)

Group project: classify images as dog or cat with convolutional neural networks.
This repository contains the shared data pipeline, the ResNet50 classifier and the scripts used to compare experiments.

## Files

| File                  | Purpose                                                                                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `preprocessing.py`  | Shared data pipeline: dataset path (`DATA_DIR`), subset selection, augmentation, normalisation, data loaders, test-set reader. **Everyone imports this file** so that all models see identical data. |
| `train_resnet50.py` | Fine-tunes / trains ResNet50, writes one folder per run in`runs/`                                                                                                                                          |
| `compare_runs.py`   | Reads all runs, prints validation and test tables, draws figures into`results/`                                                                                                                            |
| `test_answer.csv`   | Ground-truth labels of the 500 test images (used only for final evaluation, never for tuning)                                                                                                                |
| `runs/`             | One folder per experiment:`history.json`, `submission.csv`, `test_probs.csv` (weights `*.pt` are not tracked)                                                                                        |
| `results/`          | Tables and figures produced by`compare_runs.py`                                                                                                                                                            |

## Dataset

Place the dataset next to the Python files:

```
datasets/
  train/cat   train/dog
  val/cat     val/dog
  test/       (500 unlabelled images)
```

If the folder has a different name or location, change `DATA_DIR` at the top of `preprocessing.py`.

## Run

```
# quick check that everything works (about 1 minute)
python3 train_resnet50.py --train-per-class 200 --val-per-class 100 --epochs 1 --out-dir runs/smoke_test

# baseline: pretrained ResNet50, 5000 images per class, 5 epochs, lr 1e-4, seed 42
python3 train_resnet50.py

# examples of the experiments
python3 train_resnet50.py --seed 1
python3 train_resnet50.py --lr 1e-3
python3 train_resnet50.py --freeze-backbone --lr 1e-3
python3 train_resnet50.py --no-augment
python3 train_resnet50.py --train-per-class 1000
python3 train_resnet50.py --img-size 128
python3 train_resnet50.py --scratch --lr 1e-3 --epochs 10

# compare every run in runs/ (needs test_answer.csv for the test metrics)
python3 compare_runs.py
```

Each run is saved in `runs/<settings>/`
Labels follow the brief: `cat = 0`, `dog = 1`.
