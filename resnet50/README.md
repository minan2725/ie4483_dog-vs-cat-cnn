# ResNet50 classifier (dogs vs. cats)

ImageNet-pretrained ResNet50 whose last layer is replaced by a 2-class layer and fine-tuned on 5,000 images per class.
All commands below are run from the **repository root**.

## Best model (use this one)

| Item                              | Value                                                                                            |
| --------------------------------- | ------------------------------------------------------------------------------------------------ |
| Run folder                        | `resnet50/runs/resnet50_lr0.0001_n5000_e5_s1/` (seed 1)                                        |
| Validation accuracy               | **99.36 %** (5,000 validation images)                                                      |
| Same settings, 3 seeds (42, 1, 2) | 99.32 +/- 0.05 % validation, see`resnet50/results/seeds.csv`                                   |
| Test accuracy                     | 99.40 % (497 / 500 images), precision 99.61 %, recall 99.23 %, F1 99.42 % (dog = positive class) |
| Weights file                      | `resnet50/runs/resnet50_lr0.0001_n5000_e5_s1/best.pt` (about 95 MB)                            |
| Predictions                       | `resnet50/runs/resnet50_lr0.0001_n5000_e5_s1/submission.csv`                                   |

Training settings: pretrained weights (`IMAGENET1K_V2`), all layers fine-tuned, AdamW, learning rate 1e-4 with cosine decay, weight decay 1e-4, batch size 32, 5 epochs, 224x224 input, augmentation on (random crop, flip, rotation, colour jitter), ImageNet normalisation.

## Load the weights

```python
import sys
import torch
sys.path.insert(0, "resnet50")
from train_resnet50 import build_model

# scratch=True only builds the architecture without downloading ImageNet weights;
# the trained weights are loaded from the file right after.
model = build_model(freeze_backbone=False, scratch=True)
model.load_state_dict(torch.load("resnet50_seed1.pt", map_location="cpu"))
model.eval()
```

Input: tensor of shape (N, 3, 224, 224) preprocessed with `preprocessing.get_transforms()[1]` (resize 256, centre-crop 224, ImageNet normalisation).
Output: 2 logits, index 0 = cat, index 1 = dog (apply softmax for probabilities).

## Files and results in this folder

| Path                  | Purpose                                                                                                                                                                                      |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_resnet50.py` | Trains one run, writes`runs/<settings>/` with `history.json`, `submission.csv`, `test_probs.csv` (weights `best.pt` are ignored by git)                                            |
| `compare_runs.py`   | Compares all runs, writes tables and figures to`results/` (needs `test_answer.csv` at the repo root). Tied to the settings of `train_resnet50.py`; copy and adapt it for another model |
| `runs/`             | One folder per experiment (12 runs: seeds, learning rate, frozen backbone, augmentation, data size, image size, from scratch)                                                                |
| `results/`          | `summary.csv` (all runs), `seeds.csv` (mean and std over seeds), `error_analysis.csv`, figures `fig_*.png`                                                                           |

### Reproduce

```
python3 resnet50/train_resnet50.py              # baseline, seed 42
python3 resnet50/train_resnet50.py --seed 1
python3 resnet50/compare_runs.py                # tables and figures
```
