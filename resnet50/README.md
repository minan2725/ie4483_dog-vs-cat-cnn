# ResNet50 results

Reproduce all tables and figures: `python resnet50/compare_runs.py`

## Setup

- ResNet50 pretrained on ImageNet, all layers fine-tuned (scratch run: random start)
- 19,981 train, 5,000 val, 500 test images, 224 x 224
- AdamW (weight decay 1e-4), cosine LR schedule, batch size 32, seed 42 unless stated
- 10 epochs (only scratch run: 20 epochs)
- Saved model = the epoch with the highest val accuracy

## All runs

| Run                              | Best val % (epoch)  | Last-epoch val % | Last-epoch val loss | Train-val gap (pp) |
| -------------------------------- | ------------------- | ---------------- | ------------------- | ------------------ |
| lr 1e-5 (baseline)               | **99.50** (9) | 99.42            | 0.0173              | 0.34               |
| lr 1e-4                          | 99.40 (8)           | 99.38            | 0.0195              | 0.57               |
| lr 1e-3                          | 98.50 (10)          | 98.50            | 0.0427              | 0.98               |
| lr 1e-5, no augmentation         | 99.40 (5)           | 99.34            | 0.0210              | 0.60               |
| lr 1e-5, seed 1                  | 99.46 (6)           | 99.38            | 0.0154              | 0.34               |
| lr 1e-5, seed 2                  | 99.44 (7)           | 99.34            | 0.0173              | 0.42               |
| From scratch, lr 1e-3, 20 epochs | 96.28 (20)          | 96.28            | 0.1043              | 0.25               |

- Best val = the best epoch out of 10. This is the model we keep.
- Last-epoch val = epoch 10.
- Gap = train accuracy minus val accuracy at the last epoch.
- Noise: the 3 seeds differ by 0.06 pp (3 images). Differences near 0.1 pp (5 images) are small.

## 1. Learning rate

Figure: `results/fig_learning_rate.png`

**Conclusion: use lr 1e-5.**

- lr 1e-3 is the worst (98.50%). After epoch 1 its val accuracy is only 92.56%, while the others are above 99%. A big learning rate damages the pretrained weights.
- lr 1e-4 and 1e-5 have almost the same accuracy (5 images apart). lr 1e-5 has a lower val loss, a smaller gap and a smoother curve (lr 1e-4 dips at epoch 4).
- lr 1e-4 and 1e-3 have one seed only.

## 2. Augmentation

Figure: `results/fig_augmentation.png`

**Conclusion: keep augmentation.**

- Without augmentation the model overfits. Val loss is lowest at epoch 5 (0.0197) and then goes up (0.0210), while train loss keeps going down.
- With augmentation, val loss keeps going down until epoch 9.
- The gap is bigger without augmentation (0.60 vs 0.34 pp).
- The accuracy difference is small (5 images). One seed only.

## 3. Seeds

Seeds 42, 1, 2 of the baseline. File: `results/seeds.csv`.

| Metric                  | Mean % | Std % |
| ----------------------- | ------ | ----- |
| Best val accuracy       | 99.47  | 0.03  |
| Last-epoch val accuracy | 99.38  | 0.04  |

**Conclusion: the baseline is stable. Final val result: 99.47 ± 0.03%.**

- lr 1e-3 and scratch are far below the noise level, so these differences are real.
- lr 1e-4 and no augmentation are 0.10 pp lower. This is just above 2 std (0.06 pp), so the evidence is weak.

## 4. Pretrained vs scratch

Figure: `results/fig_pretrained_vs_scratch.png`

| Run                 | Epochs | Best val % |
| ------------------- | ------ | ---------- |
| Pretrained, lr 1e-5 | 10     | 99.50      |
| Pretrained, lr 1e-3 | 10     | 98.50      |
| Scratch, lr 1e-3    | 20     | 96.28      |

**Conclusion: pretrained weights help a lot.**

- Scratch has twice the epochs and is still 3.22 pp lower (2.22 pp lower at the same lr 1e-3).
- Scratch is still improving at epoch 20 (+0.50 pp in the last 3 epochs). So the real gap at convergence is smaller than 3.22 pp.

## Final model and test result

Final model: `resnet50/runs/resnet50_lr1e-05_n19981_e10_s42/best.pt` (best val 99.50%, epoch 9). It was chosen by val accuracy only.

Test labels were used once, for this model only.

| Metric (dog = positive) | Value                      |
| ----------------------- | -------------------------- |
| Accuracy                | **100% (500/500)**        |
| Confusion counts        | TN 241, FP 0, FN 0, TP 259 |

Figures: `results/fig_confusion_matrix.png`, `results/fig_error_samples.png`.

There are no wrong images, so `fig_error_samples.png` shows the 8 correct images with the lowest confidence. With 0 errors in 500 images, the true error rate could still be up to about 0.6%.

## Reproduce

```bash
python resnet50/train_resnet50.py --lr 1e-5 --seed 42
python resnet50/train_resnet50.py --lr 1e-4 --seed 42
python resnet50/train_resnet50.py --lr 1e-3 --seed 42
python resnet50/train_resnet50.py --lr 1e-5 --seed 42 --no-augment
python resnet50/train_resnet50.py --lr 1e-5 --seed 1
python resnet50/train_resnet50.py --lr 1e-5 --seed 2
python resnet50/train_resnet50.py --scratch --lr 1e-3 --epochs 20 --seed 42

python resnet50/compare_runs.py
python resnet50/predict_resnet50.py --run resnet50_lr1e-05_n19981_e10_s42 --evaluate
python resnet50/compare_runs.py   # run again to draw the test figures
```

## Reuse the weights

```python
import torch
from resnet50.train_resnet50 import build_model

model = build_model(scratch=True)  # only builds the architecture, best.pt has all weights
state = torch.load("resnet50/runs/resnet50_lr1e-05_n19981_e10_s42/best.pt",
                   map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
```

Labels: `cat = 0`, `dog = 1`. Use the same 224 x 224 eval transform as `preprocessing.py`.
