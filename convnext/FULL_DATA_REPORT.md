# ConvNeXt-Tiny full-data run

## Setup

- Model: TorchVision ConvNeXt-Tiny with ImageNet-1K weights and a new two-logit head (`cat=0`, `dog=1`); all layers fine-tuned.
- Data: 19,981 training images (9,987 cats, 9,994 dogs). Shared preprocessing excludes 19 training images that overlap validation or test, as listed in `duplicate_pairs.csv`. All 5,000 validation images remain separate.
- Inputs: RGB, 224 x 224. Training uses random resized crop, horizontal flip, rotation, and color jitter; validation and test use resize to 256 and center crop. All use ImageNet normalization.
- Optimization: cross-entropy, AdamW (`lr=1e-4`, `weight_decay=1e-4`), cosine decay, batch size 32, five epochs, seed 42. Device: Apple MPS; PyTorch 2.13.0.
- Selection: checkpoint with the highest validation accuracy. Test labels are not used for selection.

## Results

| Epoch | Train accuracy | Validation accuracy | Validation loss | Seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 98.60% | 99.26% | 0.0216 | 297 |
| 2 | 99.39% | **99.56%** | **0.0136** | 298 |
| 3 | 99.75% | 99.32% | 0.0195 | 283 |
| 4 | 99.91% | 99.44% | 0.0181 | 281 |
| 5 | 99.95% | 99.48% | 0.0161 | 288 |

Epoch 2 is the saved checkpoint: 4,978/5,000 validation images correct. Its submission contains all 500 test IDs and scores 497/500 (99.40%) against `test_answer.csv`: 240 true negatives, 257 true positives, one false positive, and two false negatives. The misclassified test IDs are 242, 313, and 464; their predictions and probabilities are in `error_analysis.csv`.

The older seed-42 ConvNeXt run used 10,000 training images and scored 99.50% on validation and 99.60% on test. The full-data run scored slightly higher on validation and slightly lower on test. These are single-seed results, so the difference does not establish whether more training data helped or hurt. Test performance was measured only after choosing the epoch by validation accuracy.

## Reproduce

From the repository root, with the assignment images under `datasets/`:

```sh
.venv/bin/python convnext/train_convnext.py --seed 42 --epochs 5 --lr 1e-4 --batch-size 32
.venv/bin/python convnext/predict_convnext.py --run convnext_tiny_lr0.0001_n19981_e5_s42 --evaluate
```

Run records, submission, probabilities, and error analysis are under `convnext/runs/convnext_tiny_lr0.0001_n19981_e5_s42/`. The 106 MB `best.pt` checkpoint is ignored by Git and must be reproduced or transferred separately. Do not use `test_answer.csv` to tune another run.
