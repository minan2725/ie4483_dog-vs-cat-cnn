# ConvNeXt-Tiny for Dogs vs. Cats

This folder contains the second Dogs vs. Cats classifier. Run all commands from
the repository root. CIFAR-10 is outside this model's scope.

## Setup

Use Python 3.12 (the system Python 3.9 is too old for the pinned packages):

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r convnext/requirements.txt
.venv/bin/python -c 'import torch; print(torch.backends.mps.is_available())'
```

The [assignment's dataset archive](https://drive.google.com/file/d/1q0r6yeHQMS17R3wz-s2FIbMR5DAGZK5v/view?usp=sharing)
contains `datasets/train/{cat,dog}` (10,000 each), `datasets/val/{cat,dog}`
(2,500 each), and `datasets/test` (500 flat images). Extract `datasets/` at
the repository root. It is ignored by Git. Shared preprocessing removes the
19 train images listed in `duplicate_pairs.csv` that overlap validation or
test, leaving 19,981 training images (9,987 cats; 9,994 dogs). All 5,000
validation images remain separate from training.

## Model and preprocessing

TorchVision ConvNeXt-Tiny starts from ImageNet-1K weights and replaces its last
linear classifier with a two-class layer. The classifier produces two logits:
index 0 is cat and index 1 is dog. Input is a 3 x 224 x 224 RGB tensor.

```text
RGB image -> shared resize/crop/normalization -> ConvNeXt-Tiny features
          -> global average pool -> layer norm -> 768-to-2 linear head
          -> cat/dog logits
```

The shared `preprocessing.py` applies random crop, flip, rotation, and color
jitter during training, and deterministic resize/center crop for validation
and test. Validation/test resize to 256 before cropping to 224, matching
ResNet50; TorchVision's ConvNeXt-Tiny weight preset uses resize 236 instead.
The model uses cross-entropy loss, AdamW (weight decay 1e-4), and cosine
learning-rate decay. All layers are fine-tuned unless `--freeze-backbone` is
set. The checkpoint with the highest validation accuracy is saved as `best.pt`.

## Previous 10,000-image results

| Run | Best validation accuracy | Best epoch | Mean seconds/epoch |
| --- | ---: | ---: | ---: |
| 1e-4, seed 42 | 99.50% | 4 | 158 |
| 1e-4, seed 1 | 99.50% | 5 | 161 |
| 3e-4, seed 42 | 99.30% | 5 | 157 |

These runs used 5,000 images per class, not the complete clean training split.
The two baseline seeds average 99.50% validation accuracy. This is a small
repeatability check, not a robust estimate of run-to-run variance. The
validation-selected model is the 1e-4, seed-42 run: it ties on accuracy with
seed 1 and has lower best-epoch validation loss (0.01521 vs. 0.01568). It
scores 99.60% (498/500) on the final test set: precision 100%, recall 99.23%,
F1 99.61%, with 0 false positives and 2 false negatives for the dog class.
The higher learning rate performs worse on validation, so 1e-4 is retained.
At matching seeds, the existing ResNet50 baseline scores 99.26% (seed 42) and
99.36% (seed 1) on validation, compared with ConvNeXt's 99.50% for both.
These small differences should not be presented as a definitive model ranking.

Both false negatives (test IDs 328 and 464) are visually similar "No Photo
Available" dog-silhouette placeholders. They illustrate a limitation on
non-photographic inputs; they are not evidence that the classifier struggles
with ordinary dog photos. A correctly classified example is ID 242, a dark
cat against a dark background. These observations come from inspecting the
images and should be phrased as qualitative examples in the group report.

## Full-data training

The current trainer uses all 19,981 clean training images and all 5,000
validation images. The default recipe uses 224-pixel inputs, 10 epochs,
batch size 32, learning rate 1e-4, and seed 42:

```sh
.venv/bin/python convnext/train_convnext.py --seed 42
```

The entry point now uses the shared `dogcat_training.py` pipeline, with the
same CLI and run format as ResNet50 and VGG16. Use `--train-per-class` and
`--val-per-class` for a labelled subset run, or `--predict-test` to write a
submission after training. An existing run directory requires a different
`--out-dir` or an explicit `--overwrite`.

The run folder is `convnext/runs/convnext_tiny_lr0.0001_n19981_e10_s42`.
Each run writes `history.json` and an ignored `best.pt` in its own folder.
To use a trained checkpoint on another machine, transfer `best.pt`
separately or reproduce the run; Git will include the metrics but not weights.
The older `compare_runs.py` tables cover only the 5,000-per-class experiments;
they do not include the full-data run. Results may vary slightly on MPS even
with a fixed seed.

The 10-epoch full-data seed-42 run selected epoch 10 by validation accuracy:
99.60% (4,980/5,000), with validation loss 0.0180 and a mean epoch time of
286 seconds on MPS. Its 500-image test submission scored 99.40% (497/500;
zero false positives and three false negatives). The earlier five-epoch
full-data run selected epoch 2 at 99.56% validation accuracy and also scored
99.40% on test, while the 10,000-image seed-42 run scored 99.50% on validation
and 99.60% on test. One run per setting cannot establish that more data or
more epochs helped or hurt; test scores must not be used to tune or select a
checkpoint.

## Final prediction and report

Choose one run by validation accuracy (validation loss breaks ties), then
predict the test set once:

```sh
.venv/bin/python convnext/predict_convnext.py --run convnext_tiny_lr0.0001_n19981_e10_s42 --evaluate
```

Replace the run name with the actual validation-selected run. The command
checks all 500 image IDs against `test_answer.csv`, writes `submission.csv`
(`id,label`, with dog=1), `test_probs.csv`, `final_metrics.json`, and
`error_analysis.csv` to that run folder. `--evaluate` reads labels only for
this final evaluation. Review the listed image paths yourself before
describing correct and incorrect cases in the report. Do not use test metrics
to choose hyperparameters or checkpoints. Report the data counts, training
settings, architecture, validation accuracy, and final test examples. This
full-data result has one seed, so it does not estimate run-to-run variation.
Cite TorchVision and the ConvNeXt paper.

References: [TorchVision ConvNeXt-Tiny](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.convnext_tiny.html),
[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545).

## Files

- `train_convnext.py`: train and validate one experiment.
- `compare_runs.py`: compare validation experiments and ResNet50 baselines.
- `predict_convnext.py`: generate one final submission and optional test analysis.
- `requirements.txt`: tested package versions for Python 3.12 on Apple Silicon.
