# VGG16 for Dogs vs. Cats

The current `train_vgg16.py` entry point uses the shared `dogcat_training.py`
pipeline. From the repository root, `.venv/bin/python vgg16/train_vgg16.py` trains on
all 19,981 clean training images and validates on all 5,000 validation images
with the common five-epoch settings. Use `--train-per-class` and
`--val-per-class` for labelled subset experiments, or `--predict-test` to
write a submission after training. On MPS, use the reference 224-pixel input;
Torch's VGG adaptive pool may not support smaller feature-map sizes.

## Historical configurations
vgg16_lr0.0001_n1000_e10_s42
vgg16_lr0.0001_n1000_e10_s42_noaug --> overfitted
vgg16_lr0.0001_n2500_e10_s42
vgg16_lr0.0001_n5000_e10_s1
vgg16_lr0.0001_n5000_e10_s2 --> best
vgg16_lr0.0001_n5000_e10_s42
vgg16_lr0.0001_n5000_e10_s42_frozen
vgg16_lr0.0001_n5000_e10_s42_scratch
vgg16_lr0.0001_n5000_e10_s42_no_aug
vgg16_lr0.001_n5000_e10_s42
vgg16_lr1e-05_n5000_e10_s42
