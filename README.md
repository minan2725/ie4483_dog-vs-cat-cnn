# Dogs vs. Cats image classification (CNN) - group project

Classify images as dog or cat with convolutional neural networks. Each model lives in its own folder; the data preprocessing and the test answers (ground truth) are shared.

## Repository layout

```
preprocessing.py    shared data pipeline (everyone imports this file)
test_answer.csv     ground-truth labels of the 500 test images (final evaluation only, not for tuning)
datasets/           the images (NOT in git)
resnet50/           ResNet50 classifier: code, runs/, results/ and its own README (best weights are described there)
convnext/           ConvNeXt-Tiny classifier: code, experiments, results, and README
cifar10/            CIFAR-10 part
```

## Conventions (so that results can be compared fairly)

- Use `preprocessing.py` instead of writing a new data pipeline. If you need a change, add an optional parameter rather than changing the default behaviour, because every model depends on it.
- Labels: `cat = 0`, `dog = 1`.
- Baseline settings for comparisons: 5000 training images per class, full validation set, 224x224 input, 5 epochs, seeds 42 / 1 / 2.
- `.pt` weight files are in `.gitignore`
