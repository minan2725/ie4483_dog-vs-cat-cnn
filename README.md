# Dogs vs. Cats image classification (CNN) - group project

Classify images as dog or cat with convolutional neural networks. Each model lives in its own folder; the data preprocessing and the test answers (ground truth) are shared.

## Repository layout

```
preprocessing.py    shared data pipeline (everyone imports this file)
test_answer.csv     ground-truth labels of the 500 test images (final evaluation only, not for tuning)
datasets/           the images (NOT in git)
resnet50/           ResNet50 classifier: code, runs/, results/ and its own README (best weights are described there)
convnext/           ConvNeXt-Tiny classifier: code, experiments, results, and full-data report
vgg16/              VGG16 classifier: code, experiments, results, and README
prepare_cifar10.py  CIFAR-10 dataset preparation (separate task)
```

## Conventions (so that results can be compared fairly)

- Use `preprocessing.py` for Dogs vs. Cats. Coordinate changes to its loader interface with every model script that calls it; the current interface returns all clean training images and all validation images.
- Labels: `cat = 0`, `dog = 1`.
- The proposed full-data reference settings are in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). Existing 5000-per-class runs are historical subset experiments, not full-data runs.
- The latest ConvNeXt training result is in [convnext/FULL_DATA_REPORT.md](convnext/FULL_DATA_REPORT.md).
- `.pt` weight files are in `.gitignore`
