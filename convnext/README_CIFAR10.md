# CIFAR-10 Image Classification

## Overview

This project implements CIFAR-10 image classification using PyTorch, with support for standard training and experiments on imbalanced datasets. It evaluates model performance using training and validation metrics, saves the best model checkpoint, and applies early stopping to reduce unnecessary training.

## Implementation

### 1. Dataset Preparation

The CIFAR-10 dataset contains 10 image classes. The preprocessing pipeline prepares the training, validation, and test datasets, with configurable class distributions for imbalance experiments.

Training images can be augmented to improve generalization. The validation and test datasets are used for model evaluation rather than training.

### 2. Model Training

The training implementations support the following configurations:

- **Model architectures:** ConvNeXt-Tiny.
- **Initialization:** Pretrained ImageNet weights or training from scratch, where supported.
- **Optimization:** AdamW optimizer with configurable learning rates and weight decay.
- **Learning-rate scheduling:** Cosine annealing.
- **Evaluation metrics:** Training and validation loss, accuracy, and macro F