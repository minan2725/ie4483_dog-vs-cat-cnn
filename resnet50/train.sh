#!/usr/bin/env bash
# ResNet50 experiments (7 runs). Run from the repository root.
# Every run: all training images, 224 px, batch 32, 10 epochs.

# 1. Learning rate (fine-tuning, augmentation on)
python resnet50/train_resnet50.py --lr 1e-5 --seed 42 
python resnet50/train_resnet50.py --lr 1e-4 --seed 42 
python resnet50/train_resnet50.py --lr 1e-3 --seed 42 

# After step 1, put the best learning rate here (highest validation accuracy):
LR=1e-4

# 2. Augmentation off, at the best learning rate
python resnet50/train_resnet50.py --lr $LR --seed 42 --no-augment 

# 3. Two more seeds of the best configuration of steps 1-2
#    (add --no-augment to both lines if augmentation off was better)
python resnet50/train_resnet50.py --lr $LR --seed 1 
python resnet50/train_resnet50.py --lr $LR --seed 2 

# 4. From scratch (same batch size and epoch limit, higher learning rate)
python resnet50/train_resnet50.py --scratch --lr 1e-3 --epochs 20 --seed 42 