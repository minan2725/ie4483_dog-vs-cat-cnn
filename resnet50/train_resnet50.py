"""Train a Dogs vs. Cats ResNet50 through the shared training pipeline."""

import sys
from pathlib import Path

from torch import nn
from torchvision import models

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dogcat_training import parse_training_args, train_model  # noqa: E402


def build_model(freeze_backbone=False, scratch=False):
    weights = None if scratch else models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def main(argv=None):
    args = parse_training_args("resnet50", argv)
    train_model("resnet50", lambda cfg: build_model(cfg.freeze_backbone, cfg.scratch), args)


if __name__ == "__main__":
    main()
