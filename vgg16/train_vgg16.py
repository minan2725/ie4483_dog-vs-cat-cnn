"""Train a Dogs vs. Cats VGG16 through the shared training pipeline."""

import sys
from pathlib import Path

from torch import nn
from torchvision import models

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dogcat_training import parse_training_args, train_model  # noqa: E402


def build_model(freeze_backbone=False, scratch=False):
    weights = None if scratch else models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    return model


def main(argv=None):
    args = parse_training_args("vgg16", argv)
    train_model("vgg16", lambda cfg: build_model(cfg.freeze_backbone, cfg.scratch), args)


if __name__ == "__main__":
    main()
