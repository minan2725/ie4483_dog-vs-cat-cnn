"""Train a Dogs vs. Cats ConvNeXt-Tiny through the shared training pipeline."""

import sys
from pathlib import Path

from torch import nn
from torchvision import models

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dogcat_training import parse_training_args, train_model  # noqa: E402


def build_model(freeze_backbone=False, pretrained=True):
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    head = model.classifier[-1]
    model.classifier[-1] = nn.Linear(head.in_features, 2)
    return model


def main(argv=None):
    args = parse_training_args("convnext_tiny", argv)
    train_model("convnext_tiny",
                lambda cfg: build_model(cfg.freeze_backbone, pretrained=not cfg.scratch),
                args, runs_dir=HERE / "runs")


if __name__ == "__main__":
    main()
