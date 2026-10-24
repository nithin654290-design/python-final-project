"""
Configurable CNN backbones for the GA-CNN hybrid.

Two families are supported and both are driven by the same chromosome-derived
configuration dictionary:

  * "custom"   - a plain 4-block convolutional network built from scratch
  * "resnet18" / "efficientnet_b0" - transfer learning with a configurable head
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 7


class CustomCNN(nn.Module):
    """Simple VGG-style CNN whose width, dropout and depth come from the GA."""

    def __init__(self, base_filters: int = 32, dense_units: int = 256, dropout: float = 0.4,
                 num_blocks: int = 4, num_classes: int = NUM_CLASSES):
        super().__init__()
        blocks = []
        in_channels = 3
        channels = base_filters
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            in_channels = channels
            channels = min(channels * 2, 512)

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def _transfer_model(name: str, dense_units: int, dropout: float, freeze_backbone: bool) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, NUM_CLASSES),
        )
        return model

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, NUM_CLASSES),
        )
        return model

    raise ValueError(f"Unknown backbone: {name}")


def build_model(config: Dict) -> nn.Module:
    """Instantiate a model from a GA configuration dictionary."""
    backbone = config.get("backbone", "resnet18")
    dropout = float(config.get("dropout", 0.4))
    dense_units = int(config.get("dense_units", 256))

    if backbone == "custom":
        return CustomCNN(
            base_filters=int(config.get("base_filters", 32)),
            dense_units=dense_units,
            dropout=dropout,
            num_blocks=int(config.get("num_blocks", 4)),
        )
    return _transfer_model(backbone, dense_units, dropout, bool(config.get("freeze_backbone", False)))


def build_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Create the optimizer named by the chromosome."""
    name = str(config.get("optimizer", "adam")).lower()
    lr = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")
