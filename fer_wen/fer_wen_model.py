"""
fer_wen_model.py
================
ResNet-18 for FER2013, with an observable embedding.

A private copy, deliberately. This folder is fully standalone: it shares no
code with `vit_s16_baseline/`, `resnet18_baseline/` or `fer_baseline/`, so a
change to any of them cannot alter this experiment, and nothing here can alter
them. The isolation is the point -- this folder exists to vary the Center Loss
update rule, which the shared training loop cannot express.

The architecture is identical to the one used by `fer_baseline/`: torchvision
ResNet-18 with `weights=None`, its classification head replaced so the pooled
512-dimensional feature vector is observable, and a linear classifier onto the
seven FER2013 classes. That must stay identical, because the comparison between
the two update rules is only meaningful if the model is the same.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import resnet18

RESNET18_EMBED_DIM = 512
FER_NUM_CLASSES = 7


class FERResNet18(nn.Module):
    """ResNet-18 exposing its pooled features alongside the logits.

        forward(x)                          -> logits
        forward(x, return_embeddings=True)  -> (logits, embedding)
    """

    def __init__(self, num_classes: int = FER_NUM_CLASSES) -> None:
        super().__init__()
        backbone = resnet18(weights=None)          # architecture only
        self.embed_dim = backbone.fc.in_features   # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(
        self, x: torch.Tensor, return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedding = self.backbone(x)
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embeddings else logits


def build_model_from_config(model_cfg: Dict[str, Any]) -> FERResNet18:
    """Build the model, validating the config so a foreign one cannot slip in.

    Every experiment in this project uses a structurally identical config file.
    Without these checks, a config copied from the room experiments would train
    happily and file its results under this experiment's name.
    """
    arch = str(model_cfg.get("arch", "")).lower()
    if arch != "resnet18_fer_wen":
        raise ValueError(
            f"This builder belongs to the Center Loss update-rule comparison, "
            f"but the config declares model.arch='{arch}'. Expected "
            f"'resnet18_fer_wen'. Refusing to run so results are not mislabelled.")

    num_classes = int(model_cfg["num_classes"])
    if num_classes != FER_NUM_CLASSES:
        raise ValueError(
            f"FER2013 has {FER_NUM_CLASSES} classes, config says {num_classes}.")

    declared = model_cfg.get("embed_dim")
    if declared is not None and int(declared) != RESNET18_EMBED_DIM:
        raise ValueError(
            f"config sets model.embed_dim={declared}, but ResNet-18's feature "
            f"width is {RESNET18_EMBED_DIM}. Center Loss is sized from the "
            f"config value, so these must agree.")

    return FERResNet18(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
