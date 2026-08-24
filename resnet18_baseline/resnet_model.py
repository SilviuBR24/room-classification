"""
resnet_model.py
===============
The convolutional baseline: ResNet-18, wrapped so it presents exactly the same
interface as the from-scratch Vision Transformer.

Where the architecture comes from
---------------------------------
`torchvision.models.resnet18` -- the standard implementation of He et al.
(CVPR 2016), shipped with torchvision. It is instantiated with `weights=None`,
so **no pre-trained weights are loaded**: only the architecture is reused, and
every parameter is learned from scratch on the room dataset, exactly like the
ViT. This keeps the comparison fair and respects the thesis constraint that no
pre-training is used.

Why a wrapper is needed
-----------------------
`Trainer` calls `model(images, return_embeddings=True)` whenever Center Loss is
active, because the centre term operates on the penultimate representation.
torchvision's ResNet has no such argument -- it goes straight from the pooled
feature map into the classification head, so the embedding is never exposed.

Replacing that head with `Identity` and adding an explicit classifier makes the
embedding observable without changing the network's computation. The result is
API-compatible with `src/vit.py`, so the shared Trainer, the shared Center Loss
module and the shared evaluation script all work unchanged.

What plays the role of the ViT's CLS token here is the 512-dimensional
global-average-pooled feature vector produced by the last residual stage.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import resnet18

# Fixed by the architecture: ResNet-18's penultimate feature width.
RESNET18_EMBED_DIM = 512


class ResNet18Classifier(nn.Module):
    """ResNet-18 with an observable embedding, API-compatible with the ViT.

        forward(x)                          -> logits
        forward(x, return_embeddings=True)  -> (logits, embedding)
    """

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        backbone = resnet18(weights=None)          # architecture only, no pre-training
        self.embed_dim = backbone.fc.in_features   # 512
        backbone.fc = nn.Identity()                # expose the pooled features
        self.backbone = backbone
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(
        self, x: torch.Tensor, return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedding = self.backbone(x)               # (B, 512)
        logits = self.classifier(embedding)        # (B, num_classes)
        return (logits, embedding) if return_embeddings else logits


def build_resnet18_from_config(model_cfg: Dict[str, Any]) -> ResNet18Classifier:
    """Build the ResNet-18 baseline from the `model` section of a config.

    `embed_dim` is validated rather than used. Center Loss is sized from the
    config value, so a mismatch with the architecture's real feature width would
    produce a wrongly shaped centre matrix. Failing loudly here is far easier to
    diagnose than a shape error deep inside the training loop.
    """
    # Guard against a config copied from the ViT project being run through this
    # builder by mistake, which would silently produce ResNet results filed
    # under a ViT name.
    arch = str(model_cfg.get("arch", "resnet18")).lower()
    if arch != "resnet18":
        raise ValueError(
            f"This builder produces ResNet-18, but the config declares "
            f"model.arch='{arch}'. Refusing to run, so results are not "
            f"mislabelled."
        )

    model = ResNet18Classifier(num_classes=int(model_cfg["num_classes"]))

    declared = model_cfg.get("embed_dim")
    if declared is not None and int(declared) != RESNET18_EMBED_DIM:
        raise ValueError(
            f"config sets model.embed_dim={declared}, but ResNet-18's feature "
            f"width is {RESNET18_EMBED_DIM}. Center Loss is sized from the "
            f"config, so these must agree."
        )
    return model
