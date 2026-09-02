"""
fer_model.py
============
The ResNet-18 used for the facial-expression experiment, with an observable
embedding so Center Loss and the geometric analysis can operate on it.

This is a deliberate near-duplicate of `resnet18_baseline/resnet_model.py`.
Keeping a private copy means this experiment cannot be broken by a change made
to the room-classification folder, and vice versa. The two files are small; the
isolation is worth more than the reuse.

Why ResNet-18 and not the transformer
-------------------------------------
The question this experiment answers is whether Center Loss organises a
representation when that representation is already adequate. That requires a
model which actually reaches a usable accuracy on the task. The room
experiments already established that a Vision Transformer trained from scratch
on tens of thousands of images does not; repeating that here would only restate
a known result. The convolutional model is therefore the informative choice.

Input resolution
----------------
FER2013 images are 48x48. ResNet-18 was designed for 224x224 and reduces its
input by a factor of 32, so at 48x48 only a 2x2 feature map survives to the
pooling layer, and the 7x7 stride-2 stem plus max-pool discards most of the
image before any residual block runs. Upsampling adds no information, but it
does stop the fixed downsampling from crushing the input first. The resolution
is a config value rather than a constant here so the trade-off can be measured
instead of assumed -- see `calibrate_fer.py`.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import resnet18

# Fixed by the architecture: ResNet-18's penultimate feature width.
RESNET18_EMBED_DIM = 512

# FER2013's seven classes, in the order used by the published dataset.
FER_NUM_CLASSES = 7


class FERResNet18(nn.Module):
    """ResNet-18 with an observable embedding, API-compatible with the ViT.

        forward(x)                          -> logits
        forward(x, return_embeddings=True)  -> (logits, embedding)

    The shared Trainer calls the second form whenever Center Loss is active,
    because the centre term operates on the penultimate representation.
    """

    def __init__(self, num_classes: int = FER_NUM_CLASSES) -> None:
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


def build_fer_model_from_config(model_cfg: Dict[str, Any]) -> FERResNet18:
    """Build the facial-expression model from the `model` section of a config.

    Both `arch` and `embed_dim` are validated rather than merely read. The
    room-classification configs are structurally identical to this one, so a
    file copied from the wrong project would otherwise run happily and produce
    results filed under the wrong name.
    """
    arch = str(model_cfg.get("arch", "")).lower()
    if arch != "resnet18_fer":
        raise ValueError(
            f"This builder produces the facial-expression ResNet-18, but the "
            f"config declares model.arch='{arch}'. Expected 'resnet18_fer'. "
            f"Refusing to run, so results are not mislabelled."
        )

    num_classes = int(model_cfg["num_classes"])
    if num_classes != FER_NUM_CLASSES:
        raise ValueError(
            f"FER2013 has {FER_NUM_CLASSES} classes, but the config sets "
            f"model.num_classes={num_classes}. This usually means a config "
            f"from the six-class room experiments was used by mistake."
        )

    model = FERResNet18(num_classes=num_classes)

    declared = model_cfg.get("embed_dim")
    if declared is not None and int(declared) != RESNET18_EMBED_DIM:
        raise ValueError(
            f"config sets model.embed_dim={declared}, but ResNet-18's feature "
            f"width is {RESNET18_EMBED_DIM}. Center Loss is sized from the "
            f"config value, so these must agree or the centre matrix would "
            f"have the wrong shape."
        )
    return model
