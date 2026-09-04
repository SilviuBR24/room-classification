"""ViT-S/16 for the room experiment, built from the shared architecture.

This module deliberately does NOT contain a copy of the Vision Transformer.
The whole point of the comparison is that the architecture is identical to the
one the thesis already used, and a second copy of a twenty-one-million-parameter
model is a copy that can silently drift out of step with the original. It
imports the shared builder instead, and reads it without modifying anything in
`vit_s16_baseline/`.

The `arch` guard exists so that a configuration written for another experiment
cannot be run here by accident.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_SRC = REPO_ROOT / "vit_s16_baseline" / "src"
if str(SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(SHARED_SRC))

from vit import build_vit_from_config, VisionTransformer  # noqa: E402

EXPECTED_ARCH = "vit_s16_rooms_wen"


def build_model_from_config(model_cfg: Dict[str, Any]) -> VisionTransformer:
    """Build the ViT, refusing a configuration meant for a different experiment."""
    arch = model_cfg.get("arch")
    if arch != EXPECTED_ARCH:
        raise SystemExit(
            f"model.arch is {arch!r}, expected {EXPECTED_ARCH!r}. This module "
            "builds the room-experiment ViT only; a config from another "
            "experiment would train the wrong thing under the right run name.")

    # The builder ignores keys it does not know, so `arch` may stay in the dict.
    model = build_vit_from_config(model_cfg)

    # The embedding width is what Center Loss operates on. If the config and
    # the model disagree, the centres would be the wrong shape and the failure
    # would appear much later, as a broadcasting error inside the loss.
    expected_dim = int(model_cfg["embed_dim"])
    actual_dim = model.head.in_features if isinstance(model.head, nn.Linear) \
        else expected_dim
    if actual_dim != expected_dim:
        raise SystemExit(
            f"embed_dim {expected_dim} does not match the model's {actual_dim}")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
