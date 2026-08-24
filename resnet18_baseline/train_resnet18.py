"""
train_resnet18.py
=================
Entry point for training the ResNet-18 baseline. Same command-line interface as
`vit_s16_baseline/train.py`:

    python train_resnet18.py --config config_resnet18.yaml
    python train_resnet18.py --resume <run>/checkpoints/last_checkpoint.pt

How this reuses the existing pipeline
-------------------------------------
`train.py`'s `main()` is entirely config-driven: run folder, seeding, data
loaders, optimizer, LR schedule, mixed precision, Center Loss, checkpointing and
the Trainer are all built from the YAML. Exactly ONE line is specific to the
architecture -- the call that constructs the model.

So this script substitutes that single symbol and delegates. Nothing is
duplicated, which means the ResNet runs and the ViT runs go through literally
the same training loop -- the guarantee that makes the comparison meaningful.

The substitution is checked before it is applied: if `train.py` is ever
refactored so the symbol disappears, this fails immediately with a clear
message instead of silently training the wrong architecture.
"""
from __future__ import annotations

import shared_infrastructure  # noqa: F401  -- must come first; sets sys.path

import train as shared_train  # the shared training entry point

from resnet_model import build_resnet18_from_config

# The single architecture-specific symbol inside train.main().
MODEL_BUILDER_SYMBOL = "build_vit_from_config"


def main() -> None:
    if not hasattr(shared_train, MODEL_BUILDER_SYMBOL):
        raise RuntimeError(
            f"train.py no longer defines '{MODEL_BUILDER_SYMBOL}'. This wrapper "
            f"replaces that symbol with the ResNet-18 builder, so the shared "
            f"training script must have been refactored. Update this wrapper "
            f"before running, otherwise the wrong architecture would be trained."
        )
    setattr(shared_train, MODEL_BUILDER_SYMBOL, build_resnet18_from_config)

    # train.py logs a hard-coded "Model: ViT-S/16" line whatever it builds, so
    # say plainly what is actually being trained before handing over. The
    # authoritative record is elsewhere anyway: the run folder name, the
    # arch key in config_used.yaml, and the parameter count (11,179,590 for
    # ResNet-18 against 21,691,014 for ViT-S/16).
    print("[resnet18] architecture: ResNet-18 from torchvision, weights=None "
          "(trained from scratch)")
    print("[resnet18] NOTE: the shared trainer prints 'Model: ViT-S/16' in "
          "train.log regardless of architecture -- ignore that line here.",
          flush=True)

    shared_train.main()


if __name__ == "__main__":
    main()
