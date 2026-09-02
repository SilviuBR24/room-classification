"""
train_fer.py
============
Entry point for training one facial-expression variant. Same command-line
interface as `vit_s16_baseline/train.py`:

    python train_fer.py --config config_fer.yaml
    python train_fer.py --resume <run>/checkpoints/last_checkpoint.pt

How this reuses the existing pipeline
-------------------------------------
`train.py`'s `main()` is entirely config-driven: run folder, seeding, data
loaders, optimizer, LR schedule, mixed precision, Center Loss, checkpointing and
the Trainer are all built from the YAML. Exactly one line is specific to the
architecture -- the call that constructs the model.

This script substitutes that single symbol and delegates, so the facial
expression runs go through literally the same training loop as the room runs.
That is what makes the two settings comparable.

The substitution is checked before it is applied: if `train.py` is refactored so
the symbol disappears, this fails immediately rather than silently training
something else.
"""
from __future__ import annotations

import shared_infrastructure  # noqa: F401  -- must come first; sets sys.path

import train as shared_train  # the shared training entry point

from fer_model import build_fer_model_from_config

# The single architecture-specific symbol inside train.main().
MODEL_BUILDER_SYMBOL = "build_vit_from_config"


def main() -> None:
    if not hasattr(shared_train, MODEL_BUILDER_SYMBOL):
        raise RuntimeError(
            f"train.py no longer defines '{MODEL_BUILDER_SYMBOL}'. This wrapper "
            f"replaces that symbol with the facial-expression model builder, so "
            f"the shared training script must have been refactored. Update this "
            f"wrapper before running, otherwise the wrong model would be trained."
        )
    setattr(shared_train, MODEL_BUILDER_SYMBOL, build_fer_model_from_config)

    # The shared trainer prints a hard-coded "Model: ViT-S/16" line in train.log
    # whatever it builds. Say plainly what is actually being trained. The
    # authoritative record is elsewhere: the run folder name, the arch key in
    # config_used.yaml, and the parameter count.
    print("[fer] task: FER2013 facial expression recognition, 7 classes")
    print("[fer] architecture: ResNet-18 from torchvision, weights=None "
          "(trained from scratch)")
    print("[fer] NOTE: the shared trainer prints 'Model: ViT-S/16' in train.log "
          "regardless of architecture -- ignore that line here.", flush=True)

    shared_train.main()


if __name__ == "__main__":
    main()
