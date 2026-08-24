"""
evaluate_resnet18.py
====================
Evaluate a trained ResNet-18 checkpoint on the held-out test set. Same
command-line interface as `vit_s16_baseline/evaluate.py`:

    python evaluate_resnet18.py --checkpoint <run>/checkpoints/best_model.pt \
                                --data-dir <test_dir> --save-embeddings

Same delegation as `train_resnet18.py`, and for the same reason: `evaluate.py`
is architecture-agnostic apart from the single line that rebuilds the model from
the checkpoint's stored config. Substituting that one symbol means the ResNet
reports come out in **byte-identical format** to the ViT reports -- same
metrics.txt, same confusion matrix files, same classification report, same
predictions.csv, same embeddings. That matters: the two architectures are meant
to be compared, and a divergent report format would make that harder for no good
reason.
"""
from __future__ import annotations

import shared_infrastructure  # noqa: F401  -- must come first; sets sys.path

import evaluate as shared_evaluate  # the shared evaluation entry point

from resnet_model import build_resnet18_from_config

# The single architecture-specific symbol inside evaluate.main().
MODEL_BUILDER_SYMBOL = "build_vit_from_config"


def main() -> None:
    if not hasattr(shared_evaluate, MODEL_BUILDER_SYMBOL):
        raise RuntimeError(
            f"evaluate.py no longer defines '{MODEL_BUILDER_SYMBOL}'. This "
            f"wrapper replaces that symbol with the ResNet-18 builder, so the "
            f"shared evaluation script must have been refactored. Update this "
            f"wrapper before running, otherwise the checkpoint would be loaded "
            f"into the wrong architecture."
        )
    setattr(shared_evaluate, MODEL_BUILDER_SYMBOL, build_resnet18_from_config)
    shared_evaluate.main()


if __name__ == "__main__":
    main()
