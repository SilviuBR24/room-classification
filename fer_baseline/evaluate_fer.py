"""
evaluate_fer.py
===============
Evaluate a trained facial-expression checkpoint on the held-out test set. Same
command-line interface as `vit_s16_baseline/evaluate.py`:

    python evaluate_fer.py --checkpoint <run>/checkpoints/best_model.pt \
                           --data-dir <test_dir> --save-embeddings

Same delegation as `train_fer.py`, and for the same reason: `evaluate.py` is
architecture-agnostic apart from the single line that rebuilds the model from
the checkpoint's stored config. Substituting that symbol means the facial
expression reports come out in the same format as the room reports -- same
metrics.txt, same confusion matrix files, same classification report, same
predictions.csv, same embeddings.

That uniformity matters more here than usual. FER2013's classes are heavily
imbalanced, so the classification report's per-class precision, recall and
macro-averaged F1 carry most of the information; overall accuracy alone would
hide a model that simply ignores the rarest expression.
"""
from __future__ import annotations

import shared_infrastructure  # noqa: F401  -- must come first; sets sys.path

import evaluate as shared_evaluate  # the shared evaluation entry point

from fer_model import build_fer_model_from_config

# The single architecture-specific symbol inside evaluate.main().
MODEL_BUILDER_SYMBOL = "build_vit_from_config"


def main() -> None:
    if not hasattr(shared_evaluate, MODEL_BUILDER_SYMBOL):
        raise RuntimeError(
            f"evaluate.py no longer defines '{MODEL_BUILDER_SYMBOL}'. This "
            f"wrapper replaces that symbol with the facial-expression builder, "
            f"so the shared evaluation script must have been refactored. Update "
            f"this wrapper before running, otherwise the checkpoint would be "
            f"loaded into the wrong architecture."
        )
    setattr(shared_evaluate, MODEL_BUILDER_SYMBOL, build_fer_model_from_config)
    shared_evaluate.main()


if __name__ == "__main__":
    main()
