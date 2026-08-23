"""
src/pseudo_label.py
===================
Pseudo-labelling primitives for the Part-3 semi-supervised scenario.

The network self-labels an *unlabelled* image pool using its CURRENT knowledge:
run inference, keep only predictions whose confidence (max softmax) clears a
threshold tau, and treat those argmax classes as pseudo-labels for the next
round of training.

Everything here is deliberately small and side-effect free so it can be unit
tested and reused. Nothing in this module trains a model or touches disk; it
only turns a model + an unlabelled loader into (a) confidence statistics and
(b) a Dataset of confidently pseudo-labelled images.

Note on the "unlabelled" pool: in this project the pool DOES have ground-truth
folders (it is the held-out 1500/class). We never use those labels for
training -- only to MEASURE pseudo-label accuracy, which is exactly the
analysis the dissertation asks for.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@torch.no_grad()
def infer_confidence(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    autocast_ctx: Callable[[], Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over `loader` (which must NOT shuffle) and return, in
    dataset order:

        conf  : (N,) max softmax probability per sample
        pred  : (N,) argmax class index per sample
        truth : (N,) the loader's labels (used only to score pseudo-labels)

    The loader is expected to yield (image, label) like any RoomClassification
    loader; `label` is passed straight through as `truth`.
    """
    model.eval()
    confs, preds, truths = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast_ctx():
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        conf, pred = probs.max(dim=1)
        confs.append(conf.cpu().numpy())
        preds.append(pred.cpu().numpy())
        truths.append(labels.numpy())
    return (
        np.concatenate(confs),
        np.concatenate(preds),
        np.concatenate(truths),
    )


def select_confident(conf: np.ndarray, tau: float) -> np.ndarray:
    """Boolean mask of samples whose confidence clears the threshold tau."""
    return conf >= float(tau)


def threshold_for_coverage(conf: np.ndarray, target_coverage: float) -> float:
    """Return the threshold that selects (about) `target_coverage` of the pool.

    A FIXED tau makes coverage an *outcome*: with a weak model almost nothing
    clears a high bar, so the unlabelled pool goes unused. Deriving tau from a
    quantile of this round's confidence distribution flips that around and makes
    coverage the *controlled variable* -- "always take the most confident X%" --
    which is what lets us trace the precision/coverage trade-off curve.

    Recomputed every round, so the bar naturally rises as the model sharpens.
    """
    target = float(target_coverage)
    if not 0.0 < target < 1.0:
        raise ValueError(f"target_coverage must be in (0, 1), got {target}")
    if conf.size == 0:
        return float("inf")
    # The (1 - target) quantile: everything above it is the top `target` share.
    return float(np.quantile(conf, 1.0 - target))


def pseudo_label_stats(
    conf: np.ndarray, pred: np.ndarray, truth: np.ndarray, tau: float
) -> Dict[str, float]:
    """Summarise one self-labelling round.

    Returns a dict with:
        n_total        : size of the unlabelled pool
        n_selected     : how many cleared tau (got a pseudo-label this round)
        coverage       : n_selected / n_total
        pseudo_accuracy: accuracy of the selected pseudo-labels vs ground truth
                         (NaN when nothing was selected)
        mean_conf      : mean confidence over the whole pool (diagnostic)
    """
    mask = select_confident(conf, tau)
    n_sel = int(mask.sum())
    pseudo_acc = float((pred[mask] == truth[mask]).mean()) if n_sel > 0 else float("nan")
    return {
        "n_total": int(conf.shape[0]),
        "n_selected": n_sel,
        "coverage": n_sel / max(1, conf.shape[0]),
        "pseudo_accuracy": pseudo_acc,
        "mean_conf": float(conf.mean()),
    }


class PseudoLabeledSubset(Dataset):
    """A training view over the confidently pseudo-labelled samples.

    It wraps a base dataset built over the unlabelled pool WITH THE TRAIN
    TRANSFORM (so pseudo-labelled images get the same augmentation as real
    ones) and, for the selected indices, returns the image paired with its
    *pseudo* label instead of the ground-truth label.

    `base_train_ds` must index-align with the loader used for inference -- both
    are RoomClassificationDataset over the same directory, whose sample order
    is deterministic (fixed class order + sorted filenames), so index i refers
    to the same image in both.
    """

    def __init__(
        self,
        base_train_ds: Dataset,
        indices: Sequence[int],
        pseudo_labels: Sequence[int],
    ) -> None:
        if len(indices) != len(pseudo_labels):
            raise ValueError("indices and pseudo_labels must have equal length.")
        self.base = base_train_ds
        self.indices = [int(i) for i in indices]
        self.pseudo_labels = [int(p) for p in pseudo_labels]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        image, _true = self.base[self.indices[i]]
        return image, self.pseudo_labels[i]


def build_pseudo_subset(
    base_train_ds: Dataset, conf: np.ndarray, pred: np.ndarray, tau: float
) -> PseudoLabeledSubset:
    """Convenience: turn a confidence run into the pseudo-labelled train subset."""
    mask = select_confident(conf, tau)
    indices = np.nonzero(mask)[0]
    return PseudoLabeledSubset(base_train_ds, indices, pred[indices])
