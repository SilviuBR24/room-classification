"""
wen_center_loss.py
==================
Two Center Loss implementations that differ ONLY in how the class centres are
updated, so the effect of the update rule can be measured rather than argued.

Both compute the same scalar penalty, mini-batch averaged:

    L_C = (1 / 2B) * sum_i || x_i - c_{y_i} ||^2

and therefore send the same gradient to the embeddings. What differs is the
centre update.

MODE "gradient" -- what most public implementations do
------------------------------------------------------
The centres are ordinary learnable parameters. They are optimised by a separate
SGD optimiser on the COMBINED objective L = L_CE + lambda * L_C, so:

    c_j <- c_j - eta_c * lambda * (1/B) * sum_{i: y_i = j} (c_j - x_i)

Two consequences follow, and both are properties of this choice rather than of
the method. The coefficient lambda enters the centre gradient, so the effective
centre learning rate is eta_c * lambda. And the displacement is normalised by
the mini-batch size B, so a class that is rare in the batch moves its centre
proportionally less.

This mode reproduces the implementation used elsewhere in this project and
serves as the control run.

MODE "wen" -- Algorithm 1 of the paper
--------------------------------------
Wen and colleagues update each centre with a dedicated rule, normalised per
class by one plus the number of samples of that class in the mini-batch
(Equation 4 of the paper), applied with its own rate alpha:

    delta_c_j = ( sum_{i: y_i = j} (c_j - x_i) ) / ( 1 + n_j )
    c_j <- c_j - alpha * delta_c_j

Here alpha is independent of lambda, and every class advances at a rate set by
its own count rather than by the batch size. The centres are a buffer, not a
parameter: no gradient reaches them, and `update_centers` performs the step
explicitly after the model has been updated.

What is held constant between the two modes
-------------------------------------------
The scalar loss and hence the gradient to the embeddings; lambda; alpha and
eta_c both 0.5; the data, the schedule, the seed and the architecture. The
update rule is the only difference, which is what makes the comparison
informative.

Reference: Wen, Zhang, Li, Qiao, "A Discriminative Feature Learning Approach
for Deep Face Recognition", ECCV 2016, Equations 2-5 and Algorithm 1.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

Mode = Literal["gradient", "wen"]


class CenterLossVariant(nn.Module):
    """Center Loss with a selectable centre-update rule.

    Args:
        num_classes: number of classes (7 for FER2013).
        feat_dim:    embedding dimension (512 for ResNet-18).
        mode:        "gradient" for the parameter-and-optimiser form used by
                     most public implementations, "wen" for Algorithm 1.
        init_std:    standard deviation of the initial centres. Kept identical
                     across modes so both start from the same distribution.
    """

    def __init__(self, num_classes: int, feat_dim: int, mode: Mode,
                 init_std: float = 0.1) -> None:
        super().__init__()
        if mode not in ("gradient", "wen"):
            raise ValueError(f"mode must be 'gradient' or 'wen', got {mode!r}")
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.mode = mode

        centers = torch.randn(self.num_classes, self.feat_dim) * float(init_std)
        if mode == "gradient":
            # A parameter, so an external optimiser can step it from the
            # gradient of the combined objective.
            self.centers = nn.Parameter(centers)
        else:
            # A buffer: no gradient reaches it, and update_centers() moves it.
            # Registered so it is saved in and restored from state_dict, exactly
            # like a parameter would be.
            self.register_buffer("centers", centers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean squared distance between each embedding and its class centre.

        In "wen" mode the centres are detached, so this returns a value whose
        gradient reaches the embeddings only. In "gradient" mode they are left
        in the graph, so the external optimiser receives their gradient.
        """
        if features.dim() != 2:
            raise ValueError(f"features must be (B, feat_dim), got {tuple(features.shape)}")
        if features.size(1) != self.feat_dim:
            raise ValueError(
                f"features has width {features.size(1)} but this module was "
                f"built for {self.feat_dim}")
        if labels.dim() != 1 or labels.size(0) != features.size(0):
            raise ValueError("labels must be (B,) and match features")
        if int(labels.max()) >= self.num_classes or int(labels.min()) < 0:
            raise ValueError(
                f"labels out of range for {self.num_classes} classes: "
                f"[{int(labels.min())}, {int(labels.max())}]")

        centers = self.centers if self.mode == "gradient" else self.centers.detach()
        centers_batch = centers.index_select(0, labels)
        return 0.5 * ((features - centers_batch) ** 2).sum(dim=1).mean()

    @torch.no_grad()
    def update_centers(self, features: torch.Tensor, labels: torch.Tensor,
                       alpha: float) -> None:
        """Algorithm 1 of Wen et al. Only valid in "wen" mode.

        Computes, for every class present in the mini-batch,

            delta_c_j = sum_{i: y_i = j} (c_j - x_i) / (1 + n_j)

        and applies c_j <- c_j - alpha * delta_c_j.

        Classes absent from the batch are left untouched, which is what the
        algorithm specifies: their sum is empty, so their displacement is zero.

        Runs in float32 regardless of the precision of `features`, because a
        mixed-precision forward pass yields half-precision embeddings and the
        centres are a running quantity that should not accumulate rounding.
        """
        if self.mode != "wen":
            raise RuntimeError(
                "update_centers() is only meaningful in 'wen' mode; in "
                "'gradient' mode the centres are stepped by their optimiser.")

        feats = features.detach().to(self.centers.dtype)
        labels = labels.detach()

        # counts[j] = n_j, the number of samples of class j in this batch.
        counts = torch.zeros(self.num_classes, device=self.centers.device,
                             dtype=self.centers.dtype)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=self.centers.dtype))

        # sum_feats[j] = sum of embeddings of class j in this batch.
        sum_feats = torch.zeros_like(self.centers)
        sum_feats.index_add_(0, labels, feats)

        # sum_i (c_j - x_i) = n_j * c_j - sum_feats[j]
        numerator = counts.unsqueeze(1) * self.centers - sum_feats
        delta = numerator / (1.0 + counts).unsqueeze(1)

        self.centers -= float(alpha) * delta

    def extra_repr(self) -> str:
        return (f"num_classes={self.num_classes}, feat_dim={self.feat_dim}, "
                f"mode={self.mode!r}")
