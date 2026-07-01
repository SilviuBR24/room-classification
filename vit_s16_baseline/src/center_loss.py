"""
src/center_loss.py
==================
Center Loss (Wen et al., ECCV 2016 -- "A Discriminative Feature Learning
Approach for Deep Face Recognition").

It pulls each sample's embedding toward a learnable per-class center, shrinking
intra-class variance so the classes form tighter, better-separated clusters in
feature space. Used ADDITIVELY with cross-entropy:

    L_total  = L_CE + lambda * L_center
    L_center = (1/2) * mean_i || x_i - c_{y_i} ||^2

where x_i is the CLS embedding of sample i (from VisionTransformer with
`return_embeddings=True`) and c_{y_i} is the center of its true class.

The class centers (`centers`, shape num_classes x feat_dim) are LEARNABLE
parameters. They are updated by their own optimizer during training -- this
module only computes the loss; the training loop (src/trainer.py) owns the
optimizer step so the center learning rate stays independent of the model's.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Learnable class centers + the center-loss term.

    Args:
        num_classes: number of classes (6 for this project).
        feat_dim:    embedding dimension (384 for ViT-S, the CLS embedding).
    """

    def __init__(self, num_classes: int, feat_dim: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # One learnable center per class, small random init (like other params).
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.1)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean squared distance between each embedding and its class center.

        Args:
            features: (B, feat_dim) CLS embeddings.
            labels:   (B,) integer class indices.
        Returns:
            Scalar center-loss value.
        """
        centers_batch = self.centers.index_select(0, labels)   # (B, feat_dim)
        return 0.5 * ((features - centers_batch) ** 2).sum(dim=1).mean()
