"""
src/vit.py
==========
Vision Transformer (ViT-S/16) implemented FROM SCRATCH in PyTorch.

No timm / torchvision / HuggingFace / pretrained weights are used. Only
PyTorch core layers (nn.Linear, nn.LayerNorm, nn.Conv2d, nn.GELU, nn.Dropout)
appear, each chosen so the architecture stays easy to explain in the thesis.

Components (each is its own nn.Module so it can be described separately):
    PatchEmbedding            -> split image into patches, project to tokens
    MultiHeadSelfAttention    -> scaled dot-product attention, implemented by hand
    MLP                       -> position-wise feed-forward (GELU)
    TransformerEncoderBlock   -> pre-norm residual block (attn + MLP)
    VisionTransformer         -> CLS token + positional embeddings + blocks + head

Design note for later stages
-----------------------------
`forward_features` returns the CLS-token representation AFTER the final
LayerNorm and BEFORE the classifier head. This is exactly the embedding that
Center Loss (and embedding-space / t-SNE plots) will use later. `forward`
can optionally return it via `return_embeddings=True`, so no architectural
change is needed when those stages are added.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Patch embedding
# ----------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and linearly project them.

    A Conv2d with kernel_size = stride = patch_size is mathematically
    equivalent to flattening each patch and applying a single Linear layer,
    but is faster and cleaner. Output: a sequence of patch tokens.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by "
                f"patch_size ({patch_size})."
            )
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


# ----------------------------------------------------------------------
# Multi-head self-attention (implemented explicitly)
# ----------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention.

    Implemented by hand (rather than nn.MultiheadAttention) so every step
    (qkv projection, head split, scaled scores, softmax, weighted sum) is
    visible and explainable in the dissertation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape  # (batch, tokens, embed_dim)

        # Project to queries/keys/values and split into heads.
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention.
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(b, n, d)  # merge heads -> (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ----------------------------------------------------------------------
# Feed-forward block
# ----------------------------------------------------------------------
class MLP(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear."""

    def __init__(
        self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ----------------------------------------------------------------------
# Transformer encoder block (pre-norm)
# ----------------------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer block with residual connections.

        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Pre-norm (LayerNorm before each sub-layer) trains far more stably from
    scratch than the original post-norm formulation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads, attn_dropout, dropout, qkv_bias
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------------------------------------------------------
# Full Vision Transformer
# ----------------------------------------------------------------------
class VisionTransformer(nn.Module):
    """ViT for image classification with a CLS token.

    Sequence layout fed to the encoder:
        [CLS, patch_1, patch_2, ..., patch_N]  (length N + 1)
    Learnable positional embeddings are added to every position.
    The CLS token's final representation is used for classification.
    """

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        num_classes: int = 6,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token and learnable positional embeddings (+1 for CLS).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim, num_heads, mlp_ratio, dropout, attn_dropout, qkv_bias
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)          # final norm
        self.head = nn.Linear(embed_dim, num_classes)  # classifier

        self._init_weights()

    # -- weight initialization -----------------------------------------
    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    # -- feature extraction --------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the CLS embedding (B, embed_dim) after the final LayerNorm.

        This is the representation used later for Center Loss and embedding
        plots — kept separate from the classifier head on purpose.
        """
        b = x.shape[0]
        x = self.patch_embed(x)                       # (B, N, D)
        cls = self.cls_token.expand(b, -1, -1)        # (B, 1, D)
        x = torch.cat((cls, x), dim=1)                # (B, N+1, D)
        x = x + self.pos_embed                        # add positions
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]                                # CLS token

    # -- forward --------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute logits; optionally also return the CLS embedding."""
        features = self.forward_features(x)           # (B, D)
        logits = self.head(features)                  # (B, num_classes)
        if return_embeddings:
            return logits, features
        return logits


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def build_vit_from_config(model_cfg: Dict[str, Any]) -> VisionTransformer:
    """Construct a VisionTransformer from the `model` section of the config.

    Missing optional keys fall back to ViT-S/16 defaults so older configs
    still load. Required structural keys raise a clear error if absent.
    """
    try:
        return VisionTransformer(
            image_size=model_cfg["image_size"],
            patch_size=model_cfg["patch_size"],
            in_channels=model_cfg.get("in_channels", 3),
            embed_dim=model_cfg["embed_dim"],
            depth=model_cfg["depth"],
            num_heads=model_cfg["num_heads"],
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            dropout=model_cfg.get("dropout", 0.1),
            attn_dropout=model_cfg.get("attn_dropout", 0.0),
            num_classes=model_cfg["num_classes"],
            qkv_bias=model_cfg.get("qkv_bias", True),
        )
    except KeyError as exc:
        raise KeyError(f"Missing required model config key: {exc}") from exc
