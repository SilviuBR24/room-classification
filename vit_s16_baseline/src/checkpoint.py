"""
src/checkpoint.py
=================
Robust checkpointing so Colab disconnects never cost more than one epoch.

A checkpoint is a single dict saved with torch.save containing everything
needed to resume *exactly*:
    epoch ................. last COMPLETED epoch (resume starts at epoch+1)
    model_state_dict
    optimizer_state_dict
    scheduler_state_dict .. None if no scheduler
    scaler_state_dict ..... None if AMP disabled
    best_eval_accuracy
    metrics ............... latest train/eval metrics
    config ................ full config dict (architecture is rebuilt from it)
    seed
    class_to_idx .......... fixed class mapping
    rng_state ............. python/numpy/torch/cuda RNG states (faithful resume)

CheckpointManager writes to a single `checkpoints/` directory and knows how
to produce the four canonical files: last / best / epoch_XXX / interrupted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


class CheckpointManager:
    """Manage all checkpoint files for one run's `checkpoints/` directory."""

    LAST_NAME = "last_checkpoint.pt"
    BEST_NAME = "best_model.pt"
    INTERRUPTED_NAME = "interrupted_checkpoint.pt"

    def __init__(self, checkpoint_dir: Union[str, Path]) -> None:
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- paths ----------------------------------------------------------
    @property
    def last_path(self) -> Path:
        return self.dir / self.LAST_NAME

    @property
    def best_path(self) -> Path:
        return self.dir / self.BEST_NAME

    @property
    def interrupted_path(self) -> Path:
        return self.dir / self.INTERRUPTED_NAME

    def epoch_path(self, epoch: int) -> Path:
        return self.dir / f"epoch_{epoch:03d}.pt"

    # -- save -----------------------------------------------------------
    @staticmethod
    def _atomic_save(state: Dict[str, Any], path: Path) -> None:
        """Save to a temp file first, then rename, to avoid corrupt files
        if the process dies mid-write."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def save_last(self, state: Dict[str, Any]) -> Path:
        self._atomic_save(state, self.last_path)
        return self.last_path

    def save_best(self, state: Dict[str, Any]) -> Path:
        self._atomic_save(state, self.best_path)
        return self.best_path

    def save_epoch(self, state: Dict[str, Any], epoch: int) -> Path:
        path = self.epoch_path(epoch)
        self._atomic_save(state, path)
        return path

    def save_interrupted(self, state: Dict[str, Any]) -> Path:
        self._atomic_save(state, self.interrupted_path)
        return self.interrupted_path


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_checkpoint(
    path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = "cpu"
) -> Dict[str, Any]:
    """Load a checkpoint dict, with a clear error for a bad path.

    We load to CPU by default; the caller then moves the model to the device.
    `weights_only=False` is required because our checkpoint stores Python
    objects (config dict, RNG states) alongside tensors.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch versions do not have the weights_only argument.
        return torch.load(path, map_location=map_location)
