"""
src/utils.py
============
Small, dependency-light helpers shared across the project:

- set_seed / get/set RNG state .... reproducibility (and faithful resume)
- get_device ...................... CUDA-or-CPU selection
- load_config / save_config ....... YAML read/write with clear errors
- get_amp_components .............. version-robust mixed-precision setup
- AverageMeter / accuracy ......... running metric tracking
- count_parameters / ensure_dir ... misc conveniences

These functions deliberately contain NO training logic so they can be
unit-tested and reused by both `train.py` and `evaluate.py`.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, Tuple, Union

import numpy as np
import torch
import yaml


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    Args:
        seed: integer seed.
        deterministic: if True, set cuDNN to deterministic mode. Note that
            full bit-for-bit determinism on GPU is NOT guaranteed (some CUDA
            kernels are non-deterministic), but results become far more stable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_rng_state() -> Dict[str, Any]:
    """Capture RNG states so training can resume without a statistical jump."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG states captured by `get_rng_state` (best-effort)."""
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:  # never let a resume fail just because of RNG
        print(f"[utils] Warning: could not fully restore RNG state ({exc}).")


# ----------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------
def get_device() -> torch.device:
    """Return the best available device (CUDA if present, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Config I/O
# ----------------------------------------------------------------------
def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file into a dict, with explicit error handling."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse into a dict.")
    return cfg


def save_config(cfg: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write a config dict to YAML (used for config_used.yaml)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if missing; return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------------------------------------------------------
# Mixed precision (AMP) — robust across torch versions
# ----------------------------------------------------------------------
def get_amp_components(
    device: torch.device, enabled: bool
) -> Tuple[Any, Callable[[], ContextManager]]:
    """Return (GradScaler, autocast_context_factory) compatible with AMP.

    AMP is only truly enabled on CUDA. On CPU the scaler/autocast become
    no-ops, so the *same* training code path works on both. Newer torch
    prefers the `torch.amp` API; we fall back to `torch.cuda.amp` otherwise.
    """
    use = bool(enabled) and device.type == "cuda"
    try:  # modern API (torch >= ~2.3)
        scaler = torch.amp.GradScaler("cuda", enabled=use)

        def autocast_ctx() -> ContextManager:
            return torch.amp.autocast("cuda", enabled=use)

    except (AttributeError, TypeError):  # legacy API
        scaler = torch.cuda.amp.GradScaler(enabled=use)

        def autocast_ctx() -> ContextManager:
            return torch.cuda.amp.autocast(enabled=use)

    return scaler, autocast_ctx


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
class AverageMeter:
    """Track a running average (e.g. loss or accuracy over an epoch)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy as a fraction in [0, 1] for one batch."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.size(0))


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
