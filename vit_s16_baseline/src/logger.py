"""
src/logger.py
=============
Two complementary logging facilities:

1. setup_logger(...)      -> a standard logging.Logger that writes both to
                             logs/train.log (human-readable events) and to
                             the console. Used for messages, warnings, etc.

2. MetricsCSVLogger(...)  -> appends one row per epoch to logs/metrics.csv
                             with the exact columns the dissertation needs.

Both append on resume (the CSV keeps its single header), so reconnecting a
Colab session continues the same record instead of overwriting it.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union


# ----------------------------------------------------------------------
# Human-readable logger
# ----------------------------------------------------------------------
def setup_logger(
    log_file: Union[str, Path], name: str = "vit_train"
) -> logging.Logger:
    """Configure a logger writing to both a file (append) and stdout."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on repeated calls
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


# ----------------------------------------------------------------------
# Metrics CSV logger
# ----------------------------------------------------------------------
class MetricsCSVLogger:
    """Append epoch-level metrics to a CSV file with a fixed schema.

    Separate loss components (cross-entropy, unweighted center, weighted
    center) are logged alongside the total, so the dissertation can analyse
    them independently. `train_loss` stays the TOTAL loss for backward
    compatibility.

    Schema migration: on resume, if an existing file's header does not match
    the current schema (e.g. new columns were added between code versions),
    a new versioned file (metrics_v2.csv, metrics_v3.csv, ...) is used instead
    of corrupting the old one. If a versioned file with the correct schema
    already exists, it is continued (so history is not fragmented on every
    resume). The path actually used is exposed as `self.path`.
    """

    FIELDS = [
        "epoch",
        "train_loss",             # TOTAL loss (CE + weighted center); kept for compat
        "train_ce",               # cross-entropy component
        "train_center_raw",       # unweighted center loss (0.0 when disabled)
        "train_center_weighted",  # center_loss_weight * center_raw
        "train_accuracy",
        "val_loss",               # validation loss (used for best-model selection)
        "val_accuracy",
        "learning_rate",
        "checkpoint_path",
    ]

    def __init__(
        self,
        path: Union[str, Path],
        resume: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)

        chosen, write_header = base, True
        if resume and base.exists():
            if self._read_header(base) == self.FIELDS:
                chosen, write_header = base, False              # schema matches -> append
            else:
                chosen, write_header = self._resolve_versioned(base)
                if logger is not None:
                    logger.warning(
                        f"metrics schema changed; logging to '{chosen.name}' because "
                        f"'{base.name}' has an older schema (old file kept untouched)."
                    )

        self.path = chosen                                       # expose the real path
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    @staticmethod
    def _read_header(path: Path) -> Optional[list]:
        try:
            with open(path, "r", newline="", encoding="utf-8") as fh:
                return next(csv.reader(fh), None)
        except Exception:
            return None

    @classmethod
    def _resolve_versioned(cls, base: Path):
        """Return (path, write_header). Continue an existing versioned file if it
        already has the correct schema; otherwise create the next version."""
        n = 2
        while True:
            cand = base.with_name(f"{base.stem}_v{n}.csv")
            if not cand.exists():
                return cand, True                                # new file -> header
            if cls._read_header(cand) == cls.FIELDS:
                return cand, False                               # correct schema -> append
            n += 1

    def log(
        self,
        epoch: int,
        train_loss: float,
        train_ce: float,
        train_center_raw: float,
        train_center_weighted: float,
        train_accuracy: float,
        val_loss: Optional[float],
        val_accuracy: Optional[float],
        learning_rate: float,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Write one epoch row. Validation fields may be None on non-eval epochs."""
        def fmt(x: Optional[float]) -> str:
            return "" if x is None else f"{x:.6f}"

        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": fmt(train_loss),
            "train_ce": fmt(train_ce),
            "train_center_raw": fmt(train_center_raw),
            "train_center_weighted": fmt(train_center_weighted),
            "train_accuracy": fmt(train_accuracy),
            "val_loss": fmt(val_loss),
            "val_accuracy": fmt(val_accuracy),
            "learning_rate": "" if learning_rate is None else f"{learning_rate:.8f}",
            "checkpoint_path": checkpoint_path or "",
        }
        self._writer.writerow(row)
        self._file.flush()  # flush so data survives an abrupt disconnect

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass
