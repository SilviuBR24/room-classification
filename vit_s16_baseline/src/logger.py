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
    """Append epoch-level metrics to a CSV file with a fixed schema."""

    FIELDS = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "eval_loss",
        "eval_accuracy",
        "learning_rate",
        "checkpoint_path",
    ]

    def __init__(self, path: Union[str, Path], resume: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write the header only for a brand-new file.
        write_header = not (resume and self.path.exists())
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        eval_loss: Optional[float],
        eval_accuracy: Optional[float],
        learning_rate: float,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """Write one epoch row. Eval fields may be None on non-eval epochs."""
        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_accuracy": f"{train_accuracy:.6f}",
            "eval_loss": "" if eval_loss is None else f"{eval_loss:.6f}",
            "eval_accuracy": "" if eval_accuracy is None else f"{eval_accuracy:.6f}",
            "learning_rate": f"{learning_rate:.8f}",
            "checkpoint_path": checkpoint_path or "",
        }
        self._writer.writerow(row)
        self._file.flush()  # flush so data survives an abrupt disconnect

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass
