"""
src/trainer.py
==============
Supervised training engine for the ViT-S/16 baseline.

The Trainer owns the epoch loop and is responsible for:
    - one supervised pass over the data (CrossEntropy + AdamW + AMP)
    - evaluation (loss + accuracy)
    - per-epoch checkpointing (last / best / optional epoch)
    - emergency checkpointing on KeyboardInterrupt or any exception
    - writing metrics to the CSV logger and events to the text logger

It is intentionally model-agnostic about *what* loss is used beyond
CrossEntropy here; when Center Loss is added later, this loop only needs the
model to also return embeddings (already supported via return_embeddings).

A scheduler helper (build_scheduler) lives here too: a dependency-free
cosine schedule with optional linear warmup, stepped once per epoch.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.checkpoint import CheckpointManager
from src.logger import MetricsCSVLogger
from src.utils import AverageMeter, accuracy, get_rng_state

# Optional progress bar (present on Colab); degrade gracefully if absent.
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable: Any, **kwargs: Any) -> Any:  # type: ignore
        return iterable


# ----------------------------------------------------------------------
# Learning-rate scheduler
# ----------------------------------------------------------------------
def build_scheduler(
    optimizer: torch.optim.Optimizer, training_cfg: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build an epoch-stepped LR scheduler from config, or None.

    "cosine": linear warmup for `warmup_epochs`, then cosine decay to ~0.
    "none"  : returns None (constant LR).

    Implemented with LambdaLR so it is transparent and easy to explain.
    The returned multiplier is in [0, 1] and multiplies the base LR.
    """
    name = training_cfg.get("scheduler", "cosine")
    if name in (None, "none"):
        return None
    if name != "cosine":
        raise ValueError(f"Unknown scheduler '{name}'. Use 'cosine' or 'none'.")

    epochs = int(training_cfg["epochs"])
    warmup = int(training_cfg.get("warmup_epochs", 0))

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------
class Trainer:
    """Encapsulates the supervised training/evaluation loop."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        scaler: Any,
        autocast_ctx: Callable[[], Any],
        device: torch.device,
        config: Dict[str, Any],
        class_to_idx: Dict[str, int],
        logger: Any,
        csv_logger: MetricsCSVLogger,
        ckpt_manager: CheckpointManager,
        center_loss: Optional[nn.Module] = None,
        optimizer_center: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.autocast_ctx = autocast_ctx
        self.device = device
        self.config = config
        self.class_to_idx = class_to_idx
        self.logger = logger
        self.csv_logger = csv_logger
        self.ckpt = ckpt_manager
        self.center_loss = center_loss
        self.optimizer_center = optimizer_center

        tcfg = config["training"]
        self.epochs = int(tcfg["epochs"])
        self.seed = int(tcfg["seed"])
        self.use_amp = bool(tcfg.get("use_amp", False))
        self.grad_clip = tcfg.get("grad_clip", None)
        self.save_epoch_ckpts = bool(tcfg.get("save_epoch_checkpoints", False))
        self.eval_every = int(tcfg.get("eval_every", 1))
        # Center Loss is active only when a module was supplied.
        self.use_center_loss = center_loss is not None
        self.center_weight = float(tcfg.get("center_loss_weight", 0.0))
        # Set True if fit() is stopped by KeyboardInterrupt / exception.
        self.interrupted = False

    # -- one training epoch --------------------------------------------
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        ce_meter, center_meter = AverageMeter(), AverageMeter()  # separate components

        pbar = tqdm(loader, desc=f"Train {epoch + 1}/{self.epochs}", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_center_loss:
                self.optimizer_center.zero_grad(set_to_none=True)
            with self.autocast_ctx():
                if self.use_center_loss:
                    # Need the CLS embedding for the center term.
                    logits, feats = self.model(images, return_embeddings=True)
                    ce = self.criterion(logits, targets)
                    center_raw = self.center_loss(feats, targets)
                    loss = ce + self.center_weight * center_raw
                else:
                    logits = self.model(images)
                    ce = self.criterion(logits, targets)
                    center_raw = None
                    loss = ce

            # scaler is a no-op when AMP is disabled, so this path is uniform.
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.grad_clip)
                )
            self.scaler.step(self.optimizer)
            if self.use_center_loss:
                # Centers get their own step; scaler unscales their grads too.
                # NOTE: the center gradients still carry the center_loss_weight
                # factor, so the effective center LR = center_loss_lr * weight.
                self.scaler.step(self.optimizer_center)
            self.scaler.update()

            bs = images.size(0)
            batch_acc = accuracy(logits.detach(), targets)
            loss_meter.update(loss.item(), bs)
            ce_meter.update(ce.item(), bs)
            if self.use_center_loss:
                center_meter.update(center_raw.item(), bs)
            acc_meter.update(batch_acc, bs)

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

        center_raw_avg = center_meter.avg if self.use_center_loss else 0.0
        return {
            "train_loss": loss_meter.avg,                                # total
            "train_ce": ce_meter.avg,                                    # cross-entropy
            "train_center_raw": center_raw_avg,                          # unweighted
            "train_center_weighted": self.center_weight * center_raw_avg,  # lambda * raw
            "train_accuracy": acc_meter.avg,
        }

    # -- evaluation -----------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()

        for images, targets in tqdm(loader, desc="Val", leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with self.autocast_ctx():
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(accuracy(logits, targets), images.size(0))

        # Keys are named val_* -- this loader is the validation set in fit().
        return {"val_loss": loss_meter.avg, "val_accuracy": acc_meter.avg}

    # -- checkpoint state builder --------------------------------------
    def _build_state(
        self, epoch: int, metrics: Dict[str, float], best_acc: float
    ) -> Dict[str, Any]:
        return {
            "epoch": epoch,  # last COMPLETED epoch
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state_dict": (
                self.scaler.state_dict() if self.use_amp else None
            ),
            "center_loss_state_dict": (
                self.center_loss.state_dict() if self.use_center_loss else None
            ),
            "optimizer_center_state_dict": (
                self.optimizer_center.state_dict() if self.use_center_loss else None
            ),
            "best_val_accuracy": best_acc,
            "metrics": metrics,
            "config": self.config,
            "seed": self.seed,
            "class_to_idx": self.class_to_idx,
            "rng_state": get_rng_state(),
        }

    # -- main loop ------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 0,
        best_acc: float = 0.0,
    ) -> float:
        """Run training from `start_epoch` to the configured number of epochs.

        Best-model selection uses the VALIDATION loader (never the test set).
        `last_checkpoint.pt` holds only the state of the last COMPLETED epoch.
        On interruption a PARTIAL snapshot is saved (marked `partial`), but the
        recommended resume point is the clean `last_checkpoint.pt`. Returns the
        best validation accuracy reached.
        """
        self.logger.info(
            f"Starting training: epochs {start_epoch + 1}..{self.epochs}, "
            f"device={self.device.type}, amp={self.use_amp}"
        )
        try:
            for epoch in range(start_epoch, self.epochs):
                t0 = time.time()
                current_lr = self.optimizer.param_groups[0]["lr"]

                train_metrics = self.train_one_epoch(train_loader, epoch)

                # Validate on schedule (always on the final epoch).
                do_eval = ((epoch + 1) % self.eval_every == 0) or (
                    epoch + 1 == self.epochs
                )
                if do_eval:
                    val_metrics = self.evaluate(val_loader)
                else:
                    val_metrics = {"val_loss": None, "val_accuracy": None}

                # Step the LR schedule once per epoch (after this epoch's work).
                if self.scheduler is not None:
                    self.scheduler.step()

                metrics = {**train_metrics, **val_metrics}
                val_acc = val_metrics["val_accuracy"]

                # Decide "best" BEFORE building the saved state, so
                # last_checkpoint.pt always carries the up-to-date best_acc.
                is_best = val_acc is not None and val_acc > best_acc
                if is_best:
                    best_acc = val_acc

                state = self._build_state(epoch, metrics, best_acc)
                last_path = self.ckpt.save_last(state)
                if is_best:
                    self.ckpt.save_best(state)
                if self.save_epoch_ckpts:
                    self.ckpt.save_epoch(state, epoch)

                self.csv_logger.log(
                    epoch=epoch + 1,
                    train_loss=train_metrics["train_loss"],
                    train_ce=train_metrics["train_ce"],
                    train_center_raw=train_metrics["train_center_raw"],
                    train_center_weighted=train_metrics["train_center_weighted"],
                    train_accuracy=train_metrics["train_accuracy"],
                    val_loss=val_metrics["val_loss"],
                    val_accuracy=val_metrics["val_accuracy"],
                    learning_rate=current_lr,
                    checkpoint_path=str(last_path),
                )

                dt = time.time() - t0
                val_str = (
                    f"val_loss={val_metrics['val_loss']:.4f} val_acc={val_acc:.4f}"
                    if val_acc is not None
                    else "val=skipped"
                )
                self.logger.info(
                    f"Epoch {epoch + 1:3d}/{self.epochs} | "
                    f"lr={current_lr:.2e} | "
                    f"train_loss={train_metrics['train_loss']:.4f} "
                    f"train_acc={train_metrics['train_accuracy']:.4f} | "
                    f"{val_str} | "
                    f"best_acc={best_acc:.4f} | {dt:.1f}s"
                    + ("  <-- new best" if is_best else "")
                )

            self.logger.info(f"Training complete. Best val accuracy: {best_acc:.4f}")
            return best_acc

        except KeyboardInterrupt:
            self.interrupted = True
            self.logger.warning("KeyboardInterrupt -- saving a PARTIAL snapshot (mid-epoch).")
            self._emergency_save(locals().get("epoch", start_epoch),
                                 locals().get("metrics", {}), best_acc)
            self.logger.warning(
                "Resume from the last COMPLETED epoch with:\n"
                f"    python train.py --resume {self.ckpt.last_path}"
            )
            return best_acc

        except Exception as exc:
            self.interrupted = True
            self.logger.error(f"Exception during training: {exc} -- saving a PARTIAL snapshot.")
            self._emergency_save(locals().get("epoch", start_epoch),
                                 locals().get("metrics", {}), best_acc)
            raise  # re-raise so the full traceback is visible

    # -- initial checkpoint --------------------------------------------
    def save_initial_checkpoint(self) -> None:
        """Save an epoch=-1 checkpoint with ALL initial states (model, optimizer,
        scheduler, AMP scaler, Center Loss + its optimizer, RNG), so a resume
        works even if training is interrupted during the very first epoch.

        Intended to be called ONCE for a fresh run, before training starts.
        """
        state = self._build_state(epoch=-1, metrics={}, best_acc=0.0)
        self.ckpt.save_last(state)
        self.logger.info("Saved initial checkpoint (epoch=-1) as a safe resume point.")

    # -- emergency save -------------------------------------------------
    def _emergency_save(
        self, epoch: int, metrics: Dict[str, Any], best_acc: float
    ) -> None:
        """Save a PARTIAL, mid-epoch snapshot. It is flagged `partial=True` and
        must NOT be used for resume (`train.py` refuses it) -- it exists only as
        a forensic snapshot. `last_checkpoint.pt` is left untouched."""
        try:
            state = self._build_state(epoch, metrics, best_acc)
            state["partial"] = True  # mid-epoch; NOT a clean resume point
            path = self.ckpt.save_interrupted(state)
            self.logger.warning(
                f"Partial snapshot saved to: {path} (do NOT --resume this; "
                f"use last_checkpoint.pt)."
            )
        except Exception as exc:  # last-resort guard
            self.logger.error(f"Failed to save partial snapshot: {exc}")
