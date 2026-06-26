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

        tcfg = config["training"]
        self.epochs = int(tcfg["epochs"])
        self.seed = int(tcfg["seed"])
        self.use_amp = bool(tcfg.get("use_amp", False))
        self.grad_clip = tcfg.get("grad_clip", None)
        self.save_epoch_ckpts = bool(tcfg.get("save_epoch_checkpoints", False))
        self.eval_every = int(tcfg.get("eval_every", 1))

    # -- one training epoch --------------------------------------------
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()

        pbar = tqdm(loader, desc=f"Train {epoch + 1}/{self.epochs}", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self.autocast_ctx():
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            # scaler is a no-op when AMP is disabled, so this path is uniform.
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.grad_clip)
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_acc = accuracy(logits.detach(), targets)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(batch_acc, images.size(0))

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

        return {"train_loss": loss_meter.avg, "train_accuracy": acc_meter.avg}

    # -- evaluation -----------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()

        for images, targets in tqdm(loader, desc="Eval", leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with self.autocast_ctx():
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(accuracy(logits, targets), images.size(0))

        return {"eval_loss": loss_meter.avg, "eval_accuracy": acc_meter.avg}

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
            "best_eval_accuracy": best_acc,
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
        eval_loader: DataLoader,
        start_epoch: int = 0,
        best_acc: float = 0.0,
    ) -> float:
        """Run training from `start_epoch` to the configured number of epochs.

        Saves last/best/epoch checkpoints each epoch and an emergency
        checkpoint on interruption. Returns the best eval accuracy reached.
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

                # Evaluate on schedule (always on the final epoch).
                do_eval = ((epoch + 1) % self.eval_every == 0) or (
                    epoch + 1 == self.epochs
                )
                if do_eval:
                    eval_metrics = self.evaluate(eval_loader)
                else:
                    eval_metrics = {"eval_loss": None, "eval_accuracy": None}

                # Step the LR schedule once per epoch (after this epoch's work).
                if self.scheduler is not None:
                    self.scheduler.step()

                metrics = {**train_metrics, **eval_metrics}
                eval_acc = eval_metrics["eval_accuracy"]

                # Always save "last".
                state = self._build_state(epoch, metrics, best_acc)
                last_path = self.ckpt.save_last(state)

                # Save "best" when eval accuracy improves.
                is_best = eval_acc is not None and eval_acc > best_acc
                if is_best:
                    best_acc = eval_acc
                    # rebuild state so best_eval_accuracy reflects the new best
                    state = self._build_state(epoch, metrics, best_acc)
                    self.ckpt.save_best(state)

                # Optional per-epoch archive.
                if self.save_epoch_ckpts:
                    self.ckpt.save_epoch(state, epoch)

                self.csv_logger.log(
                    epoch=epoch + 1,
                    train_loss=train_metrics["train_loss"],
                    train_accuracy=train_metrics["train_accuracy"],
                    eval_loss=eval_metrics["eval_loss"],
                    eval_accuracy=eval_metrics["eval_accuracy"],
                    learning_rate=current_lr,
                    checkpoint_path=str(last_path),
                )

                dt = time.time() - t0
                eval_str = (
                    f"eval_loss={eval_metrics['eval_loss']:.4f} "
                    f"eval_acc={eval_acc:.4f}"
                    if eval_acc is not None
                    else "eval=skipped"
                )
                self.logger.info(
                    f"Epoch {epoch + 1:3d}/{self.epochs} | "
                    f"lr={current_lr:.2e} | "
                    f"train_loss={train_metrics['train_loss']:.4f} "
                    f"train_acc={train_metrics['train_accuracy']:.4f} | "
                    f"{eval_str} | "
                    f"best_acc={best_acc:.4f} | {dt:.1f}s"
                    + ("  <-- new best" if is_best else "")
                )

            self.logger.info(f"Training complete. Best eval accuracy: {best_acc:.4f}")
            return best_acc

        except KeyboardInterrupt:
            self.logger.warning("KeyboardInterrupt received -- saving emergency checkpoint.")
            self._emergency_save(locals().get("epoch", start_epoch),
                                 locals().get("metrics", {}), best_acc)
            self.logger.warning(
                f"Resume with:  python train.py "
                f"--resume {self.ckpt.interrupted_path}"
            )
            return best_acc

        except Exception as exc:
            self.logger.error(f"Exception during training: {exc} -- saving emergency checkpoint.")
            self._emergency_save(locals().get("epoch", start_epoch),
                                 locals().get("metrics", {}), best_acc)
            raise  # re-raise so the full traceback is visible

    # -- emergency save -------------------------------------------------
    def _emergency_save(
        self, epoch: int, metrics: Dict[str, Any], best_acc: float
    ) -> None:
        try:
            state = self._build_state(epoch, metrics, best_acc)
            path = self.ckpt.save_interrupted(state)
            self.logger.warning(f"Emergency checkpoint saved to: {path}")
        except Exception as exc:  # last-resort guard
            self.logger.error(f"Failed to save emergency checkpoint: {exc}")
