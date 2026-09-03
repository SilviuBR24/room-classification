"""
train_wen.py
============
Trains one variant of the Center Loss update-rule comparison.

    python train_wen.py --config config_wen.yaml

This folder has its own training loop rather than borrowing the shared one,
because the shared loop cannot express Algorithm 1: it steps the centres
through an optimiser on the combined objective, which is precisely the
behaviour under test.

The loop is therefore written to match the shared one exactly in every other
respect -- optimiser, schedule, mixed precision, gradient clipping, checkpoint
selection, seeding -- so that the "gradient" variant reproduces the result the
shared loop already produced. That control run is what licenses attributing any
difference in the "wen" run to the update rule.

Where the two modes diverge, and only there:

  gradient : centres are parameters; their gradient comes from
             lambda * L_C via autograd; a separate SGD optimiser steps them,
             with the scaler unscaling as for the model.

  wen      : centres are a buffer; no gradient reaches them; after the model
             step, `update_centers` applies Algorithm 1 explicitly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

from fer_wen_data import build_dataloaders
from fer_wen_model import build_model_from_config, count_parameters
from wen_center_loss import CenterLossVariant

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# small utilities, kept local so this folder depends on nothing
# ----------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> Dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def build_scheduler(optimizer: torch.optim.Optimizer, t: Dict[str, Any]):
    """Linear warmup then cosine decay, stepped once per epoch.

    Identical in form to the shared project's scheduler, so the control run can
    reproduce it.
    """
    if t.get("scheduler", "cosine") in (None, "none"):
        return None
    epochs = int(t["epochs"])
    warmup = int(t.get("warmup_epochs", 0))

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


class Meter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.n = 0

    def update(self, value: float, count: int = 1) -> None:
        self.sum += float(value) * count
        self.n += count

    @property
    def avg(self) -> float:
        return self.sum / self.n if self.n else 0.0


def log(fh, message: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {message}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def save_atomic(state: Dict[str, Any], path: Path) -> None:
    """Write to a temporary file and rename, so an interruption cannot leave a
    truncated file where a valid checkpoint used to be."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one Center Loss variant.")
    p.add_argument("--config", default=str(HERE / "config_wen.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    t = cfg["training"]
    mode = str(t["center_mode"]).lower()
    if mode not in ("gradient", "wen"):
        raise SystemExit(f"training.center_mode must be 'gradient' or 'wen', got {mode!r}")

    seed = int(t.get("seed", 42))
    set_seed(seed)

    run_dir = (Path(t.get("output_root") or cfg["paths"]["output_root"])
               / f"{datetime.now():%Y-%m-%d_%H-%M}_{cfg['run_name']}")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "logs" / "config_used.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(t.get("use_amp", True)) and device.type == "cuda"

    logfile = open(run_dir / "logs" / "train.log", "a", encoding="utf-8")
    log(logfile, f"Run directory: {run_dir}")
    log(logfile, f"Device: {device} | AMP: {use_amp}")
    log(logfile, f"CENTER MODE: {mode}")

    train_loader, val_loader, class_to_idx = build_dataloaders(cfg, pin=device.type == "cuda")
    log(logfile, f"Train images: {len(train_loader.dataset)} | "
                 f"Val images: {len(val_loader.dataset)} | classes: {class_to_idx}")
    log(logfile, f"Train class counts: {train_loader.dataset.class_counts()}")

    model = build_model_from_config(cfg["model"]).to(device)
    log(logfile, f"Model: ResNet-18 for FER2013 ({cfg['model']['arch']}) | "
                 f"trainable params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(t.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(t["learning_rate"]),
                                  weight_decay=float(t["weight_decay"]))
    scheduler = build_scheduler(optimizer, t)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    lam = float(t["center_loss_weight"])
    rate = float(t["center_loss_lr"])
    center_loss = CenterLossVariant(num_classes=int(cfg["model"]["num_classes"]),
                                    feat_dim=int(cfg["model"]["embed_dim"]),
                                    mode=mode).to(device)
    optimizer_center: Optional[torch.optim.Optimizer] = None
    if mode == "gradient":
        optimizer_center = torch.optim.SGD(center_loss.parameters(), lr=rate)
        log(logfile, f"Center Loss | mode=gradient | lambda={lam} | eta_c={rate} "
                     f"| effective centre rate = eta_c * lambda = {rate * lam}")
    else:
        log(logfile, f"Center Loss | mode=wen (Algorithm 1) | lambda={lam} | "
                     f"alpha={rate} (independent of lambda) | "
                     f"per-class normalisation by (1 + n_j)")

    metrics_path = run_dir / "logs" / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        fh.write("epoch,lr,train_loss,train_ce,train_center_raw,train_accuracy,"
                 "val_loss,val_accuracy,seconds\n")

    epochs = int(t["epochs"])
    grad_clip = t.get("grad_clip")
    best_acc = 0.0
    log(logfile, f"Starting training: epochs 1..{epochs}")

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        loss_m, ce_m, center_m, acc_m = Meter(), Meter(), Meter(), Meter()

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if optimizer_center is not None:
                optimizer_center.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits, feats = model(images, return_embeddings=True)
                ce = criterion(logits, targets)
                center_raw = center_loss(feats, targets)
                loss = ce + lam * center_raw

            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            if optimizer_center is not None:
                # Same arrangement as the shared loop: the scaler unscales the
                # centre gradients too, and those gradients carry lambda.
                scaler.step(optimizer_center)
            scaler.update()

            if mode == "wen":
                # Algorithm 1, applied after the model step, in float32 and
                # outside the graph. Uses the embeddings of this batch, which
                # is what the paper specifies.
                center_loss.update_centers(feats, targets, alpha=rate)

            bs = images.size(0)
            loss_m.update(loss.item(), bs)
            ce_m.update(ce.item(), bs)
            center_m.update(center_raw.item(), bs)
            acc_m.update(accuracy(logits.detach(), targets), bs)

        model.eval()
        vloss_m, vacc_m = Meter(), Meter()
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(images)
                    vloss = criterion(logits, targets)
                bs = images.size(0)
                vloss_m.update(vloss.item(), bs)
                vacc_m.update(accuracy(logits, targets), bs)

        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        secs = time.time() - t0

        improved = vacc_m.avg > best_acc
        if improved:
            best_acc = vacc_m.avg

        log(logfile,
            f"Epoch {epoch + 1:3d}/{epochs} | lr={lr_now:.2e} | "
            f"train_loss={loss_m.avg:.4f} train_acc={acc_m.avg:.4f} | "
            f"val_loss={vloss_m.avg:.4f} val_acc={vacc_m.avg:.4f} | "
            f"best_acc={best_acc:.4f} | {secs:.1f}s"
            + ("  <-- new best" if improved else ""))

        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(f"{epoch + 1},{lr_now},{loss_m.avg},{ce_m.avg},"
                     f"{center_m.avg},{acc_m.avg},{vloss_m.avg},{vacc_m.avg},{secs}\n")

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "center_loss_state_dict": center_loss.state_dict(),
            "optimizer_center_state_dict": (optimizer_center.state_dict()
                                            if optimizer_center else None),
            "center_mode": mode,
            "best_val_accuracy": best_acc,
            "config": cfg,
            "seed": seed,
            "class_to_idx": class_to_idx,
            "rng_state": rng_state(),
        }
        save_atomic(state, run_dir / "checkpoints" / "last_checkpoint.pt")
        if improved:
            save_atomic(state, run_dir / "checkpoints" / "best_model.pt")

    log(logfile, f"Training complete. Best validation accuracy: {best_acc:.4f}")
    with open(run_dir / "logs" / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({"run_name": cfg["run_name"], "center_mode": mode,
                   "best_val_accuracy": best_acc, "epochs": epochs,
                   "lambda": lam, "center_rate": rate}, fh, indent=2)
    logfile.close()
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()
