"""Train one arm of the room centre-update comparison.

Three modes, selected by `training.center_mode`:

    none      cross-entropy only -- the control, which must reproduce the
              existing baseline run when given seed 42
    gradient  the variant used throughout this thesis
    wen       Algorithm 1 of Wen et al., exactly as published

The loop is deliberately a copy of the one in `fer_wen/`, adapted for the room
dataset and extended with the "none" mode. It is kept separate from the shared
trainer in `vit_s16_baseline/` for the same reason as before: a comparison of
update rules should not require editing the code that produced the results
being compared against.

The architecture and the centre-loss module are imported rather than copied.
Both have been checked numerically against their references, and a copy is a
thing that can drift.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
FER_WEN = HERE.parent / "fer_wen"
if str(FER_WEN) not in sys.path:
    sys.path.insert(0, str(FER_WEN))

from rooms_wen_data import build_dataloaders            # noqa: E402
from rooms_wen_model import build_model_from_config, count_parameters  # noqa: E402
from wen_center_loss import CenterLossVariant           # noqa: E402

MODES = ("none", "gradient", "wen")


# ----------------------------------------------------------------------
# utilities, identical in behaviour to the shared loop so the control can
# reproduce the run it is checked against
# ----------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state() -> Dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def build_scheduler(optimizer: torch.optim.Optimizer, t: Dict[str, Any]):
    """Linear warmup then cosine decay, stepped once per epoch."""
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def displacement_ratio(batch_size: int, num_classes: int, lam: float,
                       alpha: float, eta_c: float) -> float:
    """How much further Algorithm 1 moves a centre per step than the gradient
    variant, at the expected per-class count of a balanced batch.

    Algorithm 1 moves a centre by alpha / (1 + n_j) times the class sum; the
    gradient variant moves it by eta_c * lambda / B times the same sum. The
    ratio is therefore alpha * B / (eta_c * lambda * (1 + n_j)).

    alpha and eta_c are taken separately even though this project sets both
    from one config key. Writing the formula with a single rate would let it
    cancel, which is correct only for as long as the two happen to be equal,
    and would then be silently wrong for anyone who separates them.

    Logged with the run so the number that explains any difference between the
    two arms is recorded rather than reconstructed afterwards.
    """
    n_j = batch_size / num_classes
    return (alpha * batch_size) / (eta_c * lam * (1.0 + n_j))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one arm of the comparison.")
    p.add_argument("--config", default=str(HERE / "config_rooms_wen.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    t = cfg["training"]
    mode = str(t["center_mode"]).lower()
    if mode not in MODES:
        raise SystemExit(f"training.center_mode must be one of {MODES}, got {mode!r}")

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
    log(logfile, f"CENTER MODE: {mode} | seed: {seed}")
    # Recorded so a run can be tied to the exact environment that produced it.
    log(logfile, f"torch {torch.__version__} | cuda {torch.version.cuda} | "
                 f"cudnn {torch.backends.cudnn.version()} | "
                 f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    train_loader, val_loader, class_to_idx = build_dataloaders(
        cfg, pin=device.type == "cuda")
    log(logfile, f"Train images: {len(train_loader.dataset)} | "
                 f"Val images: {len(val_loader.dataset)} | classes: {class_to_idx}")
    log(logfile, f"Train class counts: {train_loader.dataset.class_counts()}")

    model = build_model_from_config(cfg["model"]).to(device)
    log(logfile, f"Model: ViT-S/16 from scratch ({cfg['model']['arch']}) | "
                 f"trainable params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(t.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(t["learning_rate"]),
                                  weight_decay=float(t["weight_decay"]))
    scheduler = build_scheduler(optimizer, t)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    lam = float(t["center_loss_weight"])
    rate = float(t["center_loss_lr"])
    num_classes = int(cfg["model"]["num_classes"])

    center_loss: Optional[CenterLossVariant] = None
    optimizer_center: Optional[torch.optim.Optimizer] = None
    if mode == "none":
        log(logfile, "Center Loss | disabled: cross-entropy only (control arm)")
    else:
        center_loss = CenterLossVariant(
            num_classes=num_classes,
            feat_dim=int(cfg["model"]["embed_dim"]),
            mode=mode).to(device)
        if mode == "gradient":
            optimizer_center = torch.optim.SGD(center_loss.parameters(), lr=rate)
            log(logfile, f"Center Loss | mode=gradient | lambda={lam} | "
                         f"eta_c={rate} | effective centre rate = "
                         f"eta_c * lambda = {rate * lam}")
        else:
            log(logfile, f"Center Loss | mode=wen (Algorithm 1) | lambda={lam} | "
                         f"alpha={rate} (independent of lambda) | "
                         f"per-class normalisation by (1 + n_j)")
        ratio = displacement_ratio(int(t["batch_size"]), num_classes, lam,
                                   alpha=rate, eta_c=rate)
        log(logfile, f"Centre displacement, Algorithm 1 relative to gradient, "
                     f"at the balanced per-class count "
                     f"n_j = {int(t['batch_size']) / num_classes:.2f}: "
                     f"{ratio:,.0f}x")

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
                if center_loss is None:
                    center_raw = torch.zeros((), device=device)
                    loss = ce
                else:
                    center_raw = center_loss(feats, targets)
                    loss = ce + lam * center_raw

            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            if optimizer_center is not None:
                # As in the shared loop: the scaler unscales the centre
                # gradients too, and those gradients carry lambda.
                scaler.step(optimizer_center)
            scaler.update()

            if mode == "wen":
                # Algorithm 1, applied after the model step, in float32 and
                # outside the graph, on this batch's embeddings.
                #
                # The guard is not decorative. Under mixed precision the scaler
                # silently skips an optimiser step whose gradients are not
                # finite, so the gradient mode recovers on its own. This manual
                # update has no such protection: one non-finite embedding would
                # write NaN into a centre, and since the centres are a running
                # quantity that NaN would never wash out. Failing loudly beats
                # reporting a corrupted run.
                if not torch.isfinite(feats).all():
                    raise FloatingPointError(
                        "Non-finite embeddings in the forward pass; the manual "
                        "centre update was aborted rather than writing NaN into "
                        "the centres. Re-run with training.use_amp: false.")
                center_loss.update_centers(feats, targets, alpha=rate)

            bs = images.size(0)
            loss_m.update(loss.item(), bs)
            ce_m.update(ce.item(), bs)
            center_m.update(float(center_raw), bs)
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

        lr_now = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
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
            "center_loss_state_dict": (center_loss.state_dict()
                                       if center_loss is not None else None),
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

    log(logfile, f"Training complete. Best val accuracy: {best_acc:.4f}")
    logfile.close()


if __name__ == "__main__":
    main()
