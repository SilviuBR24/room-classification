"""Train one arm of the room centre-update comparison.

Three modes, selected by `training.center_mode`:

    none      cross-entropy only -- the control, which at seed 42 should
              closely match the existing baseline run. It is a consistency
              check, not a bit-for-bit reproduction: the historical run
              used a different runtime and GPU.
    gradient  the variant used throughout this thesis
    wen       the original centre-update rule, Equation (4) and Algorithm 1 of Wen et al.

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
import json
import math
import os
import random
import shutil
import subprocess
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
    """Ratio of the two rules' centre-update coefficients, at the expected
    per-class count of a balanced batch.

    This is a property of the hyperparameters, not a measurement. It says
    how much further one rule would move a centre than the other given the
    same class sum, at E[n_j]. It is not the ratio of the displacements two
    trained models actually produce: once the arms diverge they see different
    embeddings, and n_j fluctuates from batch to batch, so the realised ratio
    varies. The measured displacements are logged separately each epoch.

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


def provenance() -> Dict[str, Any]:
    """What produced this run: which code, and in what environment.

    Nine runs across several Colab sessions can easily be produced by different
    commits and different runtimes. Comparing configurations does not catch
    that, because the configuration can be identical while the code that read it
    is not. This is recorded with every run and checked before one is reused.
    """
    def sh(*args: str) -> Optional[str]:
        try:
            r = subprocess.run(args, cwd=str(HERE), capture_output=True,
                               text=True, timeout=30)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    commit = sh("git", "rev-parse", "HEAD")
    status = sh("git", "status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu"),
        "python": sys.version.split()[0],
    }


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

    # Seconds in the name, and the directory must not already exist. The
    # minute-resolution name plus exist_ok=True meant two arms started in the
    # same minute would have written into one directory, mixing their logs and
    # overwriting each other's checkpoints.
    root = Path(t.get("output_root") or cfg["paths"]["output_root"])
    run_dir = root / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{cfg['run_name']}"
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists; refusing to write into it")
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    shutil.copy2(args.config, run_dir / "logs" / "config_used.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(t.get("use_amp", True)) and device.type == "cuda"

    prov = provenance()
    with open(run_dir / "logs" / "provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)

    logfile = open(run_dir / "logs" / "train.log", "a", encoding="utf-8")
    log(logfile, f"Run directory: {run_dir}")
    log(logfile, f"Code: commit {prov['git_commit']} "
                 f"(dirty={prov['git_dirty']})")
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
        # The centres are drawn from the global generator, which the control arm
        # never touches. Left alone, that would leave the three arms in
        # different global RNG states before training starts, and any
        # randomness taken from the global generator afterwards -- augmentations
        # applied in this process when num_workers is 0, dropout, anything added
        # later -- would differ between arms for the same seed. The state is
        # therefore captured and put back, so the draw is identical in the two
        # centre arms and invisible to everything that follows.
        _global_rng = torch.get_rng_state()
        _cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        center_loss = CenterLossVariant(
            num_classes=num_classes,
            feat_dim=int(cfg["model"]["embed_dim"]),
            mode=mode).to(device)
        torch.set_rng_state(_global_rng)
        if _cuda_rng is not None:
            torch.cuda.set_rng_state_all(_cuda_rng)
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
        log(logfile,
            f"Coefficient ratio, Wen rule relative to the gradient variant, at "
            f"E[n_j] = {int(t['batch_size']) / num_classes:.2f}: {ratio:,.0f}x. "
            f"A property of the hyperparameters, not a measurement: n_j varies "
            f"per batch, and once the arms diverge they see different "
            f"embeddings. The displacements actually produced are logged each "
            f"epoch as centre_shift_mean / centre_shift_max.")

    metrics_path = run_dir / "logs" / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        fh.write("epoch,lr,train_loss,train_ce,train_center_raw,"
                 "train_center_weighted,train_accuracy,val_loss,val_accuracy,"
                 "centre_norm_mean,centre_norm_max,centre_shift_mean,"
                 "centre_shift_max,amp_steps_skipped,seconds\n")

    epochs = int(t["epochs"])
    grad_clip = t.get("grad_clip")
    # Starts below zero, not at zero: with 0.0 an epoch scoring exactly 0.0
    # would never count as an improvement and best_model.pt would never be
    # written, leaving the run without the checkpoint the evaluation needs.
    best_acc = -1.0
    log(logfile, f"Starting training: epochs 1..{epochs}")

    total_skipped = 0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        loss_m, ce_m, center_m, acc_m = Meter(), Meter(), Meter(), Meter()
        shift_mean_m = Meter()
        shift_max = 0.0
        skipped_steps = 0

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

            # Unscale both sets of gradients before deciding anything, so the
            # finiteness test below sees real values rather than scaled ones.
            scaler.unscale_(optimizer)
            if optimizer_center is not None:
                scaler.unscale_(optimizer_center)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            # GradScaler decides per optimiser, not once per iteration. With the
            # centres held in a second optimiser, the model's step can be skipped
            # for overflowing gradients while the centres, whose gradients are
            # finite, step anyway -- measured on this build: the network did not
            # move and the centres did. That asymmetry would make the two arms
            # differ by more than their update rule, so the centre step is
            # neutralised by hand whenever the model's step will be skipped.
            #
            # The gradients are zeroed rather than the step being skipped
            # outright, because the scaler only lowers its scale when a step it
            # was asked to take found the overflow. Skipping both steps and
            # calling update() leaves the scale where it was -- measured: it
            # stayed at 65536 -- and every following batch would overflow again.
            def _finite(params) -> bool:
                return all(torch.isfinite(q.grad).all()
                           for q in params if q.grad is not None)

            model_finite = _finite(model.parameters())
            centres_finite = (optimizer_center is None
                              or _finite(center_loss.parameters()))
            model_finite = model_finite and centres_finite

            if not model_finite:
                skipped_steps += 1
                # Whichever side still has finite gradients would otherwise
                # step on its own, so its gradients are removed. Both
                # directions matter: an overflow in the centre gradients would
                # otherwise let the network advance while the centres stood
                # still, which is the same asymmetry as the reverse case.
                #
                # Removed, not zeroed. AdamW applies its decoupled weight decay
                # to every parameter it is given, gradient or not, so a zero
                # gradient still shrinks the weights -- measured: the model
                # moved anyway. A gradient of None makes the optimiser skip the
                # parameter entirely.
                for q in model.parameters():
                    q.grad = None
                if optimizer_center is not None:
                    for q in center_loss.parameters():
                        q.grad = None

            centres_before = (center_loss.centers.detach().clone()
                              if center_loss is not None else None)
            scaler.step(optimizer)
            if optimizer_center is not None:
                scaler.step(optimizer_center)
            scaler.update()

            step_taken = model_finite

            if mode == "wen":
                # Algorithm 1, applied after the model step, in float32 and
                # outside the graph, on this batch's embeddings.
                #
                # Two conditions, and neither is decorative:
                #
                # The step guard. The gradient arm is protected by the scaler
                # itself: when a step is skipped, neither the weights nor the
                # centres move. This manual update has no such protection, so
                # without the guard a skipped batch would leave the network
                # unchanged while still advancing the centres -- the two arms
                # would then no longer differ only in the update rule.
                # Embeddings can be perfectly finite while the gradients that
                # overflow are not, so checking the embeddings alone does not
                # detect this.
                #
                # The finiteness guard. One non-finite embedding would write NaN
                # into a centre, and since the centres are a running quantity
                # that NaN would never wash out. Failing loudly beats reporting
                # a corrupted run.
                if step_taken:
                    if not torch.isfinite(feats).all():
                        raise FloatingPointError(
                            "Non-finite embeddings in the forward pass; the "
                            "manual centre update was aborted rather than "
                            "writing NaN into the centres. Re-run with "
                            "training.use_amp: false.")
                    center_loss.update_centers(feats, targets, alpha=rate)
                    if not torch.isfinite(center_loss.centers).all():
                        raise FloatingPointError(
                            "The centre update produced a non-finite centre. "
                            "The run is stopped rather than continued with a "
                            "value that can never wash out.")

            # Measured for both centre arms, so the displacement the two rules
            # actually produce can be compared against the theoretical ratio.
            if center_loss is not None:
                shift = (center_loss.centers.detach() - centres_before).norm(dim=1)
                shift_mean_m.update(float(shift.mean()))
                shift_max = max(shift_max, float(shift.max()))

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

        total_skipped += skipped_steps
        centre_norm_mean = centre_norm_max = 0.0
        if center_loss is not None:
            norms = center_loss.centers.detach().norm(dim=1)
            centre_norm_mean, centre_norm_max = float(norms.mean()), float(norms.max())

        extra = ""
        if center_loss is not None:
            extra = (f" | centre_norm={centre_norm_mean:.3f}/{centre_norm_max:.3f}"
                     f" shift={shift_mean_m.avg:.2e}/{shift_max:.2e}"
                     f" lam*center={lam * center_m.avg:.4f}")
        if skipped_steps:
            extra += f" | AMP skipped {skipped_steps}"

        log(logfile,
            f"Epoch {epoch + 1:3d}/{epochs} | lr={lr_now:.2e} | "
            f"train_loss={loss_m.avg:.4f} train_acc={acc_m.avg:.4f} | "
            f"val_loss={vloss_m.avg:.4f} val_acc={vacc_m.avg:.4f} | "
            f"best_acc={best_acc:.4f} | {secs:.1f}s" + extra
            + ("  <-- new best" if improved else ""))

        if not all(map(math.isfinite, (loss_m.avg, vloss_m.avg))):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch + 1}: "
                f"train {loss_m.avg}, val {vloss_m.avg}")

        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(f"{epoch + 1},{lr_now},{loss_m.avg},{ce_m.avg},"
                     f"{center_m.avg},{lam * center_m.avg},{acc_m.avg},"
                     f"{vloss_m.avg},{vacc_m.avg},{centre_norm_mean},"
                     f"{centre_norm_max},{shift_mean_m.avg},{shift_max},"
                     f"{skipped_steps},{secs}\n")

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
            "provenance": prov,
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
    log(logfile, f"AMP steps skipped in total: {total_skipped}")
    logfile.close()


if __name__ == "__main__":
    main()
