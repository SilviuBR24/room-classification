"""
train.py
========
Entry point for SUPERVISED ViT-S/16 baseline training.

Usage
-----
Fresh run:
    python train.py --config config.yaml

Resume after an interruption (continues from the NEXT epoch in the SAME run
folder inferred from the checkpoint location):
    python train.py --resume runs/<run>/checkpoints/last_checkpoint.pt

What it does
------------
1. Loads config (or, on resume, the config stored inside the checkpoint).
2. Sets the seed for reproducibility.
3. Creates a unique experiment folder:
       <output_root>/<YYYY-MM-DD_HH-MM>_<run_name>/
   containing checkpoints/, logs/, outputs/ and a copy of config_used.yaml.
4. Builds dataloaders, the from-scratch ViT, AdamW, the LR scheduler and AMP.
5. Trains, checkpointing every epoch and on interruption.

Assumptions are stated inline and summarized in the project README/answer.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from src.checkpoint import CheckpointManager, load_checkpoint
from src.dataset import build_dataloaders
from src.logger import MetricsCSVLogger, setup_logger
from src.trainer import Trainer, build_scheduler
from src.utils import (
    count_parameters,
    ensure_dir,
    get_amp_components,
    get_device,
    load_config,
    save_config,
    set_rng_state,
    set_seed,
)
from src.vit import build_vit_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViT-S/16 baseline.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML config (ignored on --resume).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override the run_name used for the folder name.")
    return parser.parse_args()


def create_run_dir(output_root: str, run_name: str) -> Path:
    """Create a unique, timestamped experiment folder with subdirectories."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = Path(output_root) / f"{timestamp}_{run_name}"
    # In the unlikely event of a name clash (same minute), add seconds.
    if run_dir.exists():
        run_dir = Path(output_root) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{run_name}"
    for sub in ("checkpoints", "logs", "outputs"):
        ensure_dir(run_dir / sub)
    return run_dir


def build_optimizer(model: nn.Module, training_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW with the configured learning rate and weight decay."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )


def main() -> None:
    args = parse_args()
    device = get_device()

    # ------------------------------------------------------------------
    # Resolve config + run folder (fresh vs resume)
    # ------------------------------------------------------------------
    resuming = args.resume is not None
    checkpoint: Dict[str, Any] = {}

    if resuming:
        ckpt_path = Path(args.resume)
        checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
        # Use the config stored in the checkpoint to guarantee architecture match.
        config = checkpoint["config"]
        # Infer the run folder as <run_dir>/checkpoints/<ckpt>.pt -> <run_dir>.
        run_dir = ckpt_path.resolve().parent.parent
        for sub in ("checkpoints", "logs", "outputs"):
            ensure_dir(run_dir / sub)
        print(f"[train] Resuming in existing run folder: {run_dir}")
    else:
        config = load_config(args.config)
        run_name = args.run_name or config.get("run_name", "vit_s16_baseline")
        output_root = config["paths"]["output_root"]
        run_dir = create_run_dir(output_root, run_name)
        print(f"[train] New run folder: {run_dir}")

    # ------------------------------------------------------------------
    # Seed + logging
    # ------------------------------------------------------------------
    seed = int(config["training"]["seed"])
    set_seed(seed, deterministic=True)

    logger = setup_logger(run_dir / "logs" / "train.log")
    csv_logger = MetricsCSVLogger(run_dir / "logs" / "metrics.csv", resume=resuming)

    # Persist the exact config actually used (root + logs copies).
    save_config(config, run_dir / "config_used.yaml")
    save_config(config, run_dir / "logs" / "config_used.yaml")

    logger.info(f"Device: {device}")
    logger.info(f"Run directory: {run_dir}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    try:
        train_loader, eval_loader, class_to_idx = build_dataloaders(config, device)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Dataset error: {exc}")
        raise
    logger.info(
        f"Train images: {len(train_loader.dataset)} | "
        f"Eval images: {len(eval_loader.dataset)} | classes: {class_to_idx}"
    )

    # ------------------------------------------------------------------
    # Model / optimizer / scheduler / AMP / loss
    # ------------------------------------------------------------------
    model = build_vit_from_config(config["model"]).to(device)
    logger.info(f"Model: ViT-S/16 from scratch | trainable params: {count_parameters(model):,}")

    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])
    scaler, autocast_ctx = get_amp_components(device, config["training"].get("use_amp", False))
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["training"].get("label_smoothing", 0.0))
    )

    # ------------------------------------------------------------------
    # Restore state on resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_acc = 0.0
    if resuming:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as exc:
                logger.warning(f"Could not restore AMP scaler state: {exc}")
        if checkpoint.get("rng_state") is not None:
            set_rng_state(checkpoint["rng_state"])
        best_acc = float(checkpoint.get("best_eval_accuracy", 0.0))
        start_epoch = int(checkpoint["epoch"]) + 1  # continue at the NEXT epoch
        logger.info(f"Resumed from epoch {checkpoint['epoch']} -> starting at epoch {start_epoch + 1}. "
                    f"Best acc so far: {best_acc:.4f}")
        if start_epoch >= int(config["training"]["epochs"]):
            logger.info("Nothing to do: checkpoint already reached the configured epoch count.")
            csv_logger.close()
            return

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    ckpt_manager = CheckpointManager(run_dir / "checkpoints")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        autocast_ctx=autocast_ctx,
        device=device,
        config=config,
        class_to_idx=class_to_idx,
        logger=logger,
        csv_logger=csv_logger,
        ckpt_manager=ckpt_manager,
    )

    best = trainer.fit(train_loader, eval_loader, start_epoch=start_epoch, best_acc=best_acc)
    csv_logger.close()

    # ------------------------------------------------------------------
    # Final instructions
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Run folder:        {run_dir}")
    print(f"Best eval acc:     {best:.4f}")
    print(f"Best model:        {ckpt_manager.best_path}")
    print(f"Last checkpoint:   {ckpt_manager.last_path}")
    print(f"Metrics CSV:       {run_dir / 'logs' / 'metrics.csv'}")
    print("\nResume (if interrupted):")
    print(f"    python train.py --resume {ckpt_manager.last_path}")
    print("\nEvaluate the best model:")
    print(f"    python evaluate.py --checkpoint {ckpt_manager.best_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
