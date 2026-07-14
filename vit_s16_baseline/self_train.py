"""
self_train.py
=============
Entry point for PART 3: semi-supervised self-labelling.

This is a SEPARATE pipeline from the supervised baseline (`train.py`), so the
Parts 1-2 workflow stays runnable and untouched. It reuses the same building
blocks (ViT, dataset, Trainer's supervised step, checkpoints) and adds the
online self-training loop from `src/semi_trainer.py`.

Scenario (Anexa 2, point 3): start from a model trained on the labelled data,
then add the "unlabelled" pool (the held-out 1500/class), self-label it in real
time with the network's current knowledge, and keep the confident predictions.

Usage
-----
    python self_train.py --config config_selftrain.yaml \
        --warmup /path/to/center/best_model.pt

The config is a normal training config plus an `ssl:` section:

    ssl:
      unlabeled_dir: /content/dataset_split/unlabeled   # required
      tau: 0.95                # confidence threshold
      rounds: 6                # self-labelling rounds
      epochs_per_round: 5      # supervised epochs per round
      use_center_loss: true    # spec asks to keep center loss on
      center_loss_weight: 0.0005
      center_loss_lr: 0.5
      labeled_per_class: null  # null = use all labelled; N = few-labels (Exp.2),
                               #   leftover labelled images join the unlabelled pool
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.center_loss import CenterLoss
from src.checkpoint import CheckpointManager, load_checkpoint
from src.dataset import build_dataset
from src.logger import setup_logger
from src.semi_trainer import SelfTrainingTrainer
from src.trainer import Trainer, build_scheduler
from src.utils import (
    count_parameters,
    get_amp_components,
    get_device,
    load_config,
    save_config,
    set_seed,
)
from src.vit import build_vit_from_config
from train import build_optimizer, create_run_dir  # reuse, don't duplicate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 3: semi-supervised self-training.")
    parser.add_argument("--config", type=str, default="config_selftrain.yaml",
                        help="Path to a YAML config that includes an `ssl:` section.")
    parser.add_argument("--warmup", type=str, default=None,
                        help="Checkpoint whose model weights seed self-training "
                             "(e.g. the center-loss best_model.pt). Optional.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override the run_name used for the folder name.")
    return parser.parse_args()


def _apply_ssl_training_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fold the ssl knobs into `training` so Trainer/scheduler stay consistent.

    Total supervised epochs = rounds * epochs_per_round (drives the LR schedule).
    Center loss is toggled here from the ssl section.
    """
    ssl = config.setdefault("ssl", {})
    tcfg = config["training"]
    tcfg["epochs"] = int(ssl.get("rounds", 6)) * int(ssl.get("epochs_per_round", 5))
    # A warm-started model should usually fine-tune at a gentler LR than the
    # from-scratch baseline; expose it as ssl.learning_rate (None = keep default).
    if ssl.get("learning_rate") is not None:
        tcfg["learning_rate"] = float(ssl["learning_rate"])
    if bool(ssl.get("use_center_loss", True)):
        tcfg["use_center_loss"] = True
        tcfg["center_loss_weight"] = float(ssl.get("center_loss_weight",
                                                   tcfg.get("center_loss_weight", 0.0005)))
        tcfg["center_loss_lr"] = float(ssl.get("center_loss_lr", 0.5))
    else:
        tcfg["use_center_loss"] = False
    return ssl


def _subsample_labeled(
    labeled_ds, unlab_infer_ds, unlab_train_ds, labeled_per_class: int, seed: int
) -> None:
    """Few-labels (Exp.2): keep `labeled_per_class` labelled images per class and
    move the rest into the unlabelled pool (appended, in the SAME order, to both
    the inference and train views so their indices stay aligned).

    Mutates the datasets in place via their public `.samples` lists.
    """
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for i, (_path, y) in enumerate(labeled_ds.samples):
        by_class.setdefault(y, []).append(i)

    keep: List[int] = []
    leftover: List[int] = []
    for y, idxs in by_class.items():
        rng.shuffle(idxs)
        keep.extend(idxs[:labeled_per_class])
        leftover.extend(idxs[labeled_per_class:])

    leftover_samples = [labeled_ds.samples[i] for i in sorted(leftover)]
    labeled_ds.samples = [labeled_ds.samples[i] for i in sorted(keep)]
    unlab_infer_ds.samples = unlab_infer_ds.samples + leftover_samples
    unlab_train_ds.samples = unlab_train_ds.samples + leftover_samples


def build_semi_datasets(config: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Dict[str, int]]:
    """Return (labeled_train_ds, unlab_infer_ds, unlab_train_ds, val_ds, class_to_idx).

    unlab_infer_ds uses the EVAL transform (deterministic, for self-labelling);
    unlab_train_ds uses the TRAIN transform (augmented, for training on pseudo-
    labels). Both are built over the same directory so their sample order (and
    thus indices) match.
    """
    data = config["data"]
    ssl = config["ssl"]
    unlab_dir = ssl["unlabeled_dir"]

    labeled_train_ds = build_dataset(data["train_dir"], config, train=True)
    unlab_infer_ds = build_dataset(unlab_dir, config, train=False)
    unlab_train_ds = build_dataset(unlab_dir, config, train=True)

    lpc = ssl.get("labeled_per_class")
    if lpc:
        _subsample_labeled(labeled_train_ds, unlab_infer_ds, unlab_train_ds,
                           int(lpc), int(config["training"]["seed"]))

    # Safety net for the index-alignment invariant the pseudo-labelling relies on:
    # the eval-view and train-view of the pool must list the same paths in the same
    # order (so a confident index from inference maps to the same training image).
    assert [p for p, _ in unlab_infer_ds.samples] == [p for p, _ in unlab_train_ds.samples], \
        "unlabelled inference/train views are not index-aligned"

    val_dir = data.get("val_dir") or data["eval_dir"]
    val_ds = build_dataset(val_dir, config, train=False)
    return labeled_train_ds, unlab_infer_ds, unlab_train_ds, val_ds, labeled_train_ds.class_to_idx


def main() -> None:
    args = parse_args()
    device = get_device()

    config = load_config(args.config)
    ssl = _apply_ssl_training_overrides(config)
    set_seed(int(config["training"]["seed"]))

    run_name = args.run_name or config.get("run_name", "vit_s16_selftrain")
    run_dir = create_run_dir(config["paths"]["output_root"], run_name)
    logger = setup_logger(run_dir / "logs" / "train.log", name="vit_selftrain")
    save_config(config, run_dir / "logs" / "config_used.yaml")
    logger.info(f"New self-training run folder: {run_dir}")
    logger.info(f"Device: {device}")

    # -- data ----------------------------------------------------------
    labeled_ds, unlab_infer_ds, unlab_train_ds, val_ds, class_to_idx = build_semi_datasets(config)
    bs = int(config["training"]["batch_size"])
    nw = int(config["training"]["num_workers"])
    pin = device.type == "cuda"
    loader_kw = dict(num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    unlab_infer_loader = DataLoader(unlab_infer_ds, batch_size=bs, shuffle=False, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kw)
    logger.info(
        f"Labelled: {len(labeled_ds)} | Unlabelled pool: {len(unlab_infer_ds)} | "
        f"Val: {len(val_ds)} | classes: {class_to_idx}"
    )

    # -- model / optim / loss (mirrors train.py) -----------------------
    model = build_vit_from_config(config["model"]).to(device)
    warm_ckpt = None
    if args.warmup:
        warm_ckpt = load_checkpoint(args.warmup, map_location="cpu")
        model.load_state_dict(warm_ckpt["model_state_dict"])
        logger.info(f"Warm-started model weights from: {args.warmup}")
    else:
        logger.warning("No --warmup checkpoint given: self-labelling starts from a "
                       "randomly initialised model (pseudo-labels will be poor early).")
    logger.info(f"Model: ViT-S/16 | trainable params: {count_parameters(model):,}")

    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])
    scaler, autocast_ctx = get_amp_components(device, config["training"].get("use_amp", False))
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["training"].get("label_smoothing", 0.0))
    )

    center_loss = None
    optimizer_center = None
    if bool(config["training"].get("use_center_loss", False)):
        center_loss = CenterLoss(
            num_classes=int(config["model"]["num_classes"]),
            feat_dim=int(config["model"]["embed_dim"]),
        ).to(device)
        optimizer_center = torch.optim.SGD(
            center_loss.parameters(),
            lr=float(config["training"].get("center_loss_lr", 0.5)),
        )
        logger.info(f"Center Loss ENABLED | weight={config['training']['center_loss_weight']} "
                    f"| center_lr={config['training']['center_loss_lr']}")
        # Restore the LEARNED class centers from the warm-start checkpoint. Without
        # this the model would resume from center-aligned features but pull them
        # toward freshly random centers, undoing the warm start. optimizer_center
        # stays fresh (its SGD has no momentum, so there is no state to carry).
        if warm_ckpt is not None and warm_ckpt.get("center_loss_state_dict") is not None:
            center_loss.load_state_dict(warm_ckpt["center_loss_state_dict"])
            logger.info("Restored Center Loss centers from the warm-start checkpoint.")

    ckpt_manager = CheckpointManager(run_dir / "checkpoints")
    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, autocast_ctx=autocast_ctx, device=device, config=config,
        class_to_idx=class_to_idx, logger=logger,
        csv_logger=None,  # unused: SelfTrainingTrainer owns its own CSV; fit() is not called
        ckpt_manager=ckpt_manager, center_loss=center_loss, optimizer_center=optimizer_center,
    )

    # -- self-training -------------------------------------------------
    semi = SelfTrainingTrainer(
        trainer=trainer, labeled_train_ds=labeled_ds,
        unlabeled_infer_loader=unlab_infer_loader, unlabeled_train_ds=unlab_train_ds,
        val_loader=val_loader, run_dir=run_dir, ssl_cfg=ssl, logger=logger,
    )
    best = semi.run()

    print("\n" + "=" * 70)
    print("SELF-TRAINING FINISHED")
    print("=" * 70)
    print(f"Run folder:      {run_dir}")
    print(f"Best val acc:    {best:.4f}")
    print(f"Best model:      {ckpt_manager.best_path}")
    print(f"Metrics CSV:     {run_dir / 'logs' / 'selftrain_metrics.csv'}")
    print("\nEvaluate the best model on the TEST set:")
    print(f"    python evaluate.py --checkpoint {ckpt_manager.best_path} "
          f"--data-dir <test_dir> --save-embeddings")
    print("=" * 70)


if __name__ == "__main__":
    main()
