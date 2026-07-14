"""
src/semi_trainer.py
===================
Online self-training engine for Part 3 (semi-supervised / self-labelling).

The loop, per ROUND, self-labels the unlabelled pool with the network's CURRENT
weights, then trains a few epochs on (real labels + confident pseudo-labels):

    for round r in 1..R:
        conf, pred = infer(model, unlabelled)          # current knowledge
        keep pred where conf >= tau  -> pseudo-labels
        combined = labelled  ++  pseudo-labelled(kept)
        train E epochs on combined (CE [+ center loss]), pick best on val
        (next round re-labels with the improved model)

Design note -- REUSE, not reimplementation: the supervised step (CrossEntropy,
optional Center Loss, AMP, optimizer) already lives in `Trainer`. This class
*composes* a Trainer and calls `train_one_epoch` / `evaluate` directly, so there
is exactly one supervised training code path in the project. It only adds the
between-round pseudo-labelling and its own CSV schema (rounds/pseudo-accuracy),
leaving the supervised `MetricsCSVLogger` untouched.

Checkpoints are written via the same CheckpointManager + state format as the
baseline, so `evaluate.py` reads a self-training `best_model.pt` unchanged.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.pseudo_label import build_pseudo_subset, infer_confidence, pseudo_label_stats
from src.trainer import Trainer


class SelfTrainingMetricsLogger:
    """Append one row per (round, epoch) with the semi-supervised schema."""

    FIELDS = [
        "round", "epoch",
        "num_pseudo", "coverage", "pseudo_accuracy", "mean_conf",
        "train_loss", "train_ce", "train_center_raw", "train_accuracy",
        "val_loss", "val_accuracy",
        "learning_rate", "checkpoint_path",
    ]

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        if self._file.tell() == 0:
            self._writer.writeheader()
            self._file.flush()

    def log(self, **row: Any) -> None:
        def fmt(v: Any) -> Any:
            return "" if v is None else (f"{v:.6f}" if isinstance(v, float) else v)

        self._writer.writerow({k: fmt(row.get(k)) for k in self.FIELDS})
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


class SelfTrainingTrainer:
    """Drive the online self-labelling rounds around a supervised `Trainer`."""

    def __init__(
        self,
        trainer: Trainer,
        labeled_train_ds: Dataset,
        unlabeled_infer_loader: DataLoader,
        unlabeled_train_ds: Dataset,
        val_loader: DataLoader,
        run_dir: Path,
        ssl_cfg: Dict[str, Any],
        logger: Any,
    ) -> None:
        self.trainer = trainer
        self.labeled_train_ds = labeled_train_ds
        self.unlabeled_infer_loader = unlabeled_infer_loader
        self.unlabeled_train_ds = unlabeled_train_ds
        self.val_loader = val_loader
        self.run_dir = Path(run_dir)
        self.logger = logger

        self.tau = float(ssl_cfg.get("tau", 0.95))
        self.rounds = int(ssl_cfg.get("rounds", 6))
        self.epochs_per_round = int(ssl_cfg.get("epochs_per_round", 5))

        tcfg = trainer.config["training"]
        self.batch_size = int(tcfg["batch_size"])
        self.num_workers = int(tcfg["num_workers"])
        self.pin = trainer.device.type == "cuda"

        self.csv = SelfTrainingMetricsLogger(run_dir / "logs" / "selftrain_metrics.csv")

    # -- combined labelled + pseudo-labelled loader for one round ---------
    def _combined_loader(self, pseudo_subset: Dataset) -> DataLoader:
        dataset: Dataset = (
            ConcatDataset([self.labeled_train_ds, pseudo_subset])
            if len(pseudo_subset) > 0
            else self.labeled_train_ds
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def _save_best_and_last(
        self, global_epoch: int, metrics: Dict[str, Any], best_acc: float, is_best: bool
    ) -> Path:
        # Reuse the baseline checkpoint format so evaluate.py works unchanged.
        state = self.trainer._build_state(global_epoch, metrics, best_acc)
        last = self.trainer.ckpt.save_last(state)
        if is_best:
            self.trainer.ckpt.save_best(state)
        return last

    # -- main loop --------------------------------------------------------
    def run(self) -> float:
        t = self.trainer
        # Warm-start reference: the val accuracy the model already has, so the
        # saved best_model.pt is never worse than the supervised starting point.
        init = t.evaluate(self.val_loader)
        best_acc = float(init["val_accuracy"])
        self._save_best_and_last(-1, {**init}, best_acc, is_best=True)
        self.logger.info(
            f"Self-training start | warm val_acc={best_acc:.4f} | "
            f"tau={self.tau} rounds={self.rounds} epochs/round={self.epochs_per_round}"
        )

        global_epoch = 0
        for r in range(self.rounds):
            # 1) self-label the pool with the CURRENT model
            conf, pred, truth = infer_confidence(
                t.model, self.unlabeled_infer_loader, t.device, t.autocast_ctx
            )
            stats = pseudo_label_stats(conf, pred, truth, self.tau)
            self.logger.info(
                f"Round {r + 1}/{self.rounds} | self-label: "
                f"{stats['n_selected']}/{stats['n_total']} kept "
                f"(coverage={stats['coverage']:.3f}, tau={self.tau}) | "
                f"pseudo_acc={stats['pseudo_accuracy']:.4f} | "
                f"mean_conf={stats['mean_conf']:.3f}"
            )
            pseudo_subset = build_pseudo_subset(self.unlabeled_train_ds, conf, pred, self.tau)
            loader = self._combined_loader(pseudo_subset)

            # 2) train E epochs on labelled + pseudo-labelled
            for _ in range(self.epochs_per_round):
                lr = t.optimizer.param_groups[0]["lr"]
                t0 = time.time()
                tm = t.train_one_epoch(loader, global_epoch)
                vm = t.evaluate(self.val_loader)
                if t.scheduler is not None:
                    t.scheduler.step()

                val_acc = float(vm["val_accuracy"])
                is_best = val_acc > best_acc
                if is_best:
                    best_acc = val_acc

                metrics = {**tm, **vm, "round": r + 1,
                           "num_pseudo": stats["n_selected"],
                           "pseudo_accuracy": stats["pseudo_accuracy"]}
                last_path = self._save_best_and_last(global_epoch, metrics, best_acc, is_best)

                self.csv.log(
                    round=r + 1, epoch=global_epoch + 1,
                    num_pseudo=stats["n_selected"], coverage=stats["coverage"],
                    pseudo_accuracy=stats["pseudo_accuracy"], mean_conf=stats["mean_conf"],
                    train_loss=tm["train_loss"], train_ce=tm["train_ce"],
                    train_center_raw=tm["train_center_raw"], train_accuracy=tm["train_accuracy"],
                    val_loss=vm["val_loss"], val_accuracy=val_acc,
                    learning_rate=lr, checkpoint_path=str(last_path),
                )
                self.logger.info(
                    f"  R{r + 1} e{global_epoch + 1:3d} | lr={lr:.2e} | "
                    f"train_loss={tm['train_loss']:.4f} train_acc={tm['train_accuracy']:.4f} | "
                    f"val_acc={val_acc:.4f} | best={best_acc:.4f} | {time.time() - t0:.1f}s"
                    + ("  <-- new best" if is_best else "")
                )
                global_epoch += 1

        self.csv.close()
        self.logger.info(f"Self-training complete. Best val accuracy: {best_acc:.4f}")
        return best_acc
