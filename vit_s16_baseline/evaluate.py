"""
evaluate.py
===========
Evaluate a saved checkpoint on an evaluation/test set.

Usage
-----
    python evaluate.py --checkpoint runs/<run>/checkpoints/best_model.pt

Optional:
    --data-dir DIR        override the eval directory from the checkpoint config
    --output-dir DIR      where to write results (default: alongside checkpoint)
    --save-embeddings     also dump CLS embeddings (.npy) for later t-SNE plots

Outputs (written to the output dir)
-----------------------------------
    confusion_matrix.png        raw-count confusion matrix
    confusion_matrix_norm.png   row-normalized confusion matrix
    confusion_matrix.csv        confusion matrix as CSV (true x pred)
    classification_report.txt   per-class precision/recall/F1 (sklearn)
    predictions.csv             per-image: path, true, pred, correct, probs
    metrics.txt                 overall + per-class accuracy
    embeddings.npy / labels.npy (only with --save-embeddings)

The model architecture and class mapping are read FROM THE CHECKPOINT, so the
exact same network is rebuilt regardless of the current config.yaml.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Headless-safe matplotlib (no display needed on Colab / servers).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
)

from src.checkpoint import load_checkpoint  # noqa: E402
from src.dataset import build_dataset  # noqa: E402
from src.utils import get_amp_components, get_device  # noqa: E402
from src.vit import build_vit_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ViT checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a checkpoint (e.g. best_model.pt).")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override eval directory from checkpoint config.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for evaluation outputs.")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="Also save CLS embeddings + labels as .npy.")
    return parser.parse_args()


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], path: Path, normalize: bool, title: str
) -> None:
    """Render a confusion matrix to a PNG (no seaborn dependency)."""
    data = cm.astype(np.float64)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(1.4 * n + 2, 1.4 * n + 1))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() > 0 else 1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = data.max() / 2.0 if data.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            txt = f"{data[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if data[i, j] > thresh else "black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_ctx: Any,
    collect_embeddings: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, probs, embeddings) in dataset order."""
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    all_probs: List[np.ndarray] = []
    all_emb: List[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with autocast_ctx():
            if collect_embeddings:
                logits, emb = model(images, return_embeddings=True)
            else:
                logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)

        all_true.extend(targets.numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())
        if collect_embeddings:
            all_emb.append(emb.float().cpu().numpy())

    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0))
    emb_arr = np.concatenate(all_emb, axis=0) if all_emb else np.empty((0, 0))
    return np.array(all_true), np.array(all_pred), probs_arr, emb_arr


def main() -> None:
    args = parse_args()
    device = get_device()

    # --- load checkpoint + rebuild model ------------------------------
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config: Dict[str, Any] = checkpoint["config"]
    class_to_idx: Dict[str, int] = checkpoint["class_to_idx"]
    # idx -> name, ordered by index.
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

    model = build_vit_from_config(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, autocast_ctx = get_amp_components(device, config["training"].get("use_amp", False))

    # --- output directory ---------------------------------------------
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path(args.checkpoint).resolve().parent.parent / "outputs" / f"eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data ----------------------------------------------------------
    eval_dir = args.data_dir or config["data"]["eval_dir"]
    eval_ds = build_dataset(eval_dir, config, train=False)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    print(f"[eval] {len(eval_ds)} images from {eval_dir}")

    # --- inference -----------------------------------------------------
    y_true, y_pred, probs, embeddings = run_inference(
        model, eval_loader, device, autocast_ctx, collect_embeddings=args.save_embeddings
    )

    # --- metrics -------------------------------------------------------
    overall_acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, digits=4, zero_division=0,
    )

    # per-class accuracy (recall along the diagonal)
    per_class_acc = {}
    for idx, name in enumerate(class_names):
        total = cm[idx].sum()
        per_class_acc[name] = float(cm[idx, idx] / total) if total else 0.0

    # --- write outputs -------------------------------------------------
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png",
                          normalize=False, title="Confusion Matrix (counts)")
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix_norm.png",
                          normalize=True, title="Confusion Matrix (row-normalized)")

    # confusion matrix CSV
    import csv as _csv
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh)
        writer.writerow(["true\\pred"] + class_names)
        for i, name in enumerate(class_names):
            writer.writerow([name] + cm[i].tolist())

    # classification report + metrics
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Checkpoint: {args.checkpoint}\n")
        fh.write(f"Eval dir:   {eval_dir}\n")
        fh.write(f"Images:     {len(y_true)}\n\n")
        fh.write(f"Overall accuracy: {overall_acc:.4f}\n\n")
        fh.write("Per-class accuracy:\n")
        for name, acc in per_class_acc.items():
            fh.write(f"  {name:15s}: {acc:.4f}\n")

    # predictions.csv (path, true, pred, correct, per-class probs)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh)
        header = ["filepath", "true_label", "pred_label", "correct"] + [f"prob_{c}" for c in class_names]
        writer.writerow(header)
        for i, (path, _) in enumerate(eval_ds.samples):
            true_name = class_names[int(y_true[i])]
            pred_name = class_names[int(y_pred[i])]
            row = [path, true_name, pred_name, int(y_true[i] == y_pred[i])]
            row += [f"{p:.6f}" for p in probs[i].tolist()]
            writer.writerow(row)

    # optional embeddings for later embedding-space plots
    if args.save_embeddings:
        np.save(out_dir / "embeddings.npy", embeddings)
        np.save(out_dir / "labels.npy", y_true)

    # --- console summary ----------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall accuracy: {overall_acc:.4f}")
    print("Per-class accuracy:")
    for name, acc in per_class_acc.items():
        print(f"  {name:15s}: {acc:.4f}")
    print(f"\nAll outputs saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
