"""Evaluate one arm of the room comparison on the held-out test set.

Adapted from `fer_wen/evaluate_wen.py`, with one addition that matters for the
analysis that follows. Paired tests such as McNemar require the two runs to be
aligned image by image. Aligning on row order assumes both loops enumerated the
directory identically, and aligning on the file path assumes the two runs saw
the data at the same location -- true today, but not something the analysis
should depend on. Each row therefore also carries a SHA-1 of the decoded pixels,
which identifies the image itself and is the same identifier the deduplication
work uses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rooms_wen_data import build_eval_loader           # noqa: E402
from rooms_wen_model import build_model_from_config    # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate one trained arm.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default=None,
                   help="override the eval directory stored in the checkpoint")
    return p.parse_args()


def plot_confusion(cm: np.ndarray, names: List[str], path: Path,
                   normalize: bool, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = cm.astype(float)
    if normalize:
        row = m.sum(axis=1, keepdims=True)
        m = np.divide(m, row, out=np.zeros_like(m), where=row > 0)
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=11)
    thresh = m.max() / 2 if m.max() else 0.5
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{m[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                    ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pixel_hash(path: str) -> str:
    """SHA-1 of the decoded RGB pixels -- the same identity the deduplication
    analysis uses, so predictions can be joined to it and to each other."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return hashlib.sha1(arr.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    class_names = list(cfg["data"]["class_names"])
    mode = ckpt.get("center_mode", "?")
    seed = ckpt.get("seed", "?")

    if args.data_dir:
        cfg = {**cfg, "data": {**cfg["data"], "eval_dir": args.data_dir}}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_config(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader, ds = build_eval_loader(cfg, pin=device.type == "cuda")

    run_dir = ckpt_path.parent.parent
    out_dir = run_dir / "outputs" / f"eval_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_probs, all_pred, all_true, all_emb = [], [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits, emb = model(images, return_embeddings=True)
            probs = torch.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_pred.append(probs.argmax(1).cpu().numpy())
            all_true.append(targets.numpy())
            all_emb.append(emb.float().cpu().numpy())

    probs = np.concatenate(all_probs)
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    emb = np.concatenate(all_emb)

    overall = float((pred == true).mean())
    cm = confusion_matrix(true, pred, labels=list(range(len(class_names))))
    report = classification_report(true, pred, labels=list(range(len(class_names))),
                                   target_names=class_names, digits=4,
                                   zero_division=0)

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Checkpoint:  {ckpt_path}\n")
        fh.write(f"Eval dir:    {cfg['data']['eval_dir']}\n")
        fh.write(f"Center mode: {mode}\n")
        fh.write(f"Seed:        {seed}\n")
        fh.write(f"Images:      {len(true)}\n\n")
        fh.write(f"Overall accuracy: {overall:.4f}\n\n")
        fh.write("Per-class accuracy:\n")
        for i, name in enumerate(class_names):
            mask = true == i
            acc = float((pred[mask] == i).mean()) if mask.any() else float("nan")
            fh.write(f"  {name:15s}: {acc:.4f}\n")

    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png", False,
                   f"Confusion matrix (counts) -- {mode}, seed {seed}")
    plot_confusion(cm, class_names, out_dir / "confusion_matrix_norm.png", True,
                   f"Confusion matrix (row-normalised) -- {mode}, seed {seed}")

    paths = ds.paths()
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sha1", "path", "true", "pred", "correct"] +
                   [f"p_{n}" for n in class_names])
        for path, tl, pl, pr in zip(paths, true, pred, probs):
            w.writerow([pixel_hash(path), path, class_names[tl],
                        class_names[pl], int(tl == pl)] +
                       [f"{v:.6f}" for v in pr])

    np.save(out_dir / "embeddings.npy", emb)
    np.save(out_dir / "labels.npy", true)

    # --- centre alignment, when the arm has centres at all ------------------
    state = ckpt.get("center_loss_state_dict") or {}
    centers = state.get("centers")
    if centers is not None:
        C = centers.float().numpy()
        rows: List[Dict[str, object]] = []
        for k, name in enumerate(class_names):
            mask = true == k
            if not mask.any():
                continue
            mu = emb[mask].mean(0)
            # A zero-length centre or centroid has no direction, so the cosine
            # is undefined rather than zero. Dividing anyway would have written
            # nan or raised, depending on the platform.
            denom = float(np.linalg.norm(C[k]) * np.linalg.norm(mu))
            cos = float(C[k] @ mu) / denom if denom > 0 else float("nan")
            rows.append({"class": name, "test_images": int(mask.sum()),
                         "cosine_centre_vs_centroid":
                             round(cos, 6) if math.isfinite(cos) else ""})
        with open(out_dir / "centre_alignment.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        ref = 1.0 / np.sqrt(C.shape[1])
        print(f"\nCentre alignment ({mode}), random reference {ref:.3f}:")
        for r in rows:
            v = r["cosine_centre_vs_centroid"]
            shown = f"{v:.3f}" if isinstance(v, float) else "undefined"
            print(f"   {r['class']:<15}{shown}")

    print(f"\n{mode} | seed {seed} | test accuracy {overall:.4f}")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
