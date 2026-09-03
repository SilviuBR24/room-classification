"""
evaluate_wen.py
===============
Evaluates a trained checkpoint on the held-out test set and writes the same
artefacts as every other experiment in this project, so the results can be read
side by side.

    python evaluate_wen.py --checkpoint <run>/checkpoints/best_model.pt \
                           --data-dir <test_dir>

Written into <run>/outputs/eval_<timestamp>/:

    metrics.txt                 overall and per-class accuracy
    classification_report.txt   precision / recall / F1, macro and weighted
    confusion_matrix.csv        counts
    confusion_matrix.png        counts
    confusion_matrix_norm.png   row-normalised
    predictions.csv             per image, with class probabilities
    embeddings.npy, labels.npy  for the geometric analysis
    centre_alignment.csv        cosine between each learned centre and the
                                empirical class centroid, with the class's
                                training count

The last file is specific to this experiment. The two variants differ only in
how the centres are updated, so how well each centre tracks its class is the
measurement that shows the difference directly, rather than through its effect
on accuracy.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from fer_wen_data import build_dataset  # noqa: E402
from fer_wen_model import build_model_from_config  # noqa: E402

HERE = Path(__file__).resolve().parent

# Training-set counts, used only to annotate the centre-alignment report.
FER_TRAIN_COUNTS = {"angry": 3995, "disgust": 436, "fear": 4097, "happy": 7215,
                    "sad": 4830, "surprise": 3171, "neutral": 4965}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on the test set.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default=None,
                   help="test directory; defaults to data.eval_dir in the checkpoint's config")
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def plot_confusion(cm: np.ndarray, names: List[str], path: Path,
                   normalize: bool, title: str) -> None:
    data = cm.astype(float)
    if normalize:
        rows = data.sum(axis=1, keepdims=True)
        data = np.divide(data, rows, out=np.zeros_like(data), where=rows > 0)
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2.5, 1.0 * len(names) + 2.2))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() else 1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    thresh = (data.max() or 1) / 2.0
    for i in range(len(names)):
        for j in range(len(names)):
            txt = f"{data[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    class_names = cfg["data"]["class_names"]
    mode = ckpt.get("center_mode", "?")

    data_dir = args.data_dir or cfg["data"]["eval_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model_from_config(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = build_dataset(data_dir, cfg, train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)

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
                                   target_names=class_names, digits=4, zero_division=0)

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Checkpoint: {ckpt_path}\n")
        fh.write(f"Eval dir:   {data_dir}\n")
        fh.write(f"Center mode: {mode}\n")
        fh.write(f"Images:     {len(true)}\n\n")
        fh.write(f"Overall accuracy: {overall:.4f}\n\n")
        fh.write("Per-class accuracy:\n")
        for i, name in enumerate(class_names):
            mask = true == i
            acc = float((pred[mask] == i).mean()) if mask.any() else float("nan")
            fh.write(f"  {name:15s}: {acc:.4f}\n")

    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png",
                   normalize=False, title=f"Confusion Matrix (counts) -- {mode}")
    plot_confusion(cm, class_names, out_dir / "confusion_matrix_norm.png",
                   normalize=True, title=f"Confusion Matrix (row-normalized) -- {mode}")

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "true", "pred"] + [f"p_{n}" for n in class_names])
        for (path, _), tl, pl, pr in zip(ds.samples, true, pred, probs):
            w.writerow([path, class_names[tl], class_names[pl]] +
                       [f"{v:.6f}" for v in pr])

    np.save(out_dir / "embeddings.npy", emb)
    np.save(out_dir / "labels.npy", true)

    # --- centre alignment, the measurement specific to this experiment ----
    centers = ckpt.get("center_loss_state_dict", {}).get("centers")
    if centers is not None:
        C = centers.float().numpy()
        rows: List[Dict[str, object]] = []
        for k, name in enumerate(class_names):
            mask = true == k
            if not mask.any():
                continue
            mu = emb[mask].mean(0)
            cos = float(C[k] @ mu / (np.linalg.norm(C[k]) * np.linalg.norm(mu)))
            rows.append({"class": name, "train_images": FER_TRAIN_COUNTS.get(name, ""),
                         "test_images": int(mask.sum()),
                         "cosine_centre_vs_centroid": round(cos, 6)})
        with open(out_dir / "centre_alignment.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        ref = 1.0 / np.sqrt(C.shape[1])
        print(f"\nCentre alignment ({mode}), random reference {ref:.3f}:")
        for r in rows:
            print(f"   {r['class']:<10}{str(r['train_images']):>7} train  "
                  f"cos={r['cosine_centre_vs_centroid']:+.3f}")

    print(f"\nOverall accuracy: {overall:.4f}")
    print(report)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
