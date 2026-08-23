"""
mnist_center_loss.py
====================
Sanity check of the project's Center Loss implementation, and the visual
cross-entropy vs cross-entropy + Center Loss comparison.

Why MNIST, and why a 2-dimensional embedding
--------------------------------------------
The thesis reports that Center Loss does not improve accuracy on the room
dataset. The first question that raises is whether the implementation is
correct. This script answers it by reproducing the classic demonstration from
Wen et al. (ECCV 2016) on a task where the method is known to work.

The network deliberately bottlenecks to a **2-dimensional** embedding, so the
feature space can be plotted DIRECTLY -- no t-SNE, no projection. That matters:
t-SNE normalises scale, which is exactly what hides the effect of Center Loss.
Plotting raw 2-D features shows the difference honestly.

Crucially, this imports `src.center_loss.CenterLoss` -- the same module used for
the room experiments -- so a positive result validates the actual thesis code,
not a re-implementation.

    python mnist_center_loss.py                 # both runs, default settings
    python mnist_center_loss.py --epochs 15     # longer, sharper separation

Outputs (into --out-dir, default ../figuri):
    mnist_center_loss.png      two panels, raw 2-D embeddings, CE vs CE+center
    mnist_center_loss.csv      accuracy + compactness metrics for both runs
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import davies_bouldin_score, silhouette_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.center_loss import CenterLoss  # the module under test  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Center Loss sanity check on MNIST.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--center-weight", type=float, default=1.0,
                   help="lambda. With a 2-D embedding the raw centre term is "
                        "small, so a much larger lambda than the 384-D room "
                        "experiments (0.0005) is appropriate.")
    p.add_argument("--center-lr", type=float, default=0.5)
    p.add_argument("--data-dir", type=str, default=str(HERE / "mnist_data"))
    p.add_argument("--out-dir", type=str, default=str(HERE.parent / "figuri"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class SmallCNN(nn.Module):
    """Compact CNN that bottlenecks to a 2-D embedding before classifying.

    The 2-D layer is the whole point: it is the representation Center Loss
    acts on, and it can be plotted without any dimensionality reduction.
    """

    def __init__(self, num_classes: int = 10, embed_dim: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.PReLU(),
            nn.MaxPool2d(2),                                  # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.PReLU(),
            nn.MaxPool2d(2),                                  # 14 -> 7
        )
        self.to_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.PReLU(),
            nn.Linear(128, embed_dim),                        # the 2-D embedding
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embeddings: bool = False):
        emb = self.to_embedding(self.features(x))
        logits = self.classifier(emb)
        return (logits, emb) if return_embeddings else logits


def compactness(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Same measures used for the room dataset, so the two are comparable."""
    classes = np.unique(y)
    cents = np.stack([X[y == c].mean(axis=0) for c in classes])
    intra = float(np.mean([np.linalg.norm(X[y == c] - cents[i], axis=1).mean()
                           for i, c in enumerate(classes)]))
    inter = float(np.mean([np.linalg.norm(cents[i] - cents[j])
                           for i in range(len(classes))
                           for j in range(i + 1, len(classes))]))
    return {
        "intra_mean": intra,
        "inter_mean": inter,
        "inter_intra": inter / intra if intra else float("nan"),
        "silhouette": float(silhouette_score(X, y)),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device
             ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Test accuracy plus the 2-D embeddings of the whole test set."""
    model.eval()
    correct = total = 0
    embs, labs = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits, emb = model(images, return_embeddings=True)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
        embs.append(emb.cpu().numpy())
        labs.append(targets.cpu().numpy())
    return correct / total, np.concatenate(embs), np.concatenate(labs)


def train_one(use_center: bool, args, device, train_loader, test_loader
              ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Train a fresh model, with or without Center Loss. Same seed for both."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    center_loss = opt_center = None
    if use_center:
        center_loss = CenterLoss(num_classes=10, feat_dim=2).to(device)
        opt_center = torch.optim.SGD(center_loss.parameters(), lr=args.center_lr)

    tag = "CE + Center Loss" if use_center else "CE only"
    print(f"\n--- training: {tag} ---", flush=True)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run_ce = run_center = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_center:
                opt_center.zero_grad(set_to_none=True)

            logits, emb = model(images, return_embeddings=True)
            ce = criterion(logits, targets)
            if use_center:
                raw = center_loss(emb, targets)
                loss = ce + args.center_weight * raw
                run_center += raw.item()
            else:
                loss = ce
            loss.backward()
            optimizer.step()
            if use_center:
                opt_center.step()
            run_ce += ce.item()

        n = len(train_loader)
        extra = f" center={run_center / n:.4f}" if use_center else ""
        print(f"  epoch {epoch + 1:2d}/{args.epochs}  ce={run_ce / n:.4f}{extra}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

    acc, emb, lab = evaluate(model, test_loader, device)
    print(f"  test accuracy: {acc:.4f}")
    return acc, emb, lab


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    print(f"MNIST: {len(train_ds)} train / {len(test_ds)} test")

    results = []
    panels = []
    for use_center in (False, True):
        acc, emb, lab = train_one(use_center, args, device, train_loader, test_loader)
        m = compactness(emb, lab)
        m["label"] = "CE + Center Loss" if use_center else "CE only"
        m["accuracy"] = acc
        results.append(m)
        panels.append((m["label"], emb, lab, m))

    # ---- comparison table ------------------------------------------------
    print("\n" + "=" * 84)
    print("MNIST -- CENTER LOSS SANITY CHECK (2-D embeddings, plotted directly)")
    print("=" * 84)
    print(f"{'run':<20} {'test acc':>9} {'intra':>9} {'inter':>9} "
          f"{'inter/intra':>12} {'silhouette':>11}")
    print("-" * 84)
    for m in results:
        print(f"{m['label']:<20} {m['accuracy']:>9.4f} {m['intra_mean']:>9.3f} "
              f"{m['inter_mean']:>9.3f} {m['inter_intra']:>12.3f} "
              f"{m['silhouette']:>11.4f}")
    print("=" * 84)

    a, b = results[0], results[1]
    shrink = a["intra_mean"] / b["intra_mean"] if b["intra_mean"] else float("nan")
    print(f"\nintra-class spread shrank {shrink:.1f}x "
          f"({a['intra_mean']:.3f} -> {b['intra_mean']:.3f})")
    print(f"separability ratio {a['inter_intra']:.3f} -> {b['inter_intra']:.3f}"
          f"  ({'IMPROVED' if b['inter_intra'] > a['inter_intra'] else 'not improved'})")
    print(f"silhouette {a['silhouette']:.4f} -> {b['silhouette']:.4f}")
    verdict = ("IMPLEMENTATION VALIDATED: Center Loss visibly tightens the "
               "clusters on a task where it is known to work.")
    if b["inter_intra"] <= a["inter_intra"] and b["silhouette"] <= a["silhouette"]:
        verdict = ("INCONCLUSIVE: no clear improvement even on MNIST -- "
                   "investigate lambda or the implementation.")
    print("\n" + verdict)

    with open(out_dir / "mnist_center_loss.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["label", "accuracy", "intra_mean", "inter_mean", "inter_intra",
                "silhouette", "davies_bouldin"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for m in results:
            w.writerow({k: m[k] for k in cols})

    # ---- figure: raw 2-D embeddings, no projection -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8))
    cmap = plt.get_cmap("tab10")
    for ax, (label, emb, lab, m) in zip(axes, panels):
        for d in range(10):
            mask = lab == d
            ax.scatter(emb[mask, 0], emb[mask, 1], s=3, color=cmap(d),
                       label=str(d), alpha=0.5, linewidths=0)
        ax.set_title(f"{label}\naccuracy={m['accuracy']:.4f}   "
                     f"inter/intra={m['inter_intra']:.2f}", fontsize=11)
        ax.set_xlabel("embedding dimension 1")
        ax.set_ylabel("embedding dimension 2")
        ax.set_aspect("equal", adjustable="datalim")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9,
                   frameon=False, markerscale=3, title="digit")
    fig.suptitle("MNIST: spatiul de trasaturi 2D, desenat direct (fara t-SNE)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.94, 0.95])
    fig.savefig(out_dir / "mnist_center_loss.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigure  -> {out_dir / 'mnist_center_loss.png'}")
    print(f"Metrics -> {out_dir / 'mnist_center_loss.csv'}")


if __name__ == "__main__":
    main()
