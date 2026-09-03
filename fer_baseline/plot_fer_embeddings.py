"""
plot_fer_embeddings.py
======================
Draws the FER2013 embedding comparison.

Why this exists instead of using analyze_embeddings.py directly
---------------------------------------------------------------
The shared analysis script carries the six room-class names as a module
constant and iterates over exactly that many classes when plotting. Run on
FER2013 it therefore mislabels every class and silently omits the seventh
(`neutral`, 626 test images), because the loop never reaches index 6. The
metrics it prints are unaffected -- they are computed from the labels
themselves -- but the figure is wrong.

Rather than edit the shared script, which the room experiments depend on, this
file reuses its metric function and does its own plotting with the correct
class names. The numbers therefore match the shared analysis exactly.

    python plot_fer_embeddings.py --runs <ce_run> <cl_run>
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

import shared_infrastructure  # noqa: F401,E402  -- sets sys.path
from analyze_embeddings import compactness  # noqa: E402  -- same metrics, unmodified

HERE = Path(__file__).resolve().parent

# FER2013 classes, in the order fixed by config_fer.yaml.
FER_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the FER2013 embedding comparison.")
    p.add_argument("--runs", nargs="+", required=True, help="run directories")
    p.add_argument("--labels", nargs="+", default=None, help="panel titles")
    p.add_argument("--out-dir", default=str(HERE.parent / "figuri" / "fer_geometry"))
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load(run_dir: str):
    hits = sorted(glob.glob(os.path.join(run_dir, "outputs", "eval_*", "embeddings.npy")))
    if not hits:
        raise SystemExit(f"No embeddings under {run_dir}")
    emb = hits[-1]
    X = np.load(emb)
    y = np.load(emb.replace("embeddings.npy", "labels.npy"))
    return X, y, emb


def main() -> None:
    args = parse_args()
    labels = args.labels or [os.path.basename(r) for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match --runs in length")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, metrics = [], []
    for run, lab in zip(args.runs, labels):
        X, y, src = load(run)
        present = sorted(int(c) for c in np.unique(y))
        print(f"[ok] {lab:34s} {X.shape[0]} x {X.shape[1]}  classes present: {present}")
        if len(present) != len(FER_CLASSES):
            print(f"     WARNING: expected {len(FER_CLASSES)} classes, found {len(present)}")
        data.append((lab, X, y))
        metrics.append(compactness(X, y))

    print()
    print("=" * 92)
    print(f"{'model':<34}{'intra':>9}{'inter':>9}{'inter/intra':>13}"
          f"{'silhouette':>12}{'davies-b.':>11}")
    print("-" * 92)
    for (lab, _, _), m in zip(data, metrics):
        print(f"{lab:<34}{m['intra_mean']:>9.3f}{m['inter_mean']:>9.3f}"
              f"{m['inter_intra']:>13.4f}{m['silhouette']:>12.4f}"
              f"{m['davies_bouldin']:>11.4f}")
    print("=" * 92)

    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.4), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, (lab, X, y), m in zip(axes[0], data, metrics):
        Z = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                 random_state=args.seed, max_iter=1000).fit_transform(X)
        # Iterate over the classes that are actually present, not over a
        # hard-coded count, so nothing is silently left out.
        for c in sorted(int(v) for v in np.unique(y)):
            mask = y == c
            ax.scatter(Z[mask, 0], Z[mask, 1], s=9, color=cmap(c % 10),
                       label=FER_CLASSES[c], alpha=0.7, linewidths=0)
        ax.set_title(f"{lab}\nsilhouette={m['silhouette']:.3f}  "
                     f"inter/intra={m['inter_intra']:.3f}", fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])

    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                       fontsize=9, frameon=False, markerscale=1.8)
    fig.tight_layout(rect=[0, 0, 0.92, 1.0])

    csv_path = out_dir / "embedding_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("label,n_samples,dim,intra_mean,inter_mean,inter_intra,"
                 "silhouette,davies_bouldin\n")
        for (lab, X, _), m in zip(data, metrics):
            fh.write(f"{lab},{X.shape[0]},{X.shape[1]},{m['intra_mean']},"
                     f"{m['inter_mean']},{m['inter_intra']},{m['silhouette']},"
                     f"{m['davies_bouldin']}\n")

    fig_path = out_dir / "tsne_fer.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nMetrics -> {csv_path}")
    print(f"Figure  -> {fig_path}")


if __name__ == "__main__":
    main()
