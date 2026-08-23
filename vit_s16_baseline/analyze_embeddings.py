"""
analyze_embeddings.py
=====================
Embedding-space analysis for Objective 4 of the assignment: check whether the
classes really do become MORE COMPACT when Center Loss is added.

For every run directory given, it loads the CLS embeddings saved by
`evaluate.py --save-embeddings` and produces:

    * a metrics table (printed, and written as embedding_metrics.csv)
    * a t-SNE figure with one panel per model (tsne_comparison.png)

Metrics, and why these ones
---------------------------
"More compact" is ambiguous, so both readings are reported:

  intra_mean   mean distance from a sample to its own class centroid.
               This is literally what Center Loss minimises, so it answers
               "did the optimisation do what it was told?" -- but it is
               SCALE-DEPENDENT: shrinking the whole space shrinks it too.

  inter_mean   mean distance between class centroids.

  inter_intra  inter_mean / intra_mean. SCALE-INVARIANT, so this is the one
               that answers the question that actually matters: are classes
               better separated relative to their own spread?

  silhouette   [-1, 1], higher is better; combines both effects per sample.
  davies_bouldin  lower is better.

A method can therefore compact clusters (intra_mean drops) while making
separability no better at all (inter_intra flat or worse), which is exactly
the distinction the dissertation needs to make.

Usage
-----
    # explicit run folders
    python analyze_embeddings.py --runs <run_dir_1> <run_dir_2> ...

    # or discover them under a root (each must contain outputs/eval_*/)
    python analyze_embeddings.py --runs-root /path/to/dissertation_runs

    # nicer panel titles
    python analyze_embeddings.py --runs A B --labels "baseline" "center 0.0005"
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score

CLASS_NAMES = ["bathroom", "bedroom", "dining_room",
               "entrance_hall", "kitchen", "living_room"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embedding-space compactness analysis.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", nargs="+", help="Run folders to compare, in order.")
    src.add_argument("--runs-root", help="Discover every run folder under this root.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Panel titles (default: derived from folder names).")
    p.add_argument("--out-dir", default=".", help="Where to write figure + CSV.")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42,
                   help="Same seed for every panel, so layouts are comparable.")
    return p.parse_args()


def load_run(run_dir: str) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """Load (embeddings, labels, source_path) from the newest eval_* folder.

    Labels come from labels.npy when present. If only predictions.csv survives,
    the true labels are recovered from its `true_label` column -- evaluate.py
    writes both files from the same non-shuffled pass, so row i of the CSV and
    row i of embeddings.npy are the same image.
    """
    hits = sorted(glob.glob(os.path.join(run_dir, "outputs", "eval_*", "embeddings.npy")))
    if not hits:
        return None
    emb_path = hits[-1]
    X = np.load(emb_path)

    lab_path = emb_path.replace("embeddings.npy", "labels.npy")
    if os.path.isfile(lab_path):
        return X, np.load(lab_path), emb_path

    pred_path = emb_path.replace("embeddings.npy", "predictions.csv")
    if os.path.isfile(pred_path):
        idx = {name: i for i, name in enumerate(CLASS_NAMES)}
        with open(pred_path, newline="", encoding="utf-8") as fh:
            y = np.array([idx[r["true_label"]] for r in csv.DictReader(fh)])
        if len(y) != X.shape[0]:
            print(f"[warn] {run_dir}: {len(y)} rows in predictions.csv but "
                  f"{X.shape[0]} embeddings -- skipping")
            return None
        return X, y, emb_path

    return None


def pretty_label(run_dir: str) -> str:
    """Turn a timestamped run folder name into something readable."""
    name = os.path.basename(os.path.normpath(run_dir))
    # strip the leading YYYY-MM-DD_HH-MM_ prefix if present
    parts = name.split("_", 2)
    return parts[2] if len(parts) == 3 and parts[0][:2] == "20" else name


def compactness(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Intra/inter distances plus two standard cluster-quality indices."""
    classes = np.unique(y)
    centroids = np.stack([X[y == c].mean(axis=0) for c in classes])

    intra = float(np.mean([
        np.linalg.norm(X[y == c] - centroids[i], axis=1).mean()
        for i, c in enumerate(classes)
    ]))
    inter = float(np.mean([
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(len(classes)) for j in range(i + 1, len(classes))
    ]))

    return {
        "n_samples": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "intra_mean": intra,
        "inter_mean": inter,
        "inter_intra": inter / intra if intra else float("nan"),
        "silhouette": float(silhouette_score(X, y)),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
    }


def main() -> None:
    args = parse_args()

    run_dirs: List[str] = args.runs or sorted(
        d for d in glob.glob(os.path.join(args.runs_root, "*"))
        if os.path.isdir(d) and load_run(d)
    )
    if not run_dirs:
        sys.exit("No run folders with saved embeddings found. Re-run evaluate.py "
                 "with --save-embeddings.")

    labels = args.labels or [pretty_label(d) for d in run_dirs]
    if len(labels) != len(run_dirs):
        sys.exit(f"--labels has {len(labels)} entries but {len(run_dirs)} runs were given.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load + measure -------------------------------------------------
    data, metrics = [], []
    for run_dir, label in zip(run_dirs, labels):
        found = load_run(run_dir)
        if not found:
            print(f"[skip] no usable embeddings in {run_dir}")
            continue
        X, y, emb_path = found
        m = compactness(X, y)
        m["label"] = label
        m["run_dir"] = run_dir
        data.append((label, X, y))
        metrics.append(m)
        print(f"[ok] {label:28s} {X.shape[0]} x {X.shape[1]}  from {emb_path}")

    if not metrics:
        sys.exit("Nothing to analyse.")

    # ---- table ----------------------------------------------------------
    print("\n" + "=" * 92)
    print("EMBEDDING-SPACE COMPACTNESS")
    print("=" * 92)
    print(f"{'model':<28} {'intra':>9} {'inter':>9} {'inter/intra':>12} "
          f"{'silhouette':>11} {'davies-b.':>10}")
    print(f"{'':28} {'(lower =':>9} {'':>9} {'(higher =':>12} "
          f"{'(higher =':>11} {'(lower =':>10}")
    print(f"{'':28} {'tighter)':>9} {'':>9} {'better)':>12} "
          f"{'better)':>11} {'better)':>10}")
    print("-" * 92)
    for m in metrics:
        print(f"{m['label']:<28} {m['intra_mean']:>9.3f} {m['inter_mean']:>9.3f} "
              f"{m['inter_intra']:>12.4f} {m['silhouette']:>11.4f} "
              f"{m['davies_bouldin']:>10.4f}")
    print("=" * 92)

    if len(metrics) >= 2:
        a, b = metrics[0], metrics[1]
        tighter = b["intra_mean"] < a["intra_mean"]
        better = b["inter_intra"] > a["inter_intra"]
        print(f"\n'{b['label']}' vs '{a['label']}':")
        print(f"  clusters are {'TIGHTER' if tighter else 'NOT tighter'} in absolute "
              f"terms (intra {a['intra_mean']:.3f} -> {b['intra_mean']:.3f})")
        print(f"  separability is {'BETTER' if better else 'NOT better'} "
              f"(inter/intra {a['inter_intra']:.4f} -> {b['inter_intra']:.4f})")
        if tighter and not better:
            print("  => compaction happened, but classes shrank together: the space "
                  "was rescaled, not better separated.")

    csv_path = out_dir / "embedding_metrics.csv"
    fields = ["label", "n_samples", "dim", "intra_mean", "inter_mean",
              "inter_intra", "silhouette", "davies_bouldin", "run_dir"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for m in metrics:
            w.writerow({k: m[k] for k in fields})
    print(f"\nMetrics -> {csv_path}")

    # ---- t-SNE figure ---------------------------------------------------
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.4), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for ax, (label, X, y), m in zip(axes[0], data, metrics):
        # Same seed/perplexity everywhere so panels are visually comparable.
        Z = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                 random_state=args.seed, max_iter=1000).fit_transform(X)
        for c in range(len(CLASS_NAMES)):
            mask = y == c
            if mask.any():
                ax.scatter(Z[mask, 0], Z[mask, 1], s=13, color=cmap(c),
                           label=CLASS_NAMES[c], alpha=0.75, linewidths=0)
        ax.set_title(f"{label}\nsilhouette={m['silhouette']:.3f}  "
                     f"inter/intra={m['inter_intra']:.3f}", fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])

    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                       fontsize=9, frameon=False)
    fig.suptitle("Spatiul de embedding (set de test), proiectie t-SNE", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])

    fig_path = out_dir / "tsne_comparison.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure  -> {fig_path}")


if __name__ == "__main__":
    main()
