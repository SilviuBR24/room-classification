"""Figures and tables comparing every FER2013 run along two axes.

The FER experiment ended up with eight runs that differ along two independent
axes, and the point of this script is to let those axes be read separately:

  * the dataset      -- the original split, which contains duplicate images
                        shared across partitions, versus the deduplicated split
  * the centre update -- the gradient formulation used throughout this thesis
                        versus Algorithm 1 as published by Wen et al.

Nothing here retrains or re-evaluates. Every run already saved the embeddings
of its evaluation set, so the geometry is recomputed from those saved arrays.
Accuracy is deliberately recomputed from each run's own predictions.csv and
compared against the number the comparison script filed in its summary CSV: if
those two disagree, a summary row is stale and the figures would be wrong.

Usage:
    python make_fer_figures.py            # verify, then write figures
    python make_fer_figures.py --check    # verify only, write nothing
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

RUNS_DIR = r"G:\My Drive\Dissertation_Thesis\dissertation_runs"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "figuri", "fer_comparison")

# The label order the training log reports; index i is class i.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

ORIGINAL = "original split (with duplicates)"
CLEAN = "deduplicated split"


@dataclass
class Run:
    key: str
    run_glob: str
    summary_csv: str
    summary_key: str
    dataset: str           # ORIGINAL or CLEAN
    centres: str           # "none", "gradient" or "algorithm1"
    label: str             # short label for figures
    run_dir: Optional[str] = None
    eval_dir: Optional[str] = None


RUNS: List[Run] = [
    # -- axis 1: the dataset, with the centre update held at "gradient" --------
    Run("orig_ce", "*_fer_crossentropy", "fer_comparison_results.csv",
        "fer_crossentropy", ORIGINAL, "none", "CE"),
    Run("orig_cl", "*_fer_crossentropy_centerloss", "fer_comparison_results.csv",
        "fer_crossentropy_centerloss", ORIGINAL, "gradient", "CE + Center Loss"),
    Run("clean_ce", "*_fer_clean_crossentropy", "fer_clean_comparison_results.csv",
        "fer_clean_crossentropy", CLEAN, "none", "CE"),
    Run("clean_cl", "*_fer_clean_crossentropy_centerloss",
        "fer_clean_comparison_results.csv",
        "fer_clean_crossentropy_centerloss", CLEAN, "gradient", "CE + Center Loss"),
    # -- axis 2: the centre update, with Center Loss always on ----------------
    Run("orig_grad", "*_fer_wen_gradient", "fer_wen_comparison_results.csv",
        "fer_wen_gradient", ORIGINAL, "gradient", "gradient update"),
    Run("orig_alg1", "*_fer_wen_algorithm1", "fer_wen_comparison_results.csv",
        "fer_wen_algorithm1", ORIGINAL, "algorithm1", "Algorithm 1"),
    Run("clean_grad", "*_fer_wen_clean_gradient", "fer_wen_comparison_results.csv",
        "fer_wen_clean_gradient", CLEAN, "gradient", "gradient update"),
    Run("clean_alg1", "*_fer_wen_clean_algorithm1", "fer_wen_comparison_results.csv",
        "fer_wen_clean_algorithm1", CLEAN, "algorithm1", "Algorithm 1"),
]


# --------------------------------------------------------------------------
# locating runs
# --------------------------------------------------------------------------
def newest_eval(run_dir: str) -> Optional[str]:
    hits = sorted(glob.glob(os.path.join(run_dir, "outputs", "eval_*")))
    return hits[-1] if hits else None


def locate(run: Run) -> None:
    """Resolve a run's directory, refusing anything ambiguous.

    The globs are anchored so that `*_fer_wen_gradient` cannot also match
    `*_fer_wen_clean_gradient`; a run name is the whole tail of the directory
    name, and glob has no way to express that on its own.
    """
    hits = [d for d in glob.glob(os.path.join(RUNS_DIR, run.run_glob))
            if os.path.isdir(d)
            and os.path.basename(d).endswith("_" + run.summary_key)]
    if len(hits) != 1:
        raise SystemExit(
            f"{run.key}: expected exactly one directory for "
            f"{run.summary_key!r}, found {len(hits)}: {hits}")
    run.run_dir = hits[0]
    run.eval_dir = newest_eval(hits[0])
    if run.eval_dir is None:
        raise SystemExit(f"{run.key}: no evaluation directory under {hits[0]}")


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def summary_row(csv_name: str, run_name: str) -> Dict[str, str]:
    path = os.path.join(RUNS_DIR, csv_name)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("run_name") == run_name:
                return row
    raise SystemExit(f"{run_name!r} not present in {csv_name}")


def accuracy_from_predictions(eval_dir: str) -> tuple[float, int]:
    """Recompute accuracy from the per-image predictions.

    This is the independent check on the summary CSV. The column names differ
    slightly between the two training loops, so both spellings are accepted,
    but a file that matches neither is an error rather than a silent zero.
    """
    path = os.path.join(eval_dir, "predictions.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")

    def pick(*names: str) -> str:
        for n in names:
            if n in rows[0]:
                return n
        raise SystemExit(f"{path}: none of {names} among {list(rows[0])}")

    t = pick("true_label", "target", "label", "true")
    p = pick("pred_label", "prediction", "pred", "predicted")
    correct = sum(1 for r in rows if r[t].strip() == r[p].strip())
    return correct / len(rows), len(rows)


def geometry(emb: np.ndarray, lab: np.ndarray) -> Dict[str, float]:
    """Scale-invariant description of how the classes sit in the space.

    intra  -- mean distance from a point to the centroid of its own class
    inter  -- mean distance between class centroids
    ratio  -- inter/intra; above 1 the classes are further apart than they are
              wide, which is the regime where a clustering term can help
    """
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    classes = np.unique(lab)
    cents = np.stack([emb[lab == c].mean(axis=0) for c in classes])
    intra = float(np.mean([
        np.linalg.norm(emb[lab == c] - cents[i], axis=1).mean()
        for i, c in enumerate(classes)]))
    d = [float(np.linalg.norm(cents[i] - cents[j]))
         for i in range(len(classes)) for j in range(i + 1, len(classes))]
    inter = float(np.mean(d))
    return {
        "n": int(len(lab)),
        "dim": int(emb.shape[1]),
        "intra": intra,
        "inter": inter,
        "ratio": inter / intra,
        "silhouette": float(silhouette_score(emb, lab)),
        "davies_bouldin": float(davies_bouldin_score(emb, lab)),
    }


def se_of_accuracy(acc: float, n: int) -> float:
    return math.sqrt(acc * (1.0 - acc) / n)


def collect() -> List[Dict]:
    out = []
    print("=" * 78)
    print("VERIFICATION: summary CSV vs. the run's own predictions.csv")
    print("=" * 78)
    for run in RUNS:
        locate(run)
        row = summary_row(run.summary_csv, run.summary_key)
        filed = float(row["test_accuracy"])
        recomputed, n = accuracy_from_predictions(run.eval_dir)

        emb = np.load(os.path.join(run.eval_dir, "embeddings.npy"))
        lab = np.load(os.path.join(run.eval_dir, "labels.npy"))
        if len(emb) != len(lab):
            raise SystemExit(f"{run.key}: {len(emb)} embeddings vs {len(lab)} labels")
        if len(lab) != n:
            raise SystemExit(
                f"{run.key}: {n} predictions but {len(lab)} embeddings")

        agree = abs(filed - recomputed) < 5e-4
        print(f"  [{'OK ' if agree else 'MISMATCH'}] {run.summary_key:34s} "
              f"filed {filed*100:6.2f}%  recomputed {recomputed*100:6.2f}%  "
              f"n={n}")
        if not agree:
            raise SystemExit(
                f"{run.key}: the summary CSV and predictions.csv disagree; "
                "the summary row is stale, refusing to draw figures from it")

        g = geometry(emb, lab)
        out.append({
            "run": run, "acc": recomputed, "n": n,
            "macro_f1": float(row["macro_f1"]),
            "se": se_of_accuracy(recomputed, n),
            "emb": emb, "lab": lab, **g,
        })
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def tsne(emb: np.ndarray, seed: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, init="pca", learning_rate="auto",
                perplexity=30, random_state=seed).fit_transform(emb)


def panel_grid(recs: List[Dict], keys: List[str], titles: List[str],
               suptitle: str, filename: str, caption: str) -> None:
    import matplotlib.pyplot as plt

    by = {r["run"].key: r for r in recs}
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
    cmap = plt.get_cmap("tab10")

    for ax, key, title in zip(axes.ravel(), keys, titles):
        r = by[key]
        xy = tsne(r["emb"])
        for c, name in enumerate(CLASS_NAMES):
            m = r["lab"] == c
            ax.scatter(xy[m, 0], xy[m, 1], s=3.5, alpha=0.55,
                       color=cmap(c % 10), label=name, linewidths=0)
        ax.set_title(
            f"{title}\nacc {r['acc']*100:.2f}%   "
            f"inter/intra {r['ratio']:.2f}   sil {r['silhouette']:.3f}",
            fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_alpha(0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False,
               markerscale=4, fontsize=10, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle(suptitle, fontsize=13)
    fig.text(0.5, 0.055, caption, ha="center", fontsize=8.5, style="italic",
             color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.085, 1, 0.965))
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def accuracy_figure(recs: List[Dict]) -> None:
    """Every run's accuracy with a one-standard-error bar.

    The error bars are the point of this figure: they are wide enough that most
    of the differences being discussed do not clear them.
    """
    import matplotlib.pyplot as plt

    order = ["orig_ce", "orig_cl", "orig_grad", "orig_alg1",
             "clean_ce", "clean_cl", "clean_grad", "clean_alg1"]
    by = {r["run"].key: r for r in recs}
    names = {
        "orig_ce": "CE", "orig_cl": "CE + CL\n(gradient)",
        "orig_grad": "CL gradient\n(replication)", "orig_alg1": "CL Algorithm 1",
        "clean_ce": "CE", "clean_cl": "CE + CL\n(gradient)",
        "clean_grad": "CL gradient\n(replication)", "clean_alg1": "CL Algorithm 1",
    }

    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = [0, 1, 2, 3, 5, 6, 7, 8]
    accs = [by[k]["acc"] * 100 for k in order]
    ses = [by[k]["se"] * 100 for k in order]
    colors = ["#b0b7c3"] * 4 + ["#3d6fb4"] * 4

    ax.bar(xs, accs, yerr=ses, capsize=4, color=colors, width=0.72,
           error_kw={"ecolor": "0.25", "lw": 1.2})
    for x, a, s in zip(xs, accs, ses):
        ax.text(x, a + s + 0.45, f"{a:.2f}", ha="center", fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels([names[k] for k in order], fontsize=9)
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(55, 74)
    ax.text(1.5, 72.3, ORIGINAL, ha="center", fontsize=10, color="0.3")
    ax.text(6.5, 72.3, CLEAN, ha="center", fontsize=10, color="#3d6fb4")
    ax.axvline(4, color="0.8", lw=1)
    ax.set_title("FER2013: every run, with one standard error", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.005,
             "Error bars are one standard error of the accuracy. Differences "
             "between the four bars within a split are smaller than the bars.",
             ha="center", fontsize=8.5, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = os.path.join(OUT_DIR, "fer_accuracy_all_runs.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def geometry_figure(recs: List[Dict]) -> None:
    """The geometry moves where the accuracy does not."""
    import matplotlib.pyplot as plt

    order = ["orig_ce", "orig_cl", "orig_grad", "orig_alg1",
             "clean_ce", "clean_cl", "clean_grad", "clean_alg1"]
    short = {"orig_ce": "CE", "orig_cl": "CL grad", "orig_grad": "CL grad (rep)",
             "orig_alg1": "CL Alg1", "clean_ce": "CE", "clean_cl": "CL grad",
             "clean_grad": "CL grad (rep)", "clean_alg1": "CL Alg1"}
    by = {r["run"].key: r for r in recs}
    xs = [0, 1, 2, 3, 5, 6, 7, 8]
    colors = ["#b0b7c3"] * 4 + ["#3d6fb4"] * 4

    metrics = [("ratio", "inter / intra  (higher is better)", 1.0),
               ("silhouette", "silhouette  (higher is better)", None),
               ("davies_bouldin", "Davies-Bouldin  (lower is better)", None)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (key, title, ref) in zip(axes, metrics):
        vals = [by[k][key] for k in order]
        ax.bar(xs, vals, color=colors, width=0.72)
        if ref is not None:
            ax.axhline(ref, color="#c0392b", lw=1.1, ls="--")
            ax.text(8.4, ref, " = 1", color="#c0392b", fontsize=8, va="center")
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([short[k] for k in order], fontsize=8, rotation=30,
                           ha="right")
        ax.axvline(4, color="0.85", lw=1)
        ax.set_title(title, fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("FER2013 embedding geometry: grey = original split, "
                 "blue = deduplicated split", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(OUT_DIR, "fer_geometry_all_runs.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def write_table(recs: List[Dict]) -> None:
    path = os.path.join(OUT_DIR, "fer_all_runs.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "dataset", "centre_update", "n_eval", "test_acc",
                    "se_acc", "macro_f1", "intra", "inter", "inter_intra",
                    "silhouette", "davies_bouldin", "run_dir"])
        for r in recs:
            run = r["run"]
            w.writerow([run.key, run.dataset, run.centres, r["n"],
                        f"{r['acc']:.4f}", f"{r['se']:.4f}", r["macro_f1"],
                        f"{r['intra']:.4f}", f"{r['inter']:.4f}",
                        f"{r['ratio']:.4f}", f"{r['silhouette']:.4f}",
                        f"{r['davies_bouldin']:.4f}",
                        os.path.basename(run.run_dir)])
    print(f"  wrote {os.path.relpath(path)}")


def print_contrasts(recs: List[Dict]) -> None:
    """The four comparisons the figures are meant to support, with their noise."""
    by = {r["run"].key: r for r in recs}

    def contrast(a: str, b: str, what: str) -> None:
        ra, rb = by[a], by[b]
        d = rb["acc"] - ra["acc"]
        sed = math.sqrt(ra["se"] ** 2 + rb["se"] ** 2)
        sigma = abs(d) / sed if sed else float("nan")
        verdict = ("indistinguishable from noise" if sigma < 2
                   else "larger than the noise")
        print(f"\n  {what}")
        print(f"    {ra['run'].summary_key:34s} {ra['acc']*100:6.2f}%  "
              f"ratio {ra['ratio']:.2f}  sil {ra['silhouette']:+.3f}")
        print(f"    {rb['run'].summary_key:34s} {rb['acc']*100:6.2f}%  "
              f"ratio {rb['ratio']:.2f}  sil {rb['silhouette']:+.3f}")
        print(f"    accuracy {d*100:+.2f}pp = {sigma:.2f} SE -> {verdict}")
        print(f"    inter/intra {rb['ratio'] - ra['ratio']:+.2f}   "
              f"silhouette {rb['silhouette'] - ra['silhouette']:+.3f}")

    print()
    print("=" * 78)
    print("THE FOUR CONTRASTS")
    print("=" * 78)
    contrast("orig_ce", "orig_cl",
             "A. Center Loss on the ORIGINAL split (with duplicates)")
    contrast("clean_ce", "clean_cl",
             "B. Center Loss on the DEDUPLICATED split")
    contrast("orig_grad", "orig_alg1",
             "C. gradient update vs Algorithm 1, ORIGINAL split")
    contrast("clean_grad", "clean_alg1",
             "D. gradient update vs Algorithm 1, DEDUPLICATED split")
    contrast("orig_ce", "clean_ce",
             "E. what deduplication alone costs (no Center Loss)")
    contrast("clean_cl", "clean_grad",
             "F. replication check: the same algorithm, two separate codebases")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the numbers and print the contrasts, "
                         "but write no figures")
    args = ap.parse_args()

    recs = collect()
    print_contrasts(recs)
    if args.check:
        print("\n--check: no files were written.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    print()
    print("=" * 78)
    print("FIGURES")
    print("=" * 78)
    write_table(recs)
    accuracy_figure(recs)
    geometry_figure(recs)
    panel_grid(
        recs,
        ["orig_ce", "orig_cl", "clean_ce", "clean_cl"],
        [f"{ORIGINAL}\ncross-entropy",
         f"{ORIGINAL}\ncross-entropy + Center Loss",
         f"{CLEAN}\ncross-entropy",
         f"{CLEAN}\ncross-entropy + Center Loss"],
        "Effect of Center Loss, before and after removing duplicate images",
        "fer_tsne_dataset.png",
        "Top row: the original split, whose test partition shares images with "
        "the training partition. Bottom row: the deduplicated split. "
        "Left column: no Center Loss. Right column: Center Loss.")
    panel_grid(
        recs,
        ["orig_grad", "orig_alg1", "clean_grad", "clean_alg1"],
        [f"{ORIGINAL}\ngradient centre update",
         f"{ORIGINAL}\nAlgorithm 1 (Wen et al.)",
         f"{CLEAN}\ngradient centre update",
         f"{CLEAN}\nAlgorithm 1 (Wen et al.)"],
        "Effect of the centre update rule, before and after removing duplicates",
        "fer_tsne_algorithm.png",
        "Both columns use Center Loss; they differ only in how the class "
        "centres are moved. Left: the gradient formulation used in this "
        "thesis. Right: Algorithm 1 as published.")


if __name__ == "__main__":
    sys.exit(main())
