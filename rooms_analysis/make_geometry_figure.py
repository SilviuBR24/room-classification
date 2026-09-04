"""The central figure of the room experiment: what Centre Loss did to the space.

The thesis asks whether Centre Loss gathers the embeddings of a class together.
It does, and by a lot. The figure exists to show the second half of that
sentence: it gathers the classes together too, by the same proportion, so the
classes end up no better separated than before.

Everything is computed from the embeddings each evaluation already saved. No
model is re-run. The runs are the four from the valid 3,200/300/600 protocol,
in which the checkpoint was selected on validation and the test set was untouched
until the final evaluation.

    python make_geometry_figure.py
"""
from __future__ import annotations

import glob
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RUNS_DIR = r"G:\My Drive\Dissertation_Thesis\dissertation_runs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "figuri", "rooms_geometry")

# label, run directory, and the lambda it was trained with
RUNS = [
    ("cross-entropy\nonly", "2026-07-11_17-49_vit_s16_3200_baseline", None),
    ("+ Centre Loss\nλ = 0.0005", "2026-07-11_19-45_vit_s16_3200_center_l00005", 5e-4),
    ("+ Centre Loss\nλ = 0.001", "2026-07-12_09-27_vit_s16_3200_center_l0001", 1e-3),
    ("+ Centre Loss\nλ = 0.01", "2026-07-12_11-25_vit_s16_3200_center_l001", 1e-2),
]
N_BOOT = 2000


def newest_eval(run: str) -> str:
    hits = sorted(glob.glob(os.path.join(RUNS_DIR, run, "outputs", "eval_*")))
    if not hits:
        raise SystemExit(f"no evaluation directory under {run}")
    return hits[-1]


def load(run: str):
    ev = newest_eval(run)
    return (np.load(os.path.join(ev, "embeddings.npy")),
            np.load(os.path.join(ev, "labels.npy")))


def intra_inter(emb: np.ndarray, lab: np.ndarray) -> tuple[float, float]:
    """Mean within-class spread, and mean distance between class centroids."""
    classes = np.unique(lab)
    cents = np.stack([emb[lab == c].mean(axis=0) for c in classes])
    intra = float(np.mean([np.linalg.norm(emb[lab == c] - cents[i], axis=1).mean()
                           for i, c in enumerate(classes)]))
    d = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=-1)
    inter = float(d[np.triu_indices(len(classes), k=1)].mean())
    return intra, inter


def paired_bootstrap(a, b, lab, n_boot=N_BOOT, seed=0):
    """Interval for the difference in inter/intra between two runs.

    Paired and stratified: one draw of indices is applied to both runs, within
    each class, because the two are scored on the same images and the class
    proportions of the evaluation set are fixed by design.
    """
    groups = [np.flatnonzero(lab == c) for c in np.unique(lab)]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for t in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        y = lab[idx]
        ia, na = intra_inter(a[idx], y)
        ib, nb = intra_inter(b[idx], y)
        out[t] = nb / ib - na / ia
    return np.percentile(out, [2.5, 97.5])


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, intra, inter, ratio = [], [], [], []
    base_emb = base_lab = None
    cis = [None]

    for i, (label, run, _) in enumerate(RUNS):
        emb, lab = load(run)
        if i == 0:
            base_emb, base_lab = emb, lab
        elif not np.array_equal(lab, base_lab):
            raise SystemExit(f"{run} is not scored on the same images as the baseline")
        a, b = intra_inter(emb, lab)
        labels.append(label); intra.append(a); inter.append(b); ratio.append(b / a)
        if i > 0:
            lo, hi = paired_bootstrap(base_emb, emb, lab)
            cis.append((lo, hi))
            print(f"  {label.replace(chr(10), ' '):28s} intra {a:6.3f}  inter {b:6.3f}  "
                  f"ratio {b/a:.4f}   diff {b/a - ratio[0]:+.4f}  "
                  f"95% [{lo:+.4f}, {hi:+.4f}]")
        else:
            print(f"  {label.replace(chr(10), ' '):28s} intra {a:6.3f}  inter {b:6.3f}  "
                  f"ratio {b/a:.4f}   (reference)")

    os.makedirs(OUT, exist_ok=True)
    x = np.arange(len(RUNS))
    ink, grey = "#1a1f26", "#8a929c"
    c_intra, c_inter, c_ratio = "#c0562f", "#2f6ea8", "#3d3f45"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.9),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # --- left: the two distances, which collapse together --------------------
    w = 0.36
    ax1.bar(x - w / 2, intra, w, color=c_intra, label="within a class")
    ax1.bar(x + w / 2, inter, w, color=c_inter, label="between classes")
    for xi, (a, b) in enumerate(zip(intra, inter)):
        ax1.text(xi - w / 2, a + 0.15, f"{a:.2f}", ha="center", fontsize=8.5, color=ink)
        ax1.text(xi + w / 2, b + 0.15, f"{b:.2f}", ha="center", fontsize=8.5, color=ink)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("mean distance in the embedding space")
    ax1.set_title("Centre Loss contracts the space —\nboth distances, by the same "
                  "proportion", fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_ylim(0, max(intra) * 1.18)

    # How much each distance shrank, placed above the pair so the two
    # percentages can be read against each other -- that they are nearly equal
    # is the whole point of the panel.
    for xi in range(1, len(RUNS)):
        pa = 100 * (1 - intra[xi] / intra[0])
        pb = 100 * (1 - inter[xi] / inter[0])
        top = max(intra[xi], inter[xi])
        ax1.annotate(f"−{pa:.0f}%  vs  −{pb:.0f}%", xy=(xi, top + 0.62),
                     ha="center", fontsize=8.5, color=grey,
                     bbox=dict(boxstyle="round,pad=0.28", fc="white",
                               ec=grey, lw=0.5, alpha=0.9))

    # --- right: the ratio, which is what separability depends on -------------
    ax2.axhline(ratio[0], color=grey, ls="--", lw=1)
    ax2.text(len(RUNS) - 0.45, ratio[0], " baseline", va="center", fontsize=8.5,
             color=grey)
    ax2.plot(x, ratio, "o-", color=c_ratio, lw=1.6, ms=7, zorder=3)
    for xi, r in enumerate(ratio):
        if cis[xi] is not None:
            lo, hi = cis[xi]
            ax2.plot([xi, xi], [ratio[0] + lo, ratio[0] + hi], color=c_ratio,
                     lw=1.2, alpha=0.55, zorder=2)
        ax2.text(xi, r + 0.011, f"{r:.3f}", ha="center", fontsize=9, color=ink)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("between / within  (higher is better separated)")
    ax2.set_title("Separation does not improve —\nat λ = 0.0005 it is measurably "
                  "worse", fontsize=11, loc="left")
    # Room for the widest bootstrap interval, not just for the points.
    los = [ratio[0] + c[0] for c in cis if c] + ratio
    his = [ratio[0] + c[1] for c in cis if c] + ratio
    pad = (max(his) - min(los)) * 0.14
    ax2.set_ylim(min(los) - pad, max(his) + pad)

    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink)

    fig.text(0.5, 0.012,
             "ViT-S/16 trained from scratch, 600 held-out test images. Bars on the "
             "right are 95% paired, class-stratified bootstrap intervals for the "
             "difference from the baseline.",
             ha="center", fontsize=8.5, color=grey, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(OUT, "rooms_centre_loss_geometry.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"\n  wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
