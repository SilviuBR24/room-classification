"""Measure what the duplicate images were worth, inside one test set.

The obvious way to state the cost of deduplication is to subtract the two
accuracies: 68.40% on the original split, 65.71% on the deduplicated one, so
2.70pp. That subtraction is weak evidence, because the two numbers come from
different test sets of different sizes, and no paired test applies to them.

This measures the same thing without that objection. The model trained on the
original split is scored only on its own test set, which is then divided in two:
the images whose pixels also occur somewhere in the training partition, and the
rest. Both groups are judged by one model in one evaluation, so the comparison
is internal and the difference cannot be attributed to a change of test set.

The identity used is the one the deduplication itself used: SHA-1 over the
decoded greyscale pixel array, so a group here is a group there.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

FER_ROOT = Path(r"C:\Users\gsilv\Desktop\Facultate\Master\Reasearch\Disertatie"
                r"\dataset_FER\fer2013")
RUNS_DIR = r"G:\My Drive\Dissertation_Thesis\dissertation_runs"
CONTAMINATED_RUN = "2026-09-02_19-52_fer_crossentropy"
CLEAN_ACCURACY = 0.6571          # fer_clean_crossentropy, for the cross-check
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "figuri", "fer_comparison")


def index_split(split: str) -> Dict[str, List[Tuple[str, str]]]:
    """hash -> [(class, filename), ...], hashed exactly as find_duplicates.py does."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    for cls in sorted((FER_ROOT / split).iterdir()):
        if not cls.is_dir():
            continue
        for f in sorted(cls.iterdir()):
            if f.suffix.lower() != ".png":
                continue
            with Image.open(f) as im:
                arr = np.asarray(im.convert("L"))
            out.setdefault(hashlib.sha1(arr.tobytes()).hexdigest(), []).append(
                (cls.name, f.name))
    return out


def wilson(k: int, n: int) -> Tuple[float, float]:
    """95% Wilson interval -- honest at the small n of the leaked group."""
    if n == 0:
        return 0.0, 0.0
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    print("hashing the original split ...", flush=True)
    train, val, ev = (index_split(s) for s in ("train", "val", "eval"))
    n_eval = sum(len(v) for v in ev.values())

    in_train = {h for h in ev if h in train}
    in_either = {h for h in ev if h in train or h in val}
    n = lambda S: sum(len(ev[h]) for h in S)

    print(f"\ntest images: {n_eval}, of which")
    print(f"  also in the training partition   : {n(in_train):4d} "
          f"({n(in_train)/n_eval*100:.2f}%)")
    print(f"  also in training or validation   : {n(in_either):4d} "
          f"({n(in_either)/n_eval*100:.2f}%)")
    print("  The stricter count is used below: those images entered the "
          "gradient updates.\n  The wider count adds images that only "
          "influenced model selection.")

    leaked = {}
    for h in ev:
        for cls, fn in ev[h]:
            leaked[(cls, fn)] = h in in_train

    hits = glob.glob(os.path.join(RUNS_DIR, CONTAMINATED_RUN,
                                  "outputs", "eval_*", "predictions.csv"))
    if len(hits) != 1:
        raise SystemExit(f"expected one predictions.csv, found {hits}")
    with open(hits[0], newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    groups: Dict[bool, List[bool]] = {True: [], False: []}
    for r in rows:
        p = Path(r["filepath"])
        key = (p.parent.name, p.name)
        if key not in leaked:
            raise SystemExit(f"{key} is not in the local copy of the dataset")
        groups[leaked[key]].append(r["true_label"].strip() == r["pred_label"].strip())

    print("=" * 74)
    print("THE CONTAMINATED MODEL, SPLIT INSIDE ITS OWN TEST SET")
    print("=" * 74)
    stats = {}
    for k, name in [(True, "images also seen in training"),
                    (False, "images never seen")]:
        b = groups[k]
        acc = sum(b) / len(b)
        lo, hi = wilson(sum(b), len(b))
        stats[k] = (acc, lo, hi, len(b))
        print(f"  {name:30s} n={len(b):5d}   {acc*100:6.2f}%   "
              f"95% CI [{lo*100:.2f}, {hi*100:.2f}]")

    a1, a0 = stats[True][0], stats[False][0]
    sed = sqrt(a1 * (1 - a1) / stats[True][3] + a0 * (1 - a0) / stats[False][3])
    print(f"\n  gap {(a1-a0)*100:+.2f}pp = {abs(a1-a0)/sed:.1f} standard errors")
    print("  The two groups are disjoint sets of images, so they are "
          "independent samples\n  and the pooled standard error applies here.")

    print(f"\n  reported overall accuracy      : "
          f"{sum(sum(b) for b in groups.values())/len(rows)*100:.2f}%")
    print(f"  on unseen images only          : {a0*100:.2f}%")
    print(f"  clean split, separate training : {CLEAN_ACCURACY*100:.2f}%")
    print(f"  difference between the last two: {(a0-CLEAN_ACCURACY)*100:+.2f}pp")
    print("\n  Those last two numbers come from different models trained on\n"
          "  different data, and they agree. That is the corroboration: the\n"
          "  deduplicated result is what the contaminated model was already\n"
          "  achieving once its memorised images are set aside.")

    if args.no_figure:
        return

    import matplotlib.pyplot as plt
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 5))
    labels = ["seen in training\n(n=%d)" % stats[True][3],
              "never seen\n(n=%d)" % stats[False][3],
              "deduplicated split\n(separate model)"]
    vals = [a1 * 100, a0 * 100, CLEAN_ACCURACY * 100]
    errs = [[(a1 - stats[True][1]) * 100, (a0 - stats[False][1]) * 100, 0],
            [(stats[True][2] - a1) * 100, (stats[False][2] - a0) * 100, 0]]
    ax.bar([0, 1], vals[:2], yerr=[errs[0][:2], errs[1][:2]], capsize=5,
           color=["#c0392b", "#b0b7c3"], width=0.6,
           error_kw={"ecolor": "0.25", "lw": 1.2})
    ax.bar([2.4], vals[2:], color="#3d6fb4", width=0.6)
    for x, v in zip([0, 1, 2.4], vals):
        ax.text(x, v + 1.6, f"{v:.2f}%", ha="center", fontsize=11)
    ax.set_xticks([0, 1, 2.4])
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 104)
    ax.axhline(a0 * 100, color="0.6", ls="--", lw=1)
    ax.set_title("What the duplicate images were worth\n"
                 "one model, one test set, divided by whether the image "
                 "occurs in training", fontsize=11.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.015,
             "Error bars are 95% Wilson intervals. The blue bar is a different "
             "model trained on the deduplicated split;\nit lands where the "
             "contaminated model already sat on images it had not memorised.",
             ha="center", fontsize=8.5, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(OUT_DIR, "fer_leakage_effect.png")
    fig.savefig(path, dpi=200)
    print(f"\n  wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
