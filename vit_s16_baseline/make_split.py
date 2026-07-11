"""
make_split.py
=============
Deterministic labeled/unlabeled + train/val split for the thesis.

From a source dataset with `train/<class>/` (the full labeled pool, e.g.
5000/class) and `eval/<class>/` (the held-out test set, e.g. 100/class), it
produces a split directory:

    <dst>/train/      N_TRAIN per class            -> fit weights
    <dst>/val/        (N_LABELED - N_TRAIN)/class  -> best-model selection
    <dst>/unlabeled/  the rest per class           -> Part 3 self-labeling
    <dst>/eval/       copied test set              -> final reporting

Determinism: a fixed --seed drives one shuffle per class (in the CLASSES
order). The labeled/unlabeled boundary is files[:N_LABELED] / files[N_LABELED:],
and the labeled part is split into train (first N_TRAIN) and val (the rest), so
lowering N_TRAIN only moves images from train to val and NEVER changes which
images are unlabeled. This is exactly how the thesis split was produced
(seed=42, n_train=3200, n_labeled=3500 -> 3200/300/1500 per class).

Usage:
    python make_split.py --src ../dataset --dst ../dataset_split
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

CLASSES = ["bathroom", "bedroom", "dining_room", "entrance_hall", "kitchen", "living_room"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic train/val/unlabeled split.")
    ap.add_argument("--src", required=True, help="source dataset dir (has train/ and eval/)")
    ap.add_argument("--dst", required=True, help="output split dir (overwritten)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=3200, help="labeled images/class used for training")
    ap.add_argument("--n-labeled", type=int, default=3500, help="labeled images/class (train + val)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if args.n_train > args.n_labeled:
        raise ValueError("--n-train must be <= --n-labeled")

    if dst.exists():
        shutil.rmtree(dst)
    for sub in ("train", "val", "unlabeled", "eval"):
        for c in CLASSES:
            (dst / sub / c).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for c in CLASSES:
        files = sorted(os.listdir(src / "train" / c))  # deterministic base order
        rng.shuffle(files)
        train = files[: args.n_train]
        val = files[args.n_train : args.n_labeled]
        unlabeled = files[args.n_labeled :]

        for f in train:
            shutil.copy(src / "train" / c / f, dst / "train" / c / f)
        for f in val:
            shutil.copy(src / "train" / c / f, dst / "val" / c / f)
        for f in unlabeled:
            shutil.copy(src / "train" / c / f, dst / "unlabeled" / c / f)
        eval_files = os.listdir(src / "eval" / c)
        for f in eval_files:
            shutil.copy(src / "eval" / c / f, dst / "eval" / c / f)

        print(f"{c:15s}: train={len(train)} val={len(val)} "
              f"unlabeled={len(unlabeled)} test={len(eval_files)}")

    print("DONE ->", dst)


if __name__ == "__main__":
    main()
