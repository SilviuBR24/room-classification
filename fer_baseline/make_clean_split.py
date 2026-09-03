"""
make_clean_split.py
===================
Builds a deduplicated, leak-free partitioning of FER2013.

The problem this solves
-----------------------
FER2013 ships with a fixed train / val / test split, and that split is not
clean. Measured by hashing decoded pixels:

    314 of 3,589 test images (8.75%) also appear in train or val
    280 of 3,589 validation images (7.80%) also appear in train
    1,236 images inside train are duplicates of another training image
    57 groups of identical images carry contradictory emotion labels

A test image that was also seen during training inflates the reported accuracy.
Excluding such images from the evaluation afterwards fixes the measurement, but
not the model: the checkpoint was still selected on a validation set containing
memorised images, and training still saw some photographs several times and
others with contradictory labels.

What this script produces
-------------------------
A split in which leakage is impossible by construction rather than removed
after the fact. Three decisions, each recorded in the manifest:

1. GROUPS, NOT IMAGES. Every set of pixel-identical images is treated as one
   unit and assigned wholly to one partition. No image content can therefore
   straddle two partitions, including duplicates this project has not
   anticipated.

2. ONE REPRESENTATIVE PER GROUP. Only the first file of each group is written,
   so no photograph is counted twice during training. The representative is
   chosen deterministically, by sorted path.

3. CONTRADICTORY GROUPS ARE DROPPED. Where identical images carry different
   labels, the correct label cannot be established -- not by a model and not by
   a person. Keeping them would mean training on contradictory targets and
   testing against an unanswerable question.

Proportions and per-class balance follow the original split (80 / 10 / 10),
assigned by seeded shuffling within each class, so the class distribution of
the source is preserved.

The original FER2013 partitioning is NOT modified. This writes a new directory.

    python make_clean_split.py
    python make_clean_split.py --src <dataset> --dst <output> --seed 42
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent.parent / "dataset_FER" / "fer2013"
DEFAULT_DST = HERE.parent.parent / "dataset_FER" / "fer2013_clean"

SPLITS = ["train", "val", "eval"]
FRACTIONS = {"train": 0.80, "val": 0.10, "eval": 0.10}
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a deduplicated FER2013 split.")
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--dst", default=str(DEFAULT_DST))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def index_by_content(src: Path) -> Dict[str, List[Tuple[str, str, Path]]]:
    """hash of the decoded pixels -> [(split, class, path), ...]"""
    index: Dict[str, List[Tuple[str, str, Path]]] = {}
    for split in SPLITS:
        for cls_dir in sorted((src / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() != ".png":
                    continue
                with Image.open(f) as im:
                    arr = np.asarray(im.convert("L"))
                h = hashlib.sha1(arr.tobytes()).hexdigest()
                index.setdefault(h, []).append((split, cls_dir.name, f))
    return index


def main() -> None:
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")

    print(f"source: {src}")
    index = index_by_content(src)
    n_images = sum(len(v) for v in index.values())
    print(f"  {n_images:,} images, {len(index):,} distinct by content")

    # ---- drop groups whose members disagree about the label ----------
    kept: Dict[str, List[Tuple[str, str, Path]]] = {}
    dropped: Dict[str, List[Tuple[str, str, Path]]] = {}
    for h, items in index.items():
        if len({c for _, c, _ in items}) > 1:
            dropped[h] = items
        else:
            kept[h] = items
    print(f"  groups with contradictory labels dropped: {len(dropped)} "
          f"({sum(len(v) for v in dropped.values())} images)")
    print(f"  groups kept: {len(kept):,}")

    # ---- assign whole groups to partitions, stratified by class ------
    by_class: Dict[str, List[str]] = collections.defaultdict(list)
    for h, items in kept.items():
        by_class[items[0][1]].append(h)

    assignment: Dict[str, str] = {}
    rng = random.Random(args.seed)
    for cls in CLASSES:
        # Sort before shuffling: directory order is filesystem-dependent, so an
        # unsorted list would give a different split on another machine even
        # with the same seed.
        hashes = sorted(by_class[cls])
        rng.shuffle(hashes)
        n = len(hashes)
        n_train = int(round(n * FRACTIONS["train"]))
        n_val = int(round(n * FRACTIONS["val"]))
        for i, h in enumerate(hashes):
            if i < n_train:
                assignment[h] = "train"
            elif i < n_train + n_val:
                assignment[h] = "val"
            else:
                assignment[h] = "eval"

    # ---- write one representative per group --------------------------
    if dst.exists():
        shutil.rmtree(dst)
    for split in SPLITS:
        for cls in CLASSES:
            (dst / split / cls).mkdir(parents=True, exist_ok=True)

    manifest = []
    counts: Dict[str, collections.Counter] = {s: collections.Counter() for s in SPLITS}
    for h, items in kept.items():
        split = assignment[h]
        cls = items[0][1]
        # deterministic representative: first by sorted original path
        rep = sorted(items, key=lambda t: str(t[2]))[0]
        target = dst / split / cls / f"{h[:12]}.png"
        shutil.copy2(rep[2], target)
        counts[split][cls] += 1
        manifest.append({
            "hash": h[:12], "assigned_split": split, "class": cls,
            "n_copies_in_source": len(items),
            "representative": str(rep[2].relative_to(src)).replace("\\", "/"),
            "original_splits": "|".join(sorted({s for s, _, _ in items})),
        })

    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(sorted(manifest, key=lambda r: (r["assigned_split"], r["class"], r["hash"])))

    with open(dst / "dropped_label_conflicts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["hash", "labels", "n_images", "files"])
        for h, items in sorted(dropped.items()):
            w.writerow([h[:12], "|".join(sorted({c for _, c, _ in items})), len(items),
                        "|".join(str(p.relative_to(src)).replace("\\", "/")
                                 for _, _, p in items)])

    # ---- report -------------------------------------------------------
    print("\nclean split written:")
    total = 0
    for split in SPLITS:
        n = sum(counts[split].values())
        total += n
        print(f"  {split:6s}: {n:6,}  {dict(counts[split])}")
    print(f"  total : {total:,}")

    print("\nverification:")
    print(f"  [{'OK ' if total == len(kept) else 'FAIL'}] every kept group written exactly once")
    all_hashes = {s: set() for s in SPLITS}
    for r in manifest:
        all_hashes[r["assigned_split"]].add(r["hash"])
    overlap = (all_hashes["train"] & all_hashes["val"]) | \
              (all_hashes["train"] & all_hashes["eval"]) | \
              (all_hashes["val"] & all_hashes["eval"])
    print(f"  [{'OK ' if not overlap else 'FAIL'}] no content shared between partitions "
          f"({len(overlap)} overlaps)")
    for split in SPLITS:
        files = sum(1 for _ in (dst / split).rglob("*.png"))
        ok = files == sum(counts[split].values())
        print(f"  [{'OK ' if ok else 'FAIL'}] {split}: {files:,} files on disk")

    print(f"\nOutput: {dst}")
    print("The original FER2013 partitioning was not modified.")


if __name__ == "__main__":
    main()
