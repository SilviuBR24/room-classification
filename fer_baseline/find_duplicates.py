"""
find_duplicates.py
==================
Identifies images that appear more than once in FER2013 and writes everything
needed to inspect them by eye before anything is excluded.

Why this matters
----------------
FER2013 is distributed as a fixed train / val / test split, and that split is
not clean: the same photograph appears in more than one partition. A test image
that was also seen during training inflates the reported accuracy, because the
model is being asked about something it has already memorised.

The dataset is also internally inconsistent: some identical photographs carry
different emotion labels, which is a known property of FER2013 rather than a
fault of this project.

Nothing is deleted here. This script only reports, so the groups can be checked
before any decision is made about them.

How duplicates are detected
---------------------------
By hashing the decoded pixels, not the file. Two files with different names,
different compression or different metadata still hash the same if the image
content is identical. The comparison is done on the greyscale channel, which is
all FER2013 carries.

Outputs, under --out:

    manifest.csv          every duplicate group, one row per image
    summary.txt           the counts, in the form used in the thesis
    sheets/               contact sheets: each group drawn side by side,
                          labelled with its partition and class
    groups/               the original files, copied one folder per group

    python find_duplicates.py
    python find_duplicates.py --root <dataset> --out <review folder>
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent.parent / "dataset_FER" / "fer2013"
DEFAULT_OUT = HERE.parent.parent / "dataset_FER" / "duplicates_review"
SPLITS = ["train", "val", "eval"]
GROUPS_PER_SHEET = 12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find and display FER2013 duplicates.")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--copy-files", action="store_true", default=True,
                   help="also copy the original files, one folder per group")
    return p.parse_args()


def index_images(root: Path) -> Dict[str, List[Tuple[str, str, Path]]]:
    """hash -> [(split, class, path), ...]"""
    index: Dict[str, List[Tuple[str, str, Path]]] = {}
    for split in SPLITS:
        for cls_dir in sorted((root / split).iterdir()):
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


def classify(items: List[Tuple[str, str, Path]]) -> str:
    """Which review category does this group belong to?"""
    splits = {s for s, _, _ in items}
    labels = {c for _, c, _ in items}
    if len(labels) > 1:
        return "label_conflict"
    if "eval" in splits and len(splits) > 1:
        return "eval_contaminated"
    if "val" in splits and "train" in splits:
        return "val_contaminated"
    return "within_split"


def contact_sheet(groups: List[Tuple[str, List]], path: Path, title: str) -> None:
    """Draw several duplicate groups on one page, labelled."""
    widest = max(len(items) for _, items in groups)
    rows = len(groups)
    fig, axes = plt.subplots(rows, widest,
                             figsize=(1.5 * widest + 1.4, 1.75 * rows),
                             squeeze=False)
    for r, (h, items) in enumerate(groups):
        for c in range(widest):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(items):
                split, cls, path_ = items[c]
                with Image.open(path_) as im:
                    ax.imshow(np.asarray(im.convert("L")), cmap="gray")
                colour = {"train": "#1f77b4", "val": "#ff7f0e", "eval": "#d62728"}[split]
                ax.set_title(f"{split} / {cls}\n{path_.name}", fontsize=6.5,
                             color=colour)
                for side in ax.spines.values():
                    side.set_edgecolor(colour)
                    side.set_linewidth(1.8)
            else:
                ax.axis("off")
        axes[r][0].set_ylabel(h[:7], fontsize=6, rotation=0,
                              labelpad=22, va="center")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root, out = Path(args.root), Path(args.out)
    if not root.is_dir():
        raise SystemExit(f"Dataset not found: {root}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(exist_ok=True)

    print(f"indexing {root} ...")
    index = index_images(root)
    total = sum(len(v) for v in index.values())
    print(f"  {total:,} images, {len(index):,} distinct")

    buckets: Dict[str, List] = collections.defaultdict(list)
    for h, items in index.items():
        if len(items) > 1:
            buckets[classify(items)].append((h, items))

    # ---------------- manifest -------------------------------------
    rows = []
    for kind, groups in buckets.items():
        for gi, (h, items) in enumerate(sorted(groups), 1):
            for split, cls, path_ in items:
                rows.append({"category": kind, "group": f"{kind}_{gi:04d}",
                             "hash": h[:12], "split": split, "class": cls,
                             "file": path_.name,
                             "relative_path": str(path_.relative_to(root)).replace("\\", "/")})
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- summary --------------------------------------
    counts = collections.Counter(s for s, _, _ in
                                 (i for v in index.values() for i in v))
    lines = []
    lines.append("FER2013 duplicate review")
    lines.append("=" * 60)
    lines.append(f"images indexed : {dict(counts)}  total {total:,}")
    lines.append(f"distinct images: {len(index):,}")
    lines.append(f"redundant      : {total - len(index):,}")
    lines.append("")
    lines.append("groups by category:")
    for kind in ["eval_contaminated", "val_contaminated", "label_conflict", "within_split"]:
        g = buckets.get(kind, [])
        n_img = sum(len(items) for _, items in g)
        lines.append(f"  {kind:20s} {len(g):5d} groups, {n_img:5d} images")
    lines.append("")

    # how many TEST images are affected -- the number that matters
    affected_eval = set()
    for kind in ("eval_contaminated", "label_conflict"):
        for h, items in buckets.get(kind, []):
            splits = {s for s, _, _ in items}
            if "eval" in splits and len(splits) > 1:
                for s, c, p in items:
                    if s == "eval":
                        affected_eval.add(str(p.relative_to(root)).replace("\\", "/"))
    lines.append(f"TEST images that also appear in train or val: {len(affected_eval)} "
                 f"of {counts['eval']} ({100.0 * len(affected_eval) / counts['eval']:.2f}%)")
    lines.append("These are the images whose exclusion changes the reported accuracy.")
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(out / "contaminated_eval.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(affected_eval)))
    print("\n".join(lines))

    # ---------------- contact sheets --------------------------------
    for kind in ["eval_contaminated", "val_contaminated", "label_conflict"]:
        groups = sorted(buckets.get(kind, []))
        for start in range(0, len(groups), GROUPS_PER_SHEET):
            chunk = groups[start:start + GROUPS_PER_SHEET]
            n = start // GROUPS_PER_SHEET + 1
            contact_sheet(chunk, out / "sheets" / f"{kind}_{n:03d}.png",
                          f"{kind.replace('_', ' ')} -- sheet {n} "
                          f"(groups {start + 1}-{start + len(chunk)} of {len(groups)})")
        print(f"  sheets for {kind}: {(len(groups) + GROUPS_PER_SHEET - 1) // GROUPS_PER_SHEET}")

    # ---------------- raw copies ------------------------------------
    if args.copy_files:
        gdir = out / "groups"
        if gdir.exists():
            shutil.rmtree(gdir)
        for kind in ["eval_contaminated", "val_contaminated", "label_conflict"]:
            for gi, (h, items) in enumerate(sorted(buckets.get(kind, [])), 1):
                d = gdir / kind / f"{gi:04d}_{h[:8]}"
                d.mkdir(parents=True, exist_ok=True)
                for split, cls, path_ in items:
                    shutil.copy2(path_, d / f"{split}__{cls}__{path_.name}")
        print(f"  raw copies written to {gdir}")

    print(f"\nReview folder: {out}")


if __name__ == "__main__":
    main()
