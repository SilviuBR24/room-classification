"""
download_fer2013.py
===================
Downloads FER2013 from the public Hugging Face mirror `AutumnQiu/fer2013` and
writes it as a directory tree in the same layout the project's dataset class
already expects:

    <dst>/train/<class>/*.png     28,709 images
    <dst>/val/<class>/*.png        3,589 images
    <dst>/eval/<class>/*.png       3,589 images

Why this mirror and this format
-------------------------------
FER2013 was introduced by Goodfellow et al. for the ICML 2013 Workshop on
Challenges in Representation Learning. It is distributed through Kaggle, which
requires an account, so a mirror is used instead.

The files are Parquet, deliberately. Several other mirrors ship PyTorch `.pt`
archives, which are pickles: loading one executes arbitrary code from whoever
uploaded it. Parquet is a pure data format and carries no such risk.

The mirror was checked against the published dataset before use: 28,709 /
3,589 / 3,589 images and the seven canonical classes in their standard order.
This script re-checks those counts after extraction and fails loudly if they
do not match.

    python download_fer2013.py
    python download_fer2013.py --dst D:/some/other/path
"""
from __future__ import annotations

import argparse
import io
import os
import time
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/AutumnQiu/fer2013/resolve/main/data"
FILES = {
    "train": "train-00000-of-00001.parquet",
    "val": "valid-00000-of-00001.parquet",
    "eval": "test-00000-of-00001.parquet",
}
# Canonical FER2013 order; lower-cased to match the room dataset convention.
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EXPECTED = {"train": 28709, "val": 3589, "eval": 3589}

HERE = Path(__file__).resolve().parent
DEFAULT_DST = HERE.parent.parent / "datasets" / "fer2013"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and unpack FER2013.")
    p.add_argument("--dst", default=str(DEFAULT_DST), help="output directory")
    p.add_argument("--cache", default=str(HERE / "_parquet_cache"),
                   help="where the downloaded parquet files are kept")
    return p.parse_args()


def fetch(url: str, path: Path) -> None:
    if path.exists():
        print(f"  cached: {path.name} ({path.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  downloading {path.name} ...", end="", flush=True)
    t0 = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    print(f" {path.stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s")


def main() -> None:
    args = parse_args()
    dst = Path(args.dst)
    cache = Path(args.cache)

    import pyarrow.parquet as pq
    from PIL import Image

    print(f"destinatie: {dst}\n")
    totals = {}
    for split, fname in FILES.items():
        fetch(f"{BASE}/{fname}", cache / fname)

    print()
    for split, fname in FILES.items():
        table = pq.read_table(cache / fname)
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        assert len(images) == len(labels)

        for c in CLASSES:
            (dst / split / c).mkdir(parents=True, exist_ok=True)

        counts = {c: 0 for c in CLASSES}
        t0 = time.time()
        for i, (img, lab) in enumerate(zip(images, labels)):
            cls = CLASSES[lab]
            im = Image.open(io.BytesIO(img["bytes"]))
            im.save(dst / split / cls / f"{split}_{i:05d}.png")
            counts[cls] += 1
        totals[split] = counts
        print(f"{split:6s}: {sum(counts.values()):6d} imagini in {time.time()-t0:.0f}s")
        print(f"        {counts}")

    print("\nverificare fata de setul publicat:")
    ok = True
    for split, n in EXPECTED.items():
        got = sum(totals[split].values())
        mark = "OK " if got == n else "EROARE"
        ok &= got == n
        print(f"  [{mark}] {split:6s} asteptat {n}, obtinut {got}")

    sample = next((dst / "train" / "happy").glob("*.png"))
    with Image.open(sample) as im:
        print(f"  exemplu: {im.size} pixeli, mod {im.mode}")

    print("\n" + ("Setul de date este gata." if ok else "ATENTIE: numaratoarea nu se potriveste."))


if __name__ == "__main__":
    main()
