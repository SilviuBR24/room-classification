"""
make_fer_zip.py
===============
Packs the extracted FER2013 tree into a single zip for upload to Google Drive
and use on Colab.

Written with Python's `zipfile` on purpose. Archives produced by Windows
Explorer or PowerShell's Compress-Archive store entry names with backslash
separators, which Linux unzip treats as part of the file name rather than as
directories, so the tree arrives flattened on Colab. `zipfile` writes forward
slashes, which extract correctly everywhere.

    python make_fer_zip.py
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent.parent / "datasets" / "fer2013"
DEFAULT_OUT = HERE.parent.parent / "datasets" / "fer2013.zip"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zip the FER2013 tree.")
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")

    files = sorted(p for p in src.rglob("*") if p.is_file())
    print(f"source : {src}")
    print(f"files  : {len(files):,}")

    t0 = time.time()
    # PNGs are already compressed; storing them avoids a slow, pointless pass.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for i, f in enumerate(files, 1):
            z.write(f, arcname=f"fer2013/{f.relative_to(src).as_posix()}")
            if i % 10000 == 0:
                print(f"   {i:,}/{len(files):,}")

    size = out.stat().st_size / 1e6
    print(f"\nwrote  : {out}  ({size:.0f} MB in {time.time()-t0:.0f}s)")

    # Verify the archive uses forward slashes and round-trips.
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        bad = [n for n in names[:2000] if "\\" in n]
        print(f"entries: {len(names):,}")
        print(f"backslash separators in first 2000 entries: {len(bad)} (must be 0)")
        print(f"example entry: {names[1]}")


if __name__ == "__main__":
    main()
