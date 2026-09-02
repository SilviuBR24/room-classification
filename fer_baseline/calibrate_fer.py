"""
calibrate_fer.py
================
Chooses the input resolution for the facial-expression experiment by measuring
it, rather than by assuming.

The question
------------
FER2013 images are 48x48. ResNet-18 was designed for 224x224 and reduces its
input by a factor of 32 on the way to the pooling layer:

    input 48x48   -> stem leaves 12x12 -> 2x2 reaches pooling
    input 96x96   -> stem leaves 24x24 -> 3x3 reaches pooling
    input 224x224 -> stem leaves 56x56 -> 7x7 reaches pooling

Upsampling a 48-pixel image adds no information. What it does is stop the fixed
7x7 stride-2 stem and max-pool from discarding most of the image before any
residual block runs. Whether that trade is worth the extra computation is an
empirical question, so this script runs a short training at each candidate
resolution and reports the validation accuracy reached.

Both trials use the same seed, the same data and the same schedule, so the only
thing that differs is the resolution. The result is not a final number -- the
runs are deliberately short -- it is a comparison used to pick one value before
committing to the full experiment.

    python calibrate_fer.py --data-root ../../datasets/fer2013
    python calibrate_fer.py --data-root ... --sizes 48 96 --epochs 5
"""
from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pick the FER input resolution by measurement.")
    p.add_argument("--config", default=str(HERE / "config_fer.yaml"),
                   help="base configuration; only resolution and length are overridden")
    p.add_argument("--data-root", required=True,
                   help="directory containing train/ val/ eval/")
    p.add_argument("--sizes", type=int, nargs="+", default=[48, 96],
                   help="input resolutions to compare")
    p.add_argument("--epochs", type=int, default=5,
                   help="epochs per trial; short on purpose")
    p.add_argument("--out-root", default=None,
                   help="where trial runs are written (default: a temporary directory)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="override the config's dataloader worker count")
    return p.parse_args()


def run_trial(base_cfg: Dict, size: int, args: argparse.Namespace,
              out_root: Path) -> Optional[Dict]:
    """Train once at one resolution and return what it reached."""
    cfg = copy.deepcopy(base_cfg)
    cfg["run_name"] = f"fer_calib_{size}"
    cfg["model"]["image_size"] = size
    cfg["training"]["epochs"] = args.epochs
    # A short trial should not spend most of its budget warming up.
    cfg["training"]["warmup_epochs"] = min(
        int(cfg["training"].get("warmup_epochs", 5)), max(1, args.epochs // 5))
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    root = Path(args.data_root)
    cfg["data"]["train_dir"] = str(root / "train")
    cfg["data"]["val_dir"] = str(root / "val")
    cfg["data"]["eval_dir"] = str(root / "eval")
    cfg["paths"]["output_root"] = str(out_root)

    cfg_path = out_root / f"config_calib_{size}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    print(f"\n{'='*70}\nTRIAL: {size}x{size}, {args.epochs} epochs\n{'='*70}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-u", str(HERE / "train_fer.py"), "--config", str(cfg_path)],
        cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={"TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1", **_env()},
    )
    lines: List[str] = []
    for raw in proc.stdout:
        line = raw.decode("utf-8", "replace")
        lines.append(line)
        if "Epoch" in line or "ERROR" in line or "Traceback" in line:
            sys.stdout.write(line)
            sys.stdout.flush()
    code = proc.wait()
    elapsed = time.time() - t0

    text = "".join(lines)
    if code != 0:
        print(f"  trial failed (exit {code}); last output:")
        print("  " + "\n  ".join(text.strip().splitlines()[-8:]))
        return None

    best = [float(m) for m in re.findall(r"best_acc=([0-9.]+)", text)]
    per_epoch = re.findall(r"\|\s*([0-9.]+)s\s*$", text, re.M)
    return {
        "size": size,
        "best_val": max(best) if best else None,
        "minutes": elapsed / 60.0,
        "sec_per_epoch": (sum(float(x) for x in per_epoch) / len(per_epoch)) if per_epoch else None,
    }


def _env() -> Dict[str, str]:
    import os
    return dict(os.environ)


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    root = Path(args.data_root)
    for split in ("train", "val", "eval"):
        if not (root / split).is_dir():
            raise SystemExit(f"Missing {root / split}. Run download_fer2013.py first.")

    tmp = None
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="fer_calib_")
        out_root = Path(tmp.name)

    print(f"data      : {root}")
    print(f"trials    : {args.sizes} at {args.epochs} epochs each")
    print(f"output    : {out_root}")

    results = [r for s in args.sizes if (r := run_trial(base_cfg, s, args, out_root))]

    print(f"\n{'='*70}\nRESULT\n{'='*70}")
    print(f"  {'resolution':<12} {'best val acc':>14} {'s/epoch':>10} {'minutes':>9}")
    for r in results:
        bv = f"{r['best_val']:.4f}" if r["best_val"] is not None else "-"
        sp = f"{r['sec_per_epoch']:.0f}" if r["sec_per_epoch"] else "-"
        print(f"  {str(r['size'])+'x'+str(r['size']):<12} {bv:>14} {sp:>10} {r['minutes']:>9.1f}")

    ranked = [r for r in results if r["best_val"] is not None]
    if len(ranked) >= 2:
        ranked.sort(key=lambda r: r["best_val"], reverse=True)
        top, second = ranked[0], ranked[1]
        gap = top["best_val"] - second["best_val"]
        cost = top["minutes"] / second["minutes"] if second["minutes"] else float("nan")
        print(f"\n  Highest validation accuracy: {top['size']}x{top['size']} "
              f"(+{gap:.4f} over {second['size']}x{second['size']}, "
              f"at {cost:.1f}x the training time).")
        print("\n  These runs are short, so the gap is indicative rather than settled.")
        print("  Set model.image_size in config_fer.yaml to the chosen value.")

    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
