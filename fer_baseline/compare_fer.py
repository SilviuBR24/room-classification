"""
compare_fer.py
==============
Runs the whole facial-expression comparison with one command: Cross-Entropy and
Cross-Entropy plus Center Loss, each trained and then evaluated on the held-out
test set, followed by a summary table.

    python compare_fer.py --config config_fer.yaml
    python compare_fer.py --config ... --dry-run                 # plan only
    python compare_fer.py --config ... --only fer_crossentropy   # one variant
    python compare_fer.py --config ... --force                   # redo a finished variant

Interrupting is safe. A summary row is appended after each variant finishes and
re-running skips variants whose evaluation already completed, so a disconnected
Colab session costs only the variant in progress.

Everything except `run_name` and `use_center_loss` comes from the config file
unchanged, which is what keeps the two variants comparable.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

HERE = Path(__file__).resolve().parent

VARIANTS: List[Dict[str, Any]] = [
    {"run_name": "fer_crossentropy",
     "description": "ResNet-18 on FER2013, cross-entropy only",
     "use_center_loss": False},
    {"run_name": "fer_crossentropy_centerloss",
     "description": "ResNet-18 on FER2013, cross-entropy + Center Loss",
     "use_center_loss": True},
]

SUMMARY_FILENAME = "fer_comparison_results.csv"
SUMMARY_FIELDS = ["run_name", "description", "use_center_loss", "center_loss_weight",
                  "image_size", "best_val_accuracy", "test_accuracy", "macro_f1",
                  "minutes", "run_dir"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full FER2013 comparison.")
    p.add_argument("--config", default=str(HERE / "config_fer.yaml"))
    p.add_argument("--only", default=None,
                   help="comma-separated run_name values to run")
    p.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    p.add_argument("--force", action="store_true",
                   help="re-run variants that already have a completed evaluation")
    return p.parse_args()


def stream(cmd: List[str]) -> int:
    """Run a command, echoing its output line by line."""
    env = {**os.environ, "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=str(HERE), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for raw in proc.stdout:
        sys.stdout.write(raw.decode("utf-8", "replace"))
        sys.stdout.flush()
    return proc.wait()


def find_run_dir(runs_dir: str, run_name: str) -> Optional[str]:
    hits = sorted(d for d in glob.glob(os.path.join(runs_dir, f"*_{run_name}"))
                  if os.path.isdir(d))
    return hits[-1] if hits else None


def newest_eval(run_dir: str) -> Optional[str]:
    hits = sorted(glob.glob(os.path.join(run_dir, "outputs", "eval_*")))
    return hits[-1] if hits else None


def read_metric(run_dir: str, pattern: str, source: str) -> Optional[str]:
    path = (newest_eval(run_dir) or "") if source == "eval" else os.path.join(run_dir, "logs")
    if source == "eval":
        path = os.path.join(path, "metrics.txt") if path else ""
    else:
        path = os.path.join(path, "train.log")
    if not path or not os.path.isfile(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read()
    hits = re.findall(pattern, text)
    return hits[-1] if hits else None


def read_best_val(run_dir: str) -> Optional[float]:
    log = os.path.join(run_dir, "logs", "train.log")
    if not os.path.isfile(log):
        return None
    hits = re.findall(r"best_acc=([0-9.]+)",
                      open(log, encoding="utf-8", errors="replace").read())
    return max(float(h) for h in hits) if hits else None


def read_macro_f1(run_dir: str) -> Optional[float]:
    ev = newest_eval(run_dir)
    if not ev:
        return None
    path = os.path.join(ev, "classification_report.txt")
    if not os.path.isfile(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"macro avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)", text)
    return float(m.group(1)) if m else None


def build_variant_config(base: Dict[str, Any], variant: Dict[str, Any]) -> Path:
    cfg = copy.deepcopy(base)
    cfg["run_name"] = variant["run_name"]
    cfg["training"]["use_center_loss"] = bool(variant["use_center_loss"])
    out = HERE / f"config_used_{variant['run_name']}.yaml"
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return out


def append_summary(path: Path, record: Dict[str, Any]) -> None:
    exists = path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: record.get(k, "") for k in SUMMARY_FIELDS})


def print_summary(records: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("FER2013 COMPARISON")
    print("=" * 78)
    print(f"{'variant':<34}{'val':>9}{'test':>9}{'macro-F1':>10}{'min':>8}")
    print("-" * 78)
    for r in records:
        def fmt(key, nd=4):
            v = r.get(key)
            return f"{float(v):.{nd}f}" if v not in (None, "") else "-"
        print(f"{r['run_name']:<34}{fmt('best_val_accuracy'):>9}"
              f"{fmt('test_accuracy'):>9}{fmt('macro_f1', 3):>10}"
              f"{float(r.get('minutes') or 0):>8.1f}")
    print("=" * 78)
    print("FER2013 classes are imbalanced (about 16:1 between the most and least")
    print("frequent). Read macro-F1 alongside accuracy: a model that ignores the")
    print("rarest expression loses little accuracy but a lot of macro-F1.")


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as fh:
        base = yaml.safe_load(fh)

    runs_dir = base["paths"]["output_root"]
    os.makedirs(runs_dir, exist_ok=True)
    summary_path = Path(runs_dir) / SUMMARY_FILENAME
    image_size = base["model"]["image_size"]

    variants = VARIANTS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        unknown = wanted - {v["run_name"] for v in VARIANTS}
        if unknown:
            raise SystemExit(f"Unknown variant(s): {sorted(unknown)}. "
                             f"Available: {[v['run_name'] for v in VARIANTS]}")
        variants = [v for v in VARIANTS if v["run_name"] in wanted]

    print(f"config      : {args.config}")
    print(f"runs dir    : {runs_dir}")
    print(f"resolution  : {image_size}x{image_size}")
    print(f"epochs      : {base['training']['epochs']}, batch {base['training']['batch_size']}")
    print(f"lambda      : {base['training']['center_loss_weight']}")
    print(f"variants    : {[v['run_name'] for v in variants]}")
    if args.dry_run:
        print("\n--dry-run: nothing was executed.")
        return

    records: List[Dict[str, Any]] = []
    for v in variants:
        name = v["run_name"]
        existing = find_run_dir(runs_dir, name)
        if existing and newest_eval(existing) and not args.force:
            print(f"\n[skip] {name}: already evaluated at {existing}")
            records.append({
                "run_name": name, "description": v["description"],
                "use_center_loss": v["use_center_loss"],
                "center_loss_weight": base["training"]["center_loss_weight"],
                "image_size": image_size,
                "best_val_accuracy": read_best_val(existing),
                "test_accuracy": read_metric(existing, r"Overall accuracy:\s*([0-9.]+)", "eval"),
                "macro_f1": read_macro_f1(existing),
                "minutes": "", "run_dir": existing,
            })
            continue

        print(f"\n{'='*78}\nTRAIN  {name}\n{'='*78}", flush=True)
        cfg_path = build_variant_config(base, v)
        t0 = time.time()
        if stream([sys.executable, "-u", str(HERE / "train_fer.py"),
                   "--config", str(cfg_path)]) != 0:
            print(f"[error] training failed for {name}; stopping.")
            return

        run_dir = find_run_dir(runs_dir, name)
        if not run_dir:
            print(f"[error] no run directory found for {name}; stopping.")
            return

        print(f"\n{'='*78}\nEVALUATE  {name}\n{'='*78}", flush=True)
        if stream([sys.executable, "-u", str(HERE / "evaluate_fer.py"),
                   "--checkpoint", os.path.join(run_dir, "checkpoints", "best_model.pt"),
                   "--data-dir", base["data"]["eval_dir"],
                   "--save-embeddings"]) != 0:
            print(f"[error] evaluation failed for {name}; stopping.")
            return

        record = {
            "run_name": name, "description": v["description"],
            "use_center_loss": v["use_center_loss"],
            "center_loss_weight": base["training"]["center_loss_weight"],
            "image_size": image_size,
            "best_val_accuracy": read_best_val(run_dir),
            "test_accuracy": read_metric(run_dir, r"Overall accuracy:\s*([0-9.]+)", "eval"),
            "macro_f1": read_macro_f1(run_dir),
            "minutes": round((time.time() - t0) / 60.0, 1),
            "run_dir": run_dir,
        }
        append_summary(summary_path, record)
        records.append(record)

    print_summary(records)
    print(f"\nSummary appended to {summary_path}")


if __name__ == "__main__":
    main()
