"""
compare_wen.py
==============
Runs both variants of the update-rule comparison with one command, evaluates
each on the held-out test set, and prints a summary.

    python compare_wen.py --config config_wen.yaml
    python compare_wen.py --config ... --dry-run
    python compare_wen.py --config ... --only fer_wen_gradient
    python compare_wen.py --config ... --force

Order matters here. The "gradient" variant runs first because it is the
control: it must reproduce the result the shared training loop already produced
in `fer_baseline/`. If it does not, this folder's loop differs in some other
way and the "wen" result cannot yet be attributed to the update rule. The
summary states that comparison explicitly rather than leaving it to the reader.

Interrupting is safe: a summary row is appended after each variant finishes,
and re-running skips variants whose evaluation already completed.
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

# The result produced by the shared training loop in fer_baseline/, which the
# control variant is expected to reproduce.
REFERENCE = {"run": "fer_crossentropy_centerloss", "test_accuracy": 0.6807,
             "source": "fer_baseline/, shared training loop"}

VARIANTS: List[Dict[str, Any]] = [
    {"run_name": "fer_wen_gradient", "center_mode": "gradient",
     "description": "CONTROL: centres as parameters, SGD on the combined objective"},
    {"run_name": "fer_wen_algorithm1", "center_mode": "wen",
     "description": "Algorithm 1 of Wen et al., per-class normalisation by (1 + n_j)"},
]

SUMMARY_FILENAME = "fer_wen_comparison_results.csv"
SUMMARY_FIELDS = ["run_name", "center_mode", "description", "center_loss_weight",
                  "center_loss_lr", "image_size", "best_val_accuracy",
                  "test_accuracy", "macro_f1", "minutes", "run_dir"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the update-rule comparison.")
    p.add_argument("--config", default=str(HERE / "config_wen.yaml"))
    p.add_argument("--only", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def stream(cmd: List[str]) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
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


def read_text(path: str) -> str:
    return open(path, encoding="utf-8", errors="replace").read() if os.path.isfile(path) else ""


def read_test_accuracy(run_dir: str) -> Optional[float]:
    ev = newest_eval(run_dir)
    if not ev:
        return None
    hits = re.findall(r"Overall accuracy:\s*([0-9.]+)", read_text(os.path.join(ev, "metrics.txt")))
    return float(hits[-1]) if hits else None


def read_macro_f1(run_dir: str) -> Optional[float]:
    ev = newest_eval(run_dir)
    if not ev:
        return None
    m = re.search(r"macro avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)",
                  read_text(os.path.join(ev, "classification_report.txt")))
    return float(m.group(1)) if m else None


def read_best_val(run_dir: str) -> Optional[float]:
    hits = re.findall(r"best_acc=([0-9.]+)",
                      read_text(os.path.join(run_dir, "logs", "train.log")))
    return max(float(h) for h in hits) if hits else None


def build_variant_config(base: Dict[str, Any], variant: Dict[str, Any]) -> Path:
    cfg = copy.deepcopy(base)
    cfg["run_name"] = variant["run_name"]
    cfg["training"]["center_mode"] = variant["center_mode"]
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
    def num(r, key, nd=4):
        v = r.get(key)
        return f"{float(v):.{nd}f}" if v not in (None, "") else "-"

    print("\n" + "=" * 84)
    print("CENTER LOSS UPDATE RULE -- FER2013")
    print("=" * 84)
    print(f"{'variant':<24}{'mode':<11}{'val':>9}{'test':>9}{'macro-F1':>11}{'min':>8}")
    print("-" * 84)
    for r in records:
        print(f"{r['run_name']:<24}{r['center_mode']:<11}"
              f"{num(r, 'best_val_accuracy'):>9}{num(r, 'test_accuracy'):>9}"
              f"{num(r, 'macro_f1', 3):>11}{float(r.get('minutes') or 0):>8.1f}")
    print("=" * 84)

    by_mode = {r["center_mode"]: r for r in records}

    control = by_mode.get("gradient")
    if control and control.get("test_accuracy") not in (None, ""):
        got = float(control["test_accuracy"])
        ref = REFERENCE["test_accuracy"]
        delta = got - ref
        print(f"\nCONTROL CHECK")
        print(f"  This loop, gradient mode : {got:.4f}")
        print(f"  Shared loop, same setting: {ref:.4f}   ({REFERENCE['source']})")
        print(f"  Difference               : {delta:+.4f}")
        if abs(delta) <= 0.01:
            print("  Within one percentage point: the two loops agree, so a difference")
            print("  in the other variant can be attributed to the update rule.")
        else:
            print("  MORE THAN ONE POINT APART. This loop differs from the shared one in")
            print("  some way beyond the update rule. Do not attribute the other")
            print("  variant's result to Algorithm 1 until this is explained.")

    if "gradient" in by_mode and "wen" in by_mode:
        g, w = by_mode["gradient"], by_mode["wen"]
        if g.get("test_accuracy") not in (None, "") and w.get("test_accuracy") not in (None, ""):
            d_acc = float(w["test_accuracy"]) - float(g["test_accuracy"])
            print(f"\nALGORITHM 1 versus GRADIENT VARIANT")
            print(f"  test accuracy : {d_acc:+.4f}")
            if g.get("macro_f1") not in (None, "") and w.get("macro_f1") not in (None, ""):
                print(f"  macro-F1      : {float(w['macro_f1']) - float(g['macro_f1']):+.4f}")
            print("\n  At 3,589 test images the standard error near 68 percent accuracy is")
            print("  about 0.008, so a difference below that is not resolved by this run.")
            print("  The per-class centre alignment in each run's centre_alignment.csv is")
            print("  the more direct measurement of what the update rule changed.")


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as fh:
        base = yaml.safe_load(fh)

    runs_dir = base["paths"]["output_root"]
    os.makedirs(runs_dir, exist_ok=True)
    summary_path = Path(runs_dir) / SUMMARY_FILENAME

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
    print(f"resolution  : {base['model']['image_size']}")
    print(f"epochs      : {base['training']['epochs']}, batch {base['training']['batch_size']}")
    print(f"lambda      : {base['training']['center_loss_weight']}")
    print(f"centre rate : {base['training']['center_loss_lr']}")
    print(f"variants    : {[(v['run_name'], v['center_mode']) for v in variants]}")
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
                "run_name": name, "center_mode": v["center_mode"],
                "description": v["description"],
                "center_loss_weight": base["training"]["center_loss_weight"],
                "center_loss_lr": base["training"]["center_loss_lr"],
                "image_size": base["model"]["image_size"],
                "best_val_accuracy": read_best_val(existing),
                "test_accuracy": read_test_accuracy(existing),
                "macro_f1": read_macro_f1(existing),
                "minutes": "", "run_dir": existing})
            continue

        print(f"\n{'=' * 84}\nTRAIN  {name}  (mode: {v['center_mode']})\n{'=' * 84}", flush=True)
        cfg_path = build_variant_config(base, v)
        t0 = time.time()
        if stream([sys.executable, "-u", str(HERE / "train_wen.py"),
                   "--config", str(cfg_path)]) != 0:
            print(f"[error] training failed for {name}; stopping.")
            return

        run_dir = find_run_dir(runs_dir, name)
        if not run_dir:
            print(f"[error] no run directory found for {name}; stopping.")
            return

        print(f"\n{'=' * 84}\nEVALUATE  {name}\n{'=' * 84}", flush=True)
        if stream([sys.executable, "-u", str(HERE / "evaluate_wen.py"),
                   "--checkpoint", os.path.join(run_dir, "checkpoints", "best_model.pt"),
                   "--data-dir", base["data"]["eval_dir"]]) != 0:
            print(f"[error] evaluation failed for {name}; stopping.")
            return

        record = {
            "run_name": name, "center_mode": v["center_mode"],
            "description": v["description"],
            "center_loss_weight": base["training"]["center_loss_weight"],
            "center_loss_lr": base["training"]["center_loss_lr"],
            "image_size": base["model"]["image_size"],
            "best_val_accuracy": read_best_val(run_dir),
            "test_accuracy": read_test_accuracy(run_dir),
            "macro_f1": read_macro_f1(run_dir),
            "minutes": round((time.time() - t0) / 60.0, 1),
            "run_dir": run_dir}
        append_summary(summary_path, record)
        records.append(record)

    print_summary(records)
    print(f"\nSummary appended to {summary_path}")


if __name__ == "__main__":
    main()
