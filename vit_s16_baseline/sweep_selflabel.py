"""
sweep_selflabel.py
==================
Run a whole self-labelling sweep in one go and report which setting (if any)
beats the supervised baseline.

Motivation
----------
The first self-labelling runs used a FIXED confidence threshold, which made
coverage an *outcome*: the model rarely cleared tau=0.95, so the unlabelled pool
went almost unused (0.08% of it in the few-labels run). Two knobs plausibly
decide whether self-labelling can beat the baseline:

  1. WHERE IT STARTS  -- warm-starting from the center-loss model (55.00%) is a
     handicap when the plain baseline (56.33%) is the number to beat.
  2. HOW MUCH OF THE POOL IT USES -- `target_coverage` forces a fixed share of
     the pool to be pseudo-labelled each round, tracing the precision/coverage
     trade-off instead of leaving it to chance.

This script sweeps both, evaluates every run on the held-out test set, and
prints a ranked summary against the baseline.

Usage
-----
    python sweep_selflabel.py --config config_sweep_base.yaml

    # only some rows, e.g. while iterating:
    python sweep_selflabel.py --config ... --only frombase_cov25,frombase_cov50

    # see the plan (and the time estimate) without training anything:
    python sweep_selflabel.py --config ... --dry-run

The base config supplies data paths / training params / an `ssl:` block; each
sweep row overrides `run_name`, the warm-start checkpoint and the selection
knobs on top of it.

Robustness: results are appended to `sweep_results.csv` in the output root
after every run, and a row whose run folder already exists is SKIPPED. So if
Colab disconnects, just re-run the same command and it picks up where it left
off.
"""
from __future__ import annotations

import argparse
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

# Reference numbers from the clean protocol (val-selected, test-reported).
BASELINE_TEST_ACC = 0.5633   # plain supervised ViT-S/16, 3200 labels/class
CENTER_TEST_ACC = 0.5500     # + center loss, lambda=0.0005


# ----------------------------------------------------------------------
# The sweep grid
# ----------------------------------------------------------------------
# `warmup`: which finished run to start from -- 'baseline' | 'center' | None.
# `coverage`: fraction of the pool to pseudo-label each round (None -> fixed tau).
SWEEP: List[Dict[str, Any]] = [
    # --- start from the BASELINE model: the most direct shot at beating 56.33% ---
    dict(tag="frombase_tau095", warmup="baseline", coverage=None, tau=0.95),
    dict(tag="frombase_cov25", warmup="baseline", coverage=0.25),
    dict(tag="frombase_cov50", warmup="baseline", coverage=0.50),
    # --- start from the CENTER model: the spec-compliant variant ---------------
    dict(tag="fromcenter_cov25", warmup="center", coverage=0.25),
    dict(tag="fromcenter_cov50", warmup="center", coverage=0.50),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-labelling sweep runner.")
    p.add_argument("--config", type=str, required=True,
                   help="Base YAML config (data paths, training params, ssl block).")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated tags to run (default: all).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without training.")
    p.add_argument("--force", action="store_true",
                   help="Re-run rows even if a matching run folder already exists.")
    return p.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def stream(cmd: List[str]) -> int:
    """Run a subprocess, echoing its output live (one line at a time)."""
    env = {**os.environ, "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    for line in proc.stdout:
        sys.stdout.write(line.decode("utf-8", "replace"))
        sys.stdout.flush()
    return proc.wait()


def find_checkpoint(runs_dir: str, pattern: str) -> Optional[str]:
    """Newest best_model.pt whose run folder matches `pattern`."""
    hits = sorted(glob.glob(os.path.join(runs_dir, pattern, "checkpoints", "best_model.pt")))
    return hits[-1] if hits else None


def find_run_dir(runs_dir: str, run_name: str) -> Optional[str]:
    """Newest run folder for an exact run_name (folders are timestamp-prefixed)."""
    hits = sorted(glob.glob(os.path.join(runs_dir, f"*_{run_name}")))
    return hits[-1] if hits else None


def read_test_accuracy(run_dir: str) -> Optional[float]:
    """Overall accuracy from the newest eval_*/metrics.txt inside a run folder."""
    evals = sorted(glob.glob(os.path.join(run_dir, "outputs", "eval_*", "metrics.txt")))
    if not evals:
        return None
    text = Path(evals[-1]).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Overall accuracy:\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


def read_best_val(run_dir: str) -> Optional[float]:
    """Best val_accuracy across the self-training CSV rows."""
    csv_path = os.path.join(run_dir, "logs", "selftrain_metrics.csv")
    if not os.path.isfile(csv_path):
        return None
    best = None
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                v = float(row["val_accuracy"])
            except (KeyError, TypeError, ValueError):
                continue
            best = v if best is None else max(best, v)
    return best


def build_config(base: Dict[str, Any], row: Dict[str, Any], run_name: str) -> Path:
    """Write the per-row config: base + this row's selection knobs."""
    cfg = yaml.safe_load(yaml.safe_dump(base))  # deep copy
    cfg["run_name"] = run_name
    ssl = cfg.setdefault("ssl", {})
    ssl["target_coverage"] = row.get("coverage")
    if row.get("tau") is not None:
        ssl["tau"] = row["tau"]
    path = HERE / f"config_sweep_{row['tag']}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


def append_result(results_csv: Path, record: Dict[str, Any]) -> None:
    """Append one finished row (written after every run, so a crash keeps history)."""
    fields = ["tag", "warmup", "coverage", "tau", "run_name",
              "best_val_acc", "test_acc", "delta_vs_baseline", "minutes"]
    exists = results_csv.is_file()
    with open(results_csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: record.get(k, "") for k in fields})


def _val_key(record: Dict[str, Any]) -> float:
    """Sort key: validation accuracy, the only legitimate selection criterion.

    Runs without a recorded validation accuracy sort last rather than crashing.
    """
    value = record.get("best_val_acc")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def print_summary(records: List[Dict[str, Any]]) -> None:
    """Ranked table of every finished run against the baseline.

    Ranked by VALIDATION accuracy, deliberately. Test accuracy is reported in
    its own column but must never drive the choice of configuration: picking the
    setting with the best test score is selection-on-test, the very mistake that
    invalidated the first phase of this project. The ordering here enforces the
    correct discipline instead of leaving it to the reader.
    """
    print("\n" + "=" * 78)
    print("SWEEP SUMMARY -- ranked by VALIDATION accuracy (the selection criterion)")
    print("=" * 78)
    print(f"{'tag':<20} {'warm-start':<10} {'coverage':<10} "
          f"{'val':<8} {'test':<8} {'vs baseline':<12}")
    print("-" * 78)

    done = [r for r in records if r.get("test_acc") is not None]
    for r in sorted(done, key=lambda x: -_val_key(x)):
        cov = f"{float(r['coverage']):.0%}" if r.get("coverage") else f"tau={r.get('tau', 0.95)}"
        delta = float(r["test_acc"]) - BASELINE_TEST_ACC
        mark = "  <-- BEATS IT" if delta > 0 else ""
        val = f"{float(r['best_val_acc']):.4f}" if r.get("best_val_acc") else "n/a"
        print(f"{r['tag']:<20} {str(r['warmup']):<10} {cov:<10} "
              f"{val:<8} {float(r['test_acc']):.4f}   {delta:+.4f}{mark}")

    print("-" * 78)
    print(f"{'BASELINE (reference)':<20} {'--':<10} {'--':<10} "
          f"{'0.5683':<8} {BASELINE_TEST_ACC:.4f}")
    print(f"{'center l=0.0005':<20} {'--':<10} {'--':<10} "
          f"{'0.5628':<8} {CENTER_TEST_ACC:.4f}   {CENTER_TEST_ACC - BASELINE_TEST_ACC:+.4f}")
    print("=" * 78)

    # The winner is chosen on VALIDATION, then its test score is reported --
    # never the other way round.
    if done:
        best = max(done, key=_val_key)
        best_test = float(best["test_acc"])
        delta = best_test - BASELINE_TEST_ACC
        print(f"\nSelected on validation: {best['tag']} "
              f"(val {_val_key(best):.4f}).")
        print(f"Its test accuracy is {best_test:.4f} "
              f"({delta:+.4f} versus the baseline).")
        if delta > 0:
            print("It beats the baseline on test as well, and validation agrees in "
                  "direction -- the strongest evidence available here.")
        else:
            print("It does not beat the baseline on test. That is a legitimate "
                  "result and must be reported as such; do not go looking for a "
                  "different row with a better test score.")
        print("Keep the magnitude in perspective: 600 test images means one "
              "percentage point is six images, and the standard error of a "
              "proportion at this accuracy is about two points.")
    else:
        print("\nNo configuration completed, so there is nothing to rank.")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    runs_dir = base["paths"]["output_root"]
    test_dir = base["data"]["eval_dir"]
    results_csv = Path(runs_dir) / "sweep_results.csv"

    rows = SWEEP
    if args.only:
        wanted = {t.strip() for t in args.only.split(",")}
        rows = [r for r in rows if r["tag"] in wanted]
        missing = wanted - {r["tag"] for r in rows}
        if missing:
            raise SystemExit(f"Unknown tag(s): {sorted(missing)}. "
                             f"Available: {[r['tag'] for r in SWEEP]}")

    # Resolve warm-start checkpoints up front, so a missing one fails fast
    # instead of halfway through a multi-hour sweep.
    sources = {
        "baseline": "*vit_s16_3200_baseline",
        "center": "*vit_s16_3200_center_l00005",
    }
    resolved: Dict[str, Optional[str]] = {}
    for key, pattern in sources.items():
        if any(r.get("warmup") == key for r in rows):
            ckpt = find_checkpoint(runs_dir, pattern)
            if ckpt is None:
                raise SystemExit(
                    f"Warm-start checkpoint for '{key}' not found under {runs_dir} "
                    f"(pattern {pattern}). Run the baseline notebook first."
                )
            resolved[key] = ckpt

    print("=" * 78)
    print(f"SELF-LABELLING SWEEP -- {len(rows)} configuration(s)")
    print(f"Baseline to beat: {BASELINE_TEST_ACC:.4f} on the test set")
    print("=" * 78)
    for r in rows:
        cov = f"{r['coverage']:.0%} of pool" if r.get("coverage") else f"fixed tau={r.get('tau', 0.95)}"
        print(f"  {r['tag']:<20} warm-start={str(r['warmup']):<9} selection={cov}")
    ssl_cfg = base.get("ssl", {})
    epochs = int(ssl_cfg.get("rounds", 6)) * int(ssl_cfg.get("epochs_per_round", 5))
    print(f"\nEach run: {ssl_cfg.get('rounds')} rounds x {ssl_cfg.get('epochs_per_round')} "
          f"epochs = {epochs} epochs, then a test evaluation.")
    print(f"Results accumulate in: {results_csv}")
    print("Re-running the same command skips finished rows, so a disconnect is safe.")

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    records: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        run_name = f"vit_s16_sweep_{row['tag']}"
        existing = find_run_dir(runs_dir, run_name)
        if existing and not args.force:
            acc = read_test_accuracy(existing)
            if acc is not None:
                print(f"\n[{i}/{len(rows)}] {row['tag']}: already done "
                      f"(test={acc:.4f}) -- skipping. Use --force to redo.")
                records.append(dict(row, run_name=run_name, test_acc=acc,
                                    best_val_acc=read_best_val(existing)))
                continue

        print("\n" + "=" * 78)
        print(f"[{i}/{len(rows)}] RUN {row['tag']}  ->  {run_name}")
        print("=" * 78, flush=True)

        cfg_path = build_config(base, row, run_name)
        # sys.executable, not "python": the child must use the SAME interpreter as
        # this script, or it can pick up a different environment (e.g. one without torch).
        cmd = [sys.executable, "-u", str(HERE / "self_train.py"), "--config", str(cfg_path)]
        if row.get("warmup"):
            cmd += ["--warmup", resolved[row["warmup"]]]

        t0 = time.time()
        rc = stream(cmd)
        if rc != 0:
            print(f"!! training FAILED for {row['tag']} (exit {rc}) -- skipping to next.")
            continue

        run_dir = find_run_dir(runs_dir, run_name)
        if run_dir is None:
            print(f"!! no run folder found for {run_name} -- skipping evaluation.")
            continue

        print(f"\n--- evaluating {row['tag']} on the TEST set ---", flush=True)
        rc = stream([sys.executable, "-u", str(HERE / "evaluate.py"),
                     "--checkpoint", os.path.join(run_dir, "checkpoints", "best_model.pt"),
                     "--data-dir", test_dir, "--save-embeddings"])
        if rc != 0:
            print(f"!! evaluation FAILED for {row['tag']} (exit {rc}).")
            continue

        minutes = (time.time() - t0) / 60.0
        record = dict(
            row,
            run_name=run_name,
            best_val_acc=read_best_val(run_dir),
            test_acc=read_test_accuracy(run_dir),
            minutes=round(minutes, 1),
        )
        if record["test_acc"] is not None:
            record["delta_vs_baseline"] = round(record["test_acc"] - BASELINE_TEST_ACC, 4)
        records.append(record)
        append_result(results_csv, record)

        acc = record["test_acc"]
        print(f"\n>>> {row['tag']}: test={acc:.4f} "
              f"({acc - BASELINE_TEST_ACC:+.4f} vs baseline) in {minutes:.0f} min")

    print_summary(records)


if __name__ == "__main__":
    main()
