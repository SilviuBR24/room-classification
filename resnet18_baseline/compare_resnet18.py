"""
compare_resnet18.py
===================
ONE command that produces the whole ResNet-18 comparison the supervisor asked
for: train the convolutional baseline with plain cross-entropy, then with
cross-entropy plus Center Loss, evaluate both on the held-out test set, and
print a ranked summary against the ViT results.

    python compare_resnet18.py --config config_resnet18.yaml

    python compare_resnet18.py --config ... --only resnet18_crossentropy
    python compare_resnet18.py --config ... --dry-run     # plan only, no training

What it writes
--------------
Per variant, in the shared runs directory, using self-describing names:

    <YYYY-MM-DD_HH-MM>_resnet18_crossentropy/
    <YYYY-MM-DD_HH-MM>_resnet18_crossentropy_centerloss/
        checkpoints/   best_model.pt, last_checkpoint.pt
        logs/          train.log, metrics.csv, config_used.yaml
        outputs/       eval_<timestamp>/  metrics.txt, classification_report.txt,
                       confusion_matrix.png, confusion_matrix_norm.png,
                       confusion_matrix.csv, predictions.csv,
                       embeddings.npy, labels.npy

And one summary file, accumulated across runs:

    resnet18_comparison_results.csv

Robustness: the summary row is appended after each variant finishes, and a
variant whose evaluation already completed is SKIPPED on a re-run. A Colab
disconnect therefore costs only the variant in progress -- re-run the same
command and it picks up where it stopped.
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

# Published ViT results on the same split, for the comparison printed at the end.
VIT_BASELINE_TEST_ACC = 0.5633      # ViT-S/16, cross-entropy only
VIT_CENTERLOSS_TEST_ACC = 0.5500    # ViT-S/16, + Center Loss, lambda = 0.0005


# ----------------------------------------------------------------------
# The two variants. Everything except these keys comes from the config file.
# ----------------------------------------------------------------------
VARIANTS: List[Dict[str, Any]] = [
    dict(run_name="resnet18_crossentropy",
         use_center_loss=False,
         description="ResNet-18, cross-entropy only"),
    dict(run_name="resnet18_crossentropy_centerloss",
         use_center_loss=True,
         description="ResNet-18, cross-entropy + Center Loss"),
]

SUMMARY_FILENAME = "resnet18_comparison_results.csv"
SUMMARY_FIELDS = ["run_name", "description", "use_center_loss", "center_loss_weight",
                  "best_val_accuracy", "test_accuracy", "minutes", "run_dir"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet-18 CE vs CE+Center Loss comparison.")
    p.add_argument("--config", type=str, default=str(HERE / "config_resnet18.yaml"),
                   help="Base YAML config shared by both variants.")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated run_names to run (default: both).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without training.")
    p.add_argument("--force", action="store_true",
                   help="Re-run a variant even if it already has a completed evaluation.")
    return p.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def stream(cmd: List[str]) -> int:
    """Run a subprocess, echoing its output live, one line at a time.

    tqdm is disabled and the child runs unbuffered, so the log stays readable:
    one clean line per epoch instead of a progress bar rewriting a single line.
    """
    env = {**os.environ, "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env, cwd=str(HERE))
    for line in proc.stdout:
        sys.stdout.write(line.decode("utf-8", "replace"))
        sys.stdout.flush()
    return proc.wait()


def find_run_dir(runs_dir: str, run_name: str) -> Optional[str]:
    """Newest run folder for an exact run_name (folders are timestamp-prefixed).

    The trailing anchor matters: without it, 'resnet18_crossentropy' would also
    match 'resnet18_crossentropy_centerloss'.
    """
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


def read_best_val_accuracy(run_dir: str) -> Optional[float]:
    """Best val_accuracy across the per-epoch rows of metrics.csv."""
    csv_path = os.path.join(run_dir, "logs", "metrics.csv")
    if not os.path.isfile(csv_path):
        return None
    best = None
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                value = float(row["val_accuracy"])
            except (KeyError, TypeError, ValueError):
                continue
            best = value if best is None else max(best, value)
    return best


def build_variant_config(base: Dict[str, Any], variant: Dict[str, Any]) -> Path:
    """Write the per-variant config: the shared base plus this variant's toggles."""
    cfg = yaml.safe_load(yaml.safe_dump(base))          # deep copy
    cfg["run_name"] = variant["run_name"]
    cfg["training"]["use_center_loss"] = bool(variant["use_center_loss"])
    path = HERE / f"config_used_{variant['run_name']}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


def append_summary_row(summary_path: Path, record: Dict[str, Any]) -> None:
    """Append one finished variant. Written immediately, so a crash keeps history."""
    exists = summary_path.is_file()
    with open(summary_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in SUMMARY_FIELDS})


def print_summary(records: List[Dict[str, Any]]) -> None:
    """Final table: both ResNet variants next to the published ViT results."""
    done = [r for r in records if r.get("test_accuracy") is not None]

    print("\n" + "=" * 84)
    print("RESNET-18 COMPARISON -- cross-entropy vs cross-entropy + Center Loss")
    print("=" * 84)
    print(f"{'model':<44} {'val':>9} {'test':>9}")
    print("-" * 84)
    for r in done:
        val = f"{float(r['best_val_accuracy']):.4f}" if r.get("best_val_accuracy") else "n/a"
        print(f"{r['description']:<44} {val:>9} {float(r['test_accuracy']):>9.4f}")
    print("-" * 84)
    print(f"{'ViT-S/16, cross-entropy only (reference)':<44} {'0.5683':>9} "
          f"{VIT_BASELINE_TEST_ACC:>9.4f}")
    print(f"{'ViT-S/16, + Center Loss (reference)':<44} {'0.5628':>9} "
          f"{VIT_CENTERLOSS_TEST_ACC:>9.4f}")
    print("=" * 84)

    by_name = {r["run_name"]: r for r in done}
    ce = by_name.get("resnet18_crossentropy")
    cl = by_name.get("resnet18_crossentropy_centerloss")
    if ce and cl:
        delta = float(cl["test_accuracy"]) - float(ce["test_accuracy"])
        verdict = "HELPS" if delta > 0 else ("no effect" if delta == 0 else "HURTS")
        print(f"\nOn ResNet-18, Center Loss {verdict} "
              f"({float(ce['test_accuracy']):.4f} -> {float(cl['test_accuracy']):.4f}, "
              f"{delta:+.4f}).")
        print("On ViT-S/16 the same change was "
              f"{VIT_CENTERLOSS_TEST_ACC - VIT_BASELINE_TEST_ACC:+.4f}.")
        print("\nIf both architectures move in the same direction, the finding is a "
              "property of the task rather than of the model. Read this together "
              "with the MNIST check, which shows the implementation itself is sound.")
    if ce:
        gap = float(ce["test_accuracy"]) - VIT_BASELINE_TEST_ACC
        print(f"\nArchitecture comparison, cross-entropy only: "
              f"ResNet-18 {float(ce['test_accuracy']):.4f} vs ViT-S/16 "
              f"{VIT_BASELINE_TEST_ACC:.4f} ({gap:+.4f}).")

    print("\nReminder: the test set holds 600 images, so one percentage point is six "
          "images and the standard error of a proportion is about two points. "
          "Differences of this size are indicative, not conclusive.")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    runs_dir = base["paths"]["output_root"]
    test_dir = base["data"]["eval_dir"]
    summary_path = Path(runs_dir) / SUMMARY_FILENAME

    variants = VARIANTS
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        variants = [v for v in VARIANTS if v["run_name"] in wanted]
        unknown = wanted - {v["run_name"] for v in VARIANTS}
        if unknown:
            raise SystemExit(f"Unknown run_name(s): {sorted(unknown)}. "
                             f"Available: {[v['run_name'] for v in VARIANTS]}")

    print("=" * 84)
    print(f"RESNET-18 COMPARISON -- {len(variants)} variant(s)")
    print("=" * 84)
    for v in variants:
        centre = (f"Center Loss ON (lambda="
                  f"{base['training']['center_loss_weight']})") if v["use_center_loss"] \
                 else "Center Loss OFF"
        print(f"  {v['run_name']:<38} {centre}")
    print(f"\nEach variant: {base['training']['epochs']} epochs, batch "
          f"{base['training']['batch_size']}, then a test evaluation with embeddings.")
    print("All other settings are identical to the ViT runs, so the architecture is "
          "the only thing that differs.")
    print(f"Summary accumulates in: {summary_path}")
    print("Re-running the same command skips finished variants, so a disconnect is safe.")

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    records: List[Dict[str, Any]] = []
    for i, variant in enumerate(variants, 1):
        run_name = variant["run_name"]
        existing = find_run_dir(runs_dir, run_name)
        if existing and not args.force:
            acc = read_test_accuracy(existing)
            if acc is not None:
                print(f"\n[{i}/{len(variants)}] {run_name}: already complete "
                      f"(test={acc:.4f}) -- skipping. Use --force to redo.")
                records.append(dict(variant, run_dir=existing, test_accuracy=acc,
                                    best_val_accuracy=read_best_val_accuracy(existing),
                                    center_loss_weight=base["training"]["center_loss_weight"]))
                continue

        print("\n" + "=" * 84)
        print(f"[{i}/{len(variants)}] TRAIN {run_name}  --  {variant['description']}")
        print("=" * 84, flush=True)

        cfg_path = build_variant_config(base, variant)
        t0 = time.time()
        rc = stream([sys.executable, "-u", str(HERE / "train_resnet18.py"),
                     "--config", str(cfg_path)])
        if rc != 0:
            print(f"!! training FAILED for {run_name} (exit {rc}) -- moving on.")
            continue

        run_dir = find_run_dir(runs_dir, run_name)
        if run_dir is None:
            print(f"!! no run folder found for {run_name} -- cannot evaluate.")
            continue

        print(f"\n--- evaluating {run_name} on the TEST set ---", flush=True)
        rc = stream([sys.executable, "-u", str(HERE / "evaluate_resnet18.py"),
                     "--checkpoint", os.path.join(run_dir, "checkpoints", "best_model.pt"),
                     "--data-dir", test_dir, "--save-embeddings"])
        if rc != 0:
            print(f"!! evaluation FAILED for {run_name} (exit {rc}).")
            continue

        record = dict(
            variant,
            run_dir=run_dir,
            center_loss_weight=base["training"]["center_loss_weight"],
            best_val_accuracy=read_best_val_accuracy(run_dir),
            test_accuracy=read_test_accuracy(run_dir),
            minutes=round((time.time() - t0) / 60.0, 1),
        )
        records.append(record)
        append_summary_row(summary_path, record)

        acc = record["test_accuracy"]
        if acc is not None:
            print(f"\n>>> {run_name}: test={acc:.4f} in {record['minutes']:.0f} min")

    print_summary(records)


if __name__ == "__main__":
    main()
