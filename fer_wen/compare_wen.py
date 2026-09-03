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

# The control variant must reproduce the result the shared training loop
# produced under the same settings. That number is not hard-coded: it is read
# from the summary the other experiment wrote, so it can never go stale.
#
# It was hard-coded once, as 0.6807, and that value came from a run on the
# original FER2013 partitioning -- which is now known to share 8.75% of its test
# images with training. Comparing against it after moving to the deduplicated
# split would have silently compared two different datasets.
REFERENCE_SUMMARY = "fer_clean_comparison_results.csv"
REFERENCE_RUN = "fer_clean_crossentropy_centerloss"

VARIANTS: List[Dict[str, Any]] = [
    {"run_name": "fer_wen_clean_gradient", "center_mode": "gradient",
     "description": "CONTROL: centres as parameters, SGD on the combined objective"},
    {"run_name": "fer_wen_clean_algorithm1", "center_mode": "wen",
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


def run_matches(run_dir: str, base: Dict[str, Any],
                variant: Dict[str, Any]) -> tuple[bool, str]:
    """Is an existing run directory safe to reuse for this variant?

    Checks the settings that would change the numbers, plus the presence of the
    artefacts a completed evaluation must have written. Anything else -- paths,
    worker counts -- does not affect the result and is ignored.
    """
    cfg_path = os.path.join(run_dir, "logs", "config_used.yaml")
    if not os.path.isfile(cfg_path):
        return False, "it has no logs/config_used.yaml, so its settings are unknown"
    with open(cfg_path, encoding="utf-8") as fh:
        old = yaml.safe_load(fh)

    checks = [
        ("training.center_mode", old["training"].get("center_mode"), variant["center_mode"]),
        ("training.center_loss_weight", old["training"].get("center_loss_weight"),
         base["training"]["center_loss_weight"]),
        ("training.center_loss_lr", old["training"].get("center_loss_lr"),
         base["training"]["center_loss_lr"]),
        ("training.epochs", old["training"].get("epochs"), base["training"]["epochs"]),
        ("training.batch_size", old["training"].get("batch_size"), base["training"]["batch_size"]),
        ("training.learning_rate", old["training"].get("learning_rate"),
         base["training"]["learning_rate"]),
        ("training.seed", old["training"].get("seed"), base["training"]["seed"]),
        ("model.image_size", old["model"].get("image_size"), base["model"]["image_size"]),
        ("model.num_classes", old["model"].get("num_classes"), base["model"]["num_classes"]),
    ]
    for key, was, now in checks:
        if was != now:
            return False, f"{key} was {was!r} in that run but is {now!r} now"

    ev = newest_eval(run_dir)
    required = ["metrics.txt", "classification_report.txt", "confusion_matrix.csv"]
    missing = [f for f in required if not os.path.isfile(os.path.join(ev, f))]
    if missing:
        return False, f"its evaluation is incomplete, missing {missing}"
    if read_test_accuracy(run_dir) is None:
        return False, "no overall accuracy could be read from its metrics.txt"
    return True, ""


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


def read_reference(runs_dir: str) -> Optional[Dict[str, Any]]:
    """The result the shared training loop produced under the same settings.

    Read from the other experiment's summary rather than hard-coded, so it
    cannot drift out of date when the dataset or the settings change.
    """
    path = os.path.join(runs_dir, REFERENCE_SUMMARY)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("run_name") == REFERENCE_RUN and row.get("test_accuracy"):
                return {"test_accuracy": float(row["test_accuracy"]),
                        "n_test": None, "source": f"{REFERENCE_SUMMARY} / {REFERENCE_RUN}"}
    return None


def print_summary(records: List[Dict[str, Any]], reference: Optional[Dict[str, Any]],
                  n_test: Optional[int]) -> None:
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
    if control and control.get("test_accuracy") not in (None, "") and reference is None:
        print("\nCONTROL CHECK")
        print(f"  Cannot be performed: {REFERENCE_SUMMARY} does not yet contain a")
        print(f"  row for {REFERENCE_RUN}. Run the fer_baseline comparison on the")
        print("  same dataset first, then re-run this script to see the check.")
    elif control and control.get("test_accuracy") not in (None, "") and reference:
        got = float(control["test_accuracy"])
        ref = reference["test_accuracy"]
        delta = got - ref
        print(f"\nCONTROL CHECK")
        print(f"  This loop, gradient mode : {got:.4f}")
        print(f"  Shared loop, same setting: {ref:.4f}   ({reference['source']})")
        print(f"  Difference               : {delta:+.4f}")
        # Standard error of a difference of two proportions on n = 3,589.
        # Both runs are evaluated on the same images, so a paired test
        # (McNemar on the disagreements, or a bootstrap over the per-image
        # predictions) would be sharper. This unpaired figure is the
        # conservative version, and it is the one to quote here.
        n = n_test or 3589
        se_one = (ref * (1 - ref) / n) ** 0.5
        se_diff = se_one * (2 ** 0.5)
        print(f"  Standard error, one run  : {se_one:.4f}")
        print(f"  Standard error, difference: {se_diff:.4f}  "
              f"(unpaired; a paired test would be tighter)")
        if abs(delta) <= 2 * se_diff:
            print(f"  |difference| is within two standard errors ({2 * se_diff:.4f}).")
            print("  The two loops are consistent, so a difference in the other")
            print("  variant may reasonably be attributed to the update rule.")
        else:
            print(f"  |difference| EXCEEDS two standard errors ({2 * se_diff:.4f}).")
            print("  This loop differs from the shared one in some way beyond the")
            print("  update rule. Do not attribute the other variant's result to")
            print("  Algorithm 1 until that is explained.")
        print("\n  Note: this check is reported after both runs finish; it does not")
        print("  gate the second one. Read it before drawing any conclusion.")

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
            # A finished-looking run directory is not enough. It may have been
            # produced by an earlier, different configuration, and reusing it
            # would file old numbers under the current settings without any
            # error. Check that the run's own saved config matches, and that
            # its evaluation actually completed.
            reusable, why = run_matches(existing, base, v)
            if not reusable:
                print(f"\n[stop] {name}: an existing run was found at {existing},")
                print(f"       but it cannot be reused: {why}")
                print("       Re-run with --force to overwrite it, or move it aside.")
                return
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

    reference = read_reference(runs_dir)
    n_test = None
    for r in records:
        ev = newest_eval(r["run_dir"]) if r.get("run_dir") else None
        if ev:
            m = re.search(r"Images:\s*(\d+)", read_text(os.path.join(ev, "metrics.txt")))
            if m:
                n_test = int(m.group(1))
                break
    print_summary(records, reference, n_test)
    print(f"\nSummary appended to {summary_path}")


if __name__ == "__main__":
    main()
