"""Run the nine arms of the room centre-update comparison and summarise them.

Three centre modes across three seeds. The seeds are the point: every result
this project has reported so far comes from a single training run, so a
difference of a few points cannot be separated from what a different
initialisation would have produced anyway.

    none      cross-entropy only. With seed 42 this must reproduce the
              existing baseline run; it is the control on this folder's
              training loop.
    gradient  the variant used in this thesis.
    wen       Algorithm 1 as published.

Each arm is trained, then evaluated on the held-out test set, and the results
are written to one CSV. A finished run directory is reused only if the settings
that determine the result match the current configuration; otherwise the
comparison stops and says which setting differs, rather than filing an old
number under the current name.
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

HERE = Path(__file__).resolve().parent
SUMMARY_FILENAME = "rooms_wen_comparison_results.csv"

SEEDS: List[int] = [42, 43, 44]

MODES: List[Dict[str, str]] = [
    {"center_mode": "none",
     "description": "CONTROL: cross-entropy only, no centre term"},
    {"center_mode": "gradient",
     "description": "Centre Loss, the gradient variant used in this thesis"},
    {"center_mode": "wen",
     "description": "Centre Loss, Algorithm 1 of Wen et al. as published"},
]

SUMMARY_FIELDS = ["run_name", "center_mode", "seed", "description",
                  "center_loss_weight", "center_loss_lr", "best_val_accuracy",
                  "test_accuracy", "minutes", "run_dir"]


def variants() -> List[Dict[str, Any]]:
    out = []
    for m in MODES:
        for s in SEEDS:
            out.append({**m, "seed": s,
                        "run_name": f"rooms_wen_{m['center_mode']}_s{s}"})
    return out


# ----------------------------------------------------------------------
def newest_eval(run_dir: str) -> Optional[str]:
    hits = sorted(glob.glob(os.path.join(run_dir, "outputs", "eval_*")))
    return hits[-1] if hits else None


def find_run_dir(runs_dir: str, run_name: str) -> Optional[str]:
    hits = sorted(d for d in glob.glob(os.path.join(runs_dir, f"*_{run_name}"))
                  if os.path.isdir(d)
                  and os.path.basename(d).endswith("_" + run_name))
    return hits[-1] if hits else None


def read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def run_matches(run_dir: str, base: Dict[str, Any],
                variant: Dict[str, Any]) -> "tuple[bool, str]":
    """Is an existing run directory safe to reuse for this variant?

    A finished-looking directory is not enough: it may have been produced by a
    different configuration, and reusing it would file an old number under the
    current settings without any error. Everything that changes the result is
    compared, including the dataset directories -- all three of them, since the
    validation directory decides which checkpoint is selected.
    """
    cfg_path = os.path.join(run_dir, "logs", "config_used.yaml")
    if not os.path.isfile(cfg_path):
        return False, "it has no logs/config_used.yaml, so its settings are unknown"
    with open(cfg_path, encoding="utf-8") as fh:
        old = yaml.safe_load(fh)

    ot, om, od = old.get("training", {}), old.get("model", {}), old.get("data", {})
    bt, bm, bd = base["training"], base["model"], base["data"]
    checks = [
        ("training.center_mode", ot.get("center_mode"), variant["center_mode"]),
        ("training.seed", ot.get("seed"), variant["seed"]),
        ("training.center_loss_weight", ot.get("center_loss_weight"),
         bt["center_loss_weight"]),
        ("training.center_loss_lr", ot.get("center_loss_lr"), bt["center_loss_lr"]),
        ("training.epochs", ot.get("epochs"), bt["epochs"]),
        ("training.batch_size", ot.get("batch_size"), bt["batch_size"]),
        ("training.learning_rate", ot.get("learning_rate"), bt["learning_rate"]),
        ("training.weight_decay", ot.get("weight_decay"), bt["weight_decay"]),
        ("training.label_smoothing", ot.get("label_smoothing"), bt["label_smoothing"]),
        ("training.grad_clip", ot.get("grad_clip"), bt["grad_clip"]),
        ("training.scheduler", ot.get("scheduler"), bt["scheduler"]),
        ("training.warmup_epochs", ot.get("warmup_epochs"), bt["warmup_epochs"]),
        ("training.use_amp", ot.get("use_amp"), bt["use_amp"]),
        ("model.arch", om.get("arch"), bm["arch"]),
        ("model.image_size", om.get("image_size"), bm["image_size"]),
        ("model.embed_dim", om.get("embed_dim"), bm["embed_dim"]),
        ("model.depth", om.get("depth"), bm["depth"]),
        ("model.num_heads", om.get("num_heads"), bm["num_heads"]),
        ("model.num_classes", om.get("num_classes"), bm["num_classes"]),
        ("data.train_dir", od.get("train_dir"), bd["train_dir"]),
        ("data.val_dir", od.get("val_dir"), bd["val_dir"]),
        ("data.eval_dir", od.get("eval_dir"), bd["eval_dir"]),
        ("data.class_names", od.get("class_names"), bd["class_names"]),
        ("data.norm_mean", od.get("norm_mean"), bd["norm_mean"]),
        ("data.norm_std", od.get("norm_std"), bd["norm_std"]),
    ]
    for key, was, now in checks:
        if was != now:
            return False, f"{key} was {was!r} in that run but is {now!r} now"

    ev = newest_eval(run_dir)
    if ev is None:
        return False, "it has no evaluation directory"
    required = ["metrics.txt", "classification_report.txt", "confusion_matrix.csv",
                "predictions.csv", "embeddings.npy", "labels.npy"]
    missing = [f for f in required if not os.path.isfile(os.path.join(ev, f))]
    if missing:
        return False, f"its evaluation is incomplete, missing {missing}"
    return True, ""


def build_variant_config(base: Dict[str, Any], v: Dict[str, Any],
                         workers: Optional[int]) -> Path:
    cfg = copy.deepcopy(base)
    cfg["run_name"] = v["run_name"]
    cfg["training"]["center_mode"] = v["center_mode"]
    cfg["training"]["seed"] = v["seed"]
    if workers is not None:
        cfg["training"]["num_workers"] = int(workers)
    out = HERE / f"config_used_{v['run_name']}.yaml"
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    return out


def parse_metric(text: str, key: str) -> Optional[float]:
    for line in text.splitlines():
        if line.startswith(key):
            try:
                return float(line.split(":")[1].strip())
            except (IndexError, ValueError):
                return None
    return None


def print_summary(records: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 92)
    print(f"{'arm':>10} {'seed':>5} {'val':>9} {'test':>9} {'minutes':>8}   run")
    print("=" * 92)
    for r in records:
        val = r.get("best_val_accuracy")
        test = r.get("test_accuracy")
        print(f"{r['center_mode']:>10} {r['seed']:>5} "
              f"{(f'{val:.4f}' if val is not None else '-'):>9} "
              f"{(f'{test:.4f}' if test is not None else '-'):>9} "
              f"{str(r.get('minutes', '-')):>8}   {os.path.basename(str(r.get('run_dir','')))}")
    print("=" * 92)

    by_mode: Dict[str, List[float]] = {}
    for r in records:
        if r.get("test_accuracy") is not None:
            by_mode.setdefault(r["center_mode"], []).append(r["test_accuracy"])
    if any(len(v) > 1 for v in by_mode.values()):
        print("\nAcross seeds (test accuracy):")
        for m in ("none", "gradient", "wen"):
            vals = by_mode.get(m, [])
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                sd = (sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
                print(f"  {m:>10}: mean {mean*100:.2f}%  sd {sd*100:.2f}pp  "
                      f"n={len(vals)}  values {[f'{v*100:.2f}' for v in vals]}")
            else:
                print(f"  {m:>10}: {mean*100:.2f}%  (one seed only)")
        print("\n  The spread across seeds is the quantity every earlier "
              "single-run comparison\n  in this project was missing. Read any "
              "difference between arms against it.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the nine-arm comparison.")
    ap.add_argument("--config", default=str(HERE / "config_rooms_wen.yaml"))
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="retrain even if a matching finished run exists")
    ap.add_argument("--only", default=None,
                    help="comma-separated centre modes to run, e.g. none,wen")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    runs_dir = args.runs_dir or base["paths"]["output_root"]
    os.makedirs(runs_dir, exist_ok=True)

    wanted = ([m.strip() for m in args.only.split(",")] if args.only
              else [m["center_mode"] for m in MODES])
    todo = [v for v in variants() if v["center_mode"] in wanted]

    print("=" * 92)
    print("ROOM DATASET -- CENTRE UPDATE RULE, ACROSS THREE SEEDS")
    print("=" * 92)
    print(f"config     : {os.path.basename(args.config)}")
    print(f"runs dir   : {runs_dir}")
    print(f"resolution : {base['model']['image_size']}x{base['model']['image_size']}")
    print(f"epochs     : {base['training']['epochs']}, batch "
          f"{base['training']['batch_size']}")
    print(f"lambda     : {base['training']['center_loss_weight']}, "
          f"centre rate {base['training']['center_loss_lr']}")
    print(f"seeds      : {SEEDS}")
    print(f"arms       : {[v['run_name'] for v in todo]}")

    if args.dry_run:
        print("\n--dry-run: nothing was executed.")
        return

    records: List[Dict[str, Any]] = []
    trained_any = False

    for v in todo:
        name = v["run_name"]
        existing = find_run_dir(runs_dir, name)
        if existing and newest_eval(existing) and not args.force:
            reusable, why = run_matches(existing, base, v)
            if not reusable:
                print(f"\n[stop] {name}: a finished run exists at {existing},")
                print(f"       but it cannot be reused: {why}")
                print("       Re-run with --force to overwrite it, or move it aside.")
                return
            print(f"\n[skip] {name}: already evaluated at {existing}")
            ev = newest_eval(existing)
            m = read_text(os.path.join(ev, "metrics.txt"))
            records.append({
                "run_name": name, "center_mode": v["center_mode"], "seed": v["seed"],
                "description": v["description"],
                "center_loss_weight": base["training"]["center_loss_weight"],
                "center_loss_lr": base["training"]["center_loss_lr"],
                "best_val_accuracy": None,
                "test_accuracy": parse_metric(m, "Overall accuracy"),
                "minutes": "", "run_dir": existing,
            })
            continue

        cfg_path = build_variant_config(base, v, args.num_workers)
        print("\n" + "=" * 92)
        print(f"TRAIN  {name}   (mode={v['center_mode']}, seed={v['seed']})")
        print("=" * 92)
        t0 = time.time()
        subprocess.run([sys.executable, str(HERE / "train_rooms_wen.py"),
                        "--config", str(cfg_path)], check=True)
        minutes = round((time.time() - t0) / 60.0, 1)
        trained_any = True

        run_dir = find_run_dir(runs_dir, name)
        if run_dir is None:
            raise SystemExit(f"{name}: training finished but no run directory "
                             f"matching *_{name} appeared under {runs_dir}")
        ckpt = os.path.join(run_dir, "checkpoints", "best_model.pt")
        print(f"\nEVALUATE  {name}")
        subprocess.run([sys.executable, str(HERE / "evaluate_rooms_wen.py"),
                        "--checkpoint", ckpt], check=True)

        ev = newest_eval(run_dir)
        m = read_text(os.path.join(ev, "metrics.txt"))
        log = read_text(os.path.join(run_dir, "logs", "train.log"))
        best_val = None
        for line in log.splitlines():
            if "Best val accuracy" in line:
                try:
                    best_val = float(line.rsplit(":", 1)[1].strip())
                except (IndexError, ValueError):
                    best_val = None
        records.append({
            "run_name": name, "center_mode": v["center_mode"], "seed": v["seed"],
            "description": v["description"],
            "center_loss_weight": base["training"]["center_loss_weight"],
            "center_loss_lr": base["training"]["center_loss_lr"],
            "best_val_accuracy": best_val,
            "test_accuracy": parse_metric(m, "Overall accuracy"),
            "minutes": minutes, "run_dir": run_dir,
        })

    if not trained_any:
        print("\n" + "!" * 92)
        print("NOTHING WAS TRAINED: every arm was skipped as already complete.")
        print("The table below therefore reports earlier runs, not this invocation.")
        print("!" * 92)

    import csv as _csv
    out = os.path.join(runs_dir, SUMMARY_FILENAME)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})
    print_summary(records)
    print(f"\nSummary written to {out}")
    print(f"Finished {datetime.now():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
