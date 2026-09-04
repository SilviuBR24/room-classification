"""Run the nine arms of the room centre-update comparison and summarise them.

Three centre modes across three seeds. The seeds are the point: every result
this project has reported so far comes from a single training run, so a
difference of a few points cannot be separated from what a different
initialisation would have produced anyway.

    none      cross-entropy only. At seed 42 it should closely match the
              existing baseline run; it is the consistency check on this
              folder's training loop.
    gradient  the variant used in this thesis.
    wen       the original centre-update rule, Equation (4) and Algorithm 1 of Wen et al..

Each arm is trained, then evaluated on the held-out test set, and the results
are written to one CSV. A finished run directory is reused only if the settings
that determine the result match the current configuration; otherwise the
comparison stops and says which setting differs, rather than filing an old
number under the current name.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

HERE = Path(__file__).resolve().parent
# Generated per-variant configs go here, never into the repository.
TEMP_CONFIG_DIR = tempfile.mkdtemp(prefix="rooms_wen_cfg_")
SUMMARY_FILENAME = "rooms_wen_comparison_results.csv"

SEEDS: List[int] = [42, 43, 44]

MODES: List[Dict[str, str]] = [
    {"center_mode": "none",
     "description": "CONTROL: cross-entropy only, no centre term"},
    {"center_mode": "gradient",
     "description": "Centre Loss, the gradient variant used in this thesis"},
    {"center_mode": "wen",
     "description": "Centre Loss, the original centre-update rule, Equation (4) and Algorithm 1 of Wen et al."},
]

SUMMARY_FIELDS = ["run_name", "center_mode", "seed", "description",
                  "center_loss_weight", "center_loss_lr", "best_val_accuracy",
                  "test_accuracy", "minutes", "run_dir"]


def variants() -> List[Dict[str, Any]]:
    """The nine arms, ordered seed by seed rather than mode by mode.

    Running all three seeds of one mode before starting the next would put each
    mode in its own stretch of wall-clock time, and these nine runs will span
    several Colab sessions on whatever GPU is allocated. A change of machine
    would then line up with a change of mode, and the two could not be told
    apart. Interleaving spreads any such change across all three modes instead.
    """
    out = []
    for s in SEEDS:
        for m in MODES:
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


def canonical(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The configuration reduced to what actually determines the result."""
    import copy
    c = copy.deepcopy(cfg)
    c.pop("run_name", None)
    c.pop("paths", None)
    c.get("training", {}).pop("output_root", None)
    return c


def config_differences(old: Dict[str, Any], new: Dict[str, Any],
                       prefix: str = "") -> List[str]:
    """Every difference between two configurations, as readable paths."""
    out: List[str] = []
    for key in sorted(set(old) | set(new)):
        path = f"{prefix}{key}"
        a, b = old.get(key, "<missing>"), new.get(key, "<missing>")
        if isinstance(a, dict) and isinstance(b, dict):
            out.extend(config_differences(a, b, path + "."))
        elif a != b:
            out.append(f"{path}: was {a!r}, now {b!r}")
    return out


def current_commit() -> Optional[str]:
    """The commit this code is running from, or None outside a checkout."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(HERE),
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def run_matches(run_dir: str, base: Dict[str, Any], variant: Dict[str, Any],
                expected_commit: Optional[str] = None) -> "tuple[bool, str]":
    """Is an existing run directory safe to reuse for this variant?

    Three things have to hold, and a finished-looking directory shows none of
    them on its own.

    The configuration must match. Not a list of fields someone remembered to
    check -- the whole configuration, minus the run name and the output
    location, which do not change the numbers. A hand-kept list is a list that
    quietly goes out of date.

    The code must match. Nine runs spread over several Colab sessions can be
    produced by different commits while the configuration stays identical, and
    comparing configurations would never notice.

    The environment must be compatible. PyTorch does not guarantee identical
    results across versions or platforms, so the major PyTorch version is
    compared and a difference in GPU is reported rather than ignored.
    """
    cfg_path = os.path.join(run_dir, "logs", "config_used.yaml")
    if not os.path.isfile(cfg_path):
        return False, "it has no logs/config_used.yaml, so its settings are unknown"
    with open(cfg_path, encoding="utf-8") as fh:
        old_cfg = yaml.safe_load(fh)

    # num_workers is compared like everything else. It is not cosmetic: the
    # worker seeds are derived from the loader's generator, so a different
    # number of workers means a different augmentation stream, and two arms
    # trained with different worker counts are not the paired comparison they
    # would appear to be.
    want = variant_config(base, variant, workers=None)
    diffs = config_differences(canonical(old_cfg), canonical(want))
    if diffs:
        return False, "the configuration differs -- " + "; ".join(diffs[:3])

    prov_path = os.path.join(run_dir, "logs", "provenance.json")
    if not os.path.isfile(prov_path):
        return False, ("it has no logs/provenance.json, so the code and "
                       "environment that produced it are unknown")
    with open(prov_path, encoding="utf-8") as fh:
        prov = json.load(fh)
    if not prov.get("git_commit"):
        return False, "its provenance records no commit"
    want_commit = expected_commit if expected_commit is not None else current_commit()
    if want_commit and prov["git_commit"] != want_commit:
        return False, (f"it was produced from commit {prov['git_commit'][:8]}, "
                       f"but this code is {want_commit[:8]}")
    if prov.get("git_dirty"):
        return False, (f"it was produced from a dirty working tree at "
                       f"{prov['git_commit'][:8]}, so its code is not identified")

    ev = newest_eval(run_dir)
    if ev is None:
        return False, "it has no evaluation directory"
    required = ["metrics.txt", "classification_report.txt", "confusion_matrix.csv",
                "predictions.csv", "embeddings.npy", "labels.npy"]
    missing = [f for f in required if not os.path.isfile(os.path.join(ev, f))]
    if missing:
        return False, f"its evaluation is incomplete, missing {missing}"

    if best_val_from_checkpoint(run_dir) is None:
        return False, ("its checkpoint carries no best_val_accuracy, so the "
                       "validation result cannot be recovered")
    return True, ""


def consistent_provenance(records: List[Dict[str, Any]]) -> List[str]:
    """Complaints about runs that were not produced by the same code or runtime."""
    seen: Dict[str, set] = {"git_commit": set(), "torch": set(), "gpu": set()}
    for r in records:
        for k in seen:
            v = (r.get("provenance") or {}).get(k)
            if v:
                seen[k].add(str(v))
    out = []
    if len(seen["git_commit"]) > 1:
        out.append(f"produced by {len(seen['git_commit'])} different commits: "
                   f"{sorted(c[:8] for c in seen['git_commit'])}")
    majors = {v.split("+")[0].rsplit(".", 1)[0] for v in seen["torch"]}
    if len(majors) > 1:
        out.append(f"produced under different PyTorch versions: {sorted(seen['torch'])}")
    if len(seen["gpu"]) > 1:
        out.append(f"produced on different GPUs: {sorted(seen['gpu'])}")
    return out


def best_val_from_checkpoint(run_dir: str) -> Optional[float]:
    """The validation accuracy the run itself recorded.

    Read from the checkpoint rather than parsed out of the log. The log line is
    a formatted string that a change in wording would silently break, and after
    a disconnection the summary was filing None for every reused run, losing the
    validation column for work that had already been done.
    """
    ckpt = os.path.join(run_dir, "checkpoints", "best_model.pt")
    if not os.path.isfile(ckpt):
        return None
    try:
        import torch
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception:
        return None
    v = state.get("best_val_accuracy")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load_provenance(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "logs", "provenance.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def variant_config(base: Dict[str, Any], v: Dict[str, Any],
                   workers: Optional[int]) -> Dict[str, Any]:
    """The configuration this variant would be trained with."""
    cfg = copy.deepcopy(base)
    cfg["run_name"] = v["run_name"]
    cfg["training"]["center_mode"] = v["center_mode"]
    cfg["training"]["seed"] = v["seed"]
    if workers is not None:
        cfg["training"]["num_workers"] = int(workers)
    return cfg


def build_variant_config(base: Dict[str, Any], v: Dict[str, Any],
                         workers: Optional[int]) -> Path:
    cfg = variant_config(base, v, workers)
    # Written outside the repository. Inside it, this file made the working
    # tree dirty, the trainer recorded git_dirty=True in its provenance, and
    # the reuse check then rejected the run's own output on the next session --
    # which is precisely the resume path these nine runs depend on.
    out = Path(TEMP_CONFIG_DIR) / f"config_used_{v['run_name']}.yaml"
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


def find_valid_run_dir(runs_dir: str, base: Dict[str, Any],
                       v: Dict[str, Any]) -> Optional[str]:
    """The newest run for this variant that is actually usable.

    Taking the newest directory and stopping there hid a complete run behind an
    incomplete one -- an arm interrupted halfway would mask the finished arm
    before it and be retrained for nothing.
    """
    for d in sorted(glob.glob(os.path.join(runs_dir, f"*_{v['run_name']}")),
                    reverse=True):
        if not os.path.isdir(d) or not os.path.basename(d).endswith(
                "_" + v["run_name"]):
            continue
        ok, _ = run_matches(d, base, v)
        if ok:
            return d
    return None


def record_for(runs_dir: str, base: Dict[str, Any], v: Dict[str, Any],
               minutes: Any = "") -> Optional[Dict[str, Any]]:
    """Read back a finished run as a summary row, or None if it is not usable."""
    run_dir = find_valid_run_dir(runs_dir, base, v)
    if run_dir is None:
        return None
    ev = newest_eval(run_dir)
    return {
        "run_name": v["run_name"], "center_mode": v["center_mode"],
        "seed": v["seed"], "description": v["description"],
        "center_loss_weight": base["training"]["center_loss_weight"],
        "center_loss_lr": base["training"]["center_loss_lr"],
        "best_val_accuracy": best_val_from_checkpoint(run_dir),
        "test_accuracy": parse_metric(read_text(os.path.join(ev, "metrics.txt")),
                                      "Overall accuracy"),
        "minutes": minutes,
        "run_dir": run_dir,
        "provenance": load_provenance(run_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the nine-arm comparison.")
    ap.add_argument("--config", default=str(HERE / "config_rooms_wen.yaml"))
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--force", action="store_true",
                    help="train again even if a matching finished run exists; "
                         "the new run goes to a new directory and the old one "
                         "is left in place but no longer selected")
    ap.add_argument("--only", default=None,
                    help="comma-separated centre modes to train in this "
                         "invocation, e.g. none,wen. The summary still reports "
                         "every finished arm, not just these.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    runs_dir = args.runs_dir or base["paths"]["output_root"]
    os.makedirs(runs_dir, exist_ok=True)

    all_variants = variants()
    wanted = ([m.strip() for m in args.only.split(",")] if args.only
              else [m["center_mode"] for m in MODES])
    unknown = set(wanted) - {m["center_mode"] for m in MODES}
    if unknown:
        raise SystemExit(f"--only names unknown modes: {sorted(unknown)}")
    todo = [v for v in all_variants if v["center_mode"] in wanted]

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
    print(f"to train   : {[v['run_name'] for v in todo]}")
    if args.only:
        print("             (the summary will still cover all nine arms)")

    if args.dry_run:
        print()
        print("--dry-run: nothing was executed.")
        return

    trained: Dict[str, Any] = {}

    for v in todo:
        name = v["run_name"]
        # Look for a usable run before anything else. Checking only the newest
        # directory meant an arm interrupted before its evaluation hid the
        # finished arm behind it, and it would have been trained again for
        # nothing -- two hours each time.
        valid = None if args.force else find_valid_run_dir(runs_dir, base, v)
        if valid is not None:
            print(f"[skip] {name}: already finished at {os.path.basename(valid)}")
            continue

        existing = find_run_dir(runs_dir, name)
        if existing and newest_eval(existing) and not args.force:
            reusable, why = run_matches(existing, base, v)
            if not reusable:
                print()
                print(f"[stop] {name}: a finished run exists at {existing},")
                print(f"       but it cannot be reused: {why}")
                print("       Re-run with --force to train it again into a new")
                print("       directory, or move the old one aside.")
                return
            print()
            print(f"[skip] {name}: already finished at {os.path.basename(existing)}")
            continue

        cfg_path = build_variant_config(base, v, None)
        print()
        print("" + "=" * 92)
        print(f"TRAIN  {name}   (mode={v['center_mode']}, seed={v['seed']})")
        print("=" * 92)
        t0 = time.time()
        subprocess.run([sys.executable, str(HERE / "train_rooms_wen.py"),
                        "--config", str(cfg_path)], check=True)
        trained[name] = round((time.time() - t0) / 60.0, 1)

        run_dir = find_run_dir(runs_dir, name)
        if run_dir is None:
            raise SystemExit(f"{name}: training finished but no run directory "
                             f"matching *_{name} appeared under {runs_dir}")
        print()
        print(f"EVALUATE  {name}")
        subprocess.run([sys.executable, str(HERE / "evaluate_rooms_wen.py"),
                        "--checkpoint",
                        os.path.join(run_dir, "checkpoints", "best_model.pt")],
                       check=True)

    # The summary covers every arm that is finished and valid, whether it was
    # trained just now, in an earlier session, or under a different --only.
    # Building it from what is on disk is what makes the run survive a Colab
    # disconnection without losing the arms already done.
    records: List[Dict[str, Any]] = []
    for v in all_variants:
        r = record_for(runs_dir, base, v, minutes=trained.get(v["run_name"], ""))
        if r is not None:
            records.append(r)

    seen = [(r["center_mode"], r["seed"]) for r in records]
    if len(seen) != len(set(seen)):
        raise SystemExit(f"duplicate (mode, seed) rows in the summary: {seen}")

    complaints = consistent_provenance(records)
    if complaints:
        print()
        print("" + "!" * 92)
        print("THESE RESULTS WERE NOT ALL PRODUCED THE SAME WAY:")
        for c in complaints:
            print("  -", c)
        print("Comparing them across arms assumes they were. Check before using.")
        print("!" * 92)

    out = os.path.join(runs_dir, SUMMARY_FILENAME)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})

    if not trained:
        print()
        print("" + "!" * 92)
        print("NOTHING WAS TRAINED IN THIS INVOCATION: every requested arm was")
        print("already finished. The table below reports earlier runs.")
        print("!" * 92)

    print_summary(records)
    print()
    print(f"{len(records)}/9 arms finished")
    print(f"Summary written to {out}")
    print(f"Finished {datetime.now():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
