"""
export_verification_data.py
===========================
Collects the raw evaluation artefacts from every experiment run into ONE plain
text file, so the numbers quoted in the thesis (and in THESIS_HANDOFF.md) can be
checked against their source without needing access to Google Drive.

Why this exists
---------------
The results live in per-run folders on Drive. Handing someone a Drive link is
unreliable -- folder links render as a JavaScript app, checkpoints are large
binaries, and private folders need authentication. Everything actually worth
verifying is a handful of small text files, so exporting them into a single
pasteable document is both simpler and deterministic.

Only artefacts that can be independently checked are included: overall and
per-class accuracy, confusion matrices, classification reports, and the sweep
summaries. Checkpoints, logs and embeddings are skipped.

    python export_verification_data.py
    python export_verification_data.py --runs-dir "G:/My Drive/.../dissertation_runs"
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import List

DEFAULT_RUNS_DIR = r"G:/My Drive/Dissertation_Thesis/dissertation_runs"
DEFAULT_OUTPUT = "verificare_rezultate.txt"

# Per-run files worth exporting, in the order they should appear.
RUN_ARTEFACTS = [
    ("metrics.txt", "OVERALL AND PER-CLASS ACCURACY"),
    ("classification_report.txt", "CLASSIFICATION REPORT (precision / recall / F1)"),
    ("confusion_matrix.csv", "CONFUSION MATRIX (rows = true, columns = predicted)"),
]

# Summary files sitting directly in the runs directory.
SUMMARY_FILES = ["sweep_results.csv", "resnet18_comparison_results.csv"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export raw results for verification.")
    p.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                   help="Directory holding the timestamped run folders.")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="File to write.")
    return p.parse_args()


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace").strip()


def newest_eval_dir(run_dir: str) -> str | None:
    """Runs can be evaluated more than once; the newest evaluation is the one
    the reported numbers come from."""
    evals = sorted(glob.glob(os.path.join(run_dir, "outputs", "eval_*")))
    return evals[-1] if evals else None


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir
    if not os.path.isdir(runs_dir):
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    run_dirs = sorted(d for d in glob.glob(os.path.join(runs_dir, "*"))
                      if os.path.isdir(d))

    out: List[str] = []
    out.append("=" * 78)
    out.append("RAW EVALUATION ARTEFACTS -- exported for independent verification")
    out.append("=" * 78)
    out.append(f"Exported: {datetime.now():%Y-%m-%d %H:%M}")
    out.append(f"Source:   {runs_dir}")
    out.append("")
    out.append("Every number reported in the dissertation should be traceable to one of")
    out.append("the blocks below. These are copies of the files written by evaluate.py at")
    out.append("the end of each run -- nothing here is recomputed or summarised by hand.")
    out.append("")
    out.append("Note: some runs belong to the early, superseded protocol, in which the")
    out.append("best model was selected on the test set. Those are labelled where the run")
    out.append("name makes it clear (the 3500-image runs). The reported thesis numbers")
    out.append("come from the 3200/300/1500/100 split runs.")

    exported, skipped = 0, []
    for run_dir in run_dirs:
        name = os.path.basename(run_dir)
        eval_dir = newest_eval_dir(run_dir)
        if eval_dir is None:
            skipped.append(f"{name} (no evaluation)")
            continue

        out.append("")
        out.append("#" * 78)
        out.append(f"# RUN: {name}")
        out.append(f"# evaluation: {os.path.basename(eval_dir)}")
        out.append("#" * 78)

        for filename, heading in RUN_ARTEFACTS:
            path = os.path.join(eval_dir, filename)
            if not os.path.isfile(path):
                continue
            out.append("")
            out.append(f"--- {heading} [{filename}] ---")
            out.append(read(path))
        exported += 1

    # Sweep summaries live one level up, next to the run folders.
    for filename in SUMMARY_FILES:
        path = os.path.join(runs_dir, filename)
        if os.path.isfile(path):
            out.append("")
            out.append("#" * 78)
            out.append(f"# SUMMARY FILE: {filename}")
            out.append("#" * 78)
            out.append(read(path))

    if skipped:
        out.append("")
        out.append("#" * 78)
        out.append("# RUNS WITHOUT A COMPLETED EVALUATION (nothing to verify)")
        out.append("#" * 78)
        out.extend(f"  {s}" for s in skipped)

    text = "\n".join(out) + "\n"
    Path(args.output).write_text(text, encoding="utf-8")

    words = len(text.split())
    print(f"Wrote {args.output}")
    print(f"  runs exported : {exported}")
    print(f"  runs skipped  : {len(skipped)}")
    print(f"  size          : {len(text):,} chars, ~{int(words * 1.35):,} tokens")


if __name__ == "__main__":
    main()
