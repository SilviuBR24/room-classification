"""
shared_infrastructure.py
========================
Makes the training infrastructure that lives in `vit_s16_baseline/` importable
from this folder.

Why the two parts share code
----------------------------
The comparison the supervisor asked for -- ResNet-18 versus ViT-S/16, each with
and without Center Loss -- is only meaningful if the ONLY thing that differs
between the runs is the architecture. Re-implementing the training loop here
would put that guarantee at risk: two copies drift, and then a difference in
results could come from the training code rather than from the model.

So this folder owns exactly one thing -- the ResNet-18 model -- and borrows
everything else (Trainer, dataset, checkpointing, logging, Center Loss, the
evaluation script) from the ViT project, unmodified.

Import this module first, before any `from src...` or `import train` line:

    import shared_infrastructure  # noqa: F401  (sets sys.path)
    from src.trainer import Trainer

Note there is deliberately no `src/` folder here: that name is already taken by
`vit_s16_baseline/src`, and a second one would shadow it on `sys.path`.
"""
from __future__ import annotations

import sys
from pathlib import Path

VIT_PROJECT_ROOT = Path(__file__).resolve().parent.parent / "vit_s16_baseline"

if not VIT_PROJECT_ROOT.is_dir():
    raise RuntimeError(
        f"Expected the shared infrastructure at {VIT_PROJECT_ROOT}, but that "
        f"directory does not exist. This folder is a sibling of "
        f"'vit_s16_baseline' and depends on it."
    )

if str(VIT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(VIT_PROJECT_ROOT))
