"""
shared_infrastructure.py
========================
Makes the training infrastructure that lives in `vit_s16_baseline/` importable
from this folder.

Why this experiment borrows rather than copies
----------------------------------------------
This folder repeats the Center Loss comparison on a second task: facial
expression recognition. The point of repeating it is to find out whether the
behaviour observed on the room dataset -- Center Loss compacting the
representation without improving separability -- also appears on an unrelated
dataset, or whether it was specific to indoor scenes.

Note that this is a related task, not the original one. Center Loss was proposed
for face identity recognition, which differs from expression recognition in the
number of classes, in how coherent each class is, and in what distinguishes
them. FER2013 tests the hypothesis further; it does not reproduce the setting
the method was designed for.

That question can only be answered if the training procedure is the same one
used for the room experiments. A second copy of the training loop would drift,
and a difference in outcome could then come from the code rather than from the
task. So this folder owns exactly one thing -- its own model wrapper -- and
borrows the Trainer, dataset, checkpointing, logging, Center Loss and the
evaluation script from the ViT project, unmodified.

It deliberately does NOT import from `resnet18_baseline/`. The two folders would
otherwise be coupled, and changing one could break the other. A small amount of
duplication between two model wrappers is a fair price for that isolation.

Import this module first, before any `from src...` or `import train` line:

    import shared_infrastructure  # noqa: F401  (sets sys.path)
    from src.trainer import Trainer

There is deliberately no `src/` folder here: that name is already taken by
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
