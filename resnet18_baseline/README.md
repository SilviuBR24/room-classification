# ResNet-18 convolutional baseline

The comparison requested by the thesis supervisor: **ResNet-18 with plain
cross-entropy versus ResNet-18 with cross-entropy + Center Loss**, on the same
six-class indoor room dataset, so the convolutional baseline can be set against
the ViT-S/16 results.

## Run it

One command produces everything:

```bash
python compare_resnet18.py --config config_resnet18.yaml
```

On Colab, open `run_resnet18_colab.ipynb` and run the cells top to bottom.
Roughly 1–1.5 hours per variant on a T4, so 2–3 hours in total.

Useful flags:

```bash
python compare_resnet18.py --config ... --dry-run                        # plan only
python compare_resnet18.py --config ... --only resnet18_crossentropy     # one variant
python compare_resnet18.py --config ... --force                          # redo a finished variant
```

## What gets written

Per variant, under the runs directory from `paths.output_root`:

```
<YYYY-MM-DD_HH-MM>_resnet18_crossentropy/
<YYYY-MM-DD_HH-MM>_resnet18_crossentropy_centerloss/
    checkpoints/   best_model.pt, last_checkpoint.pt
    logs/          train.log, metrics.csv, config_used.yaml
    outputs/       eval_<timestamp>/
                       metrics.txt                 overall + per-class accuracy
                       classification_report.txt   precision / recall / F1
                       confusion_matrix.png        counts
                       confusion_matrix_norm.png   row-normalised
                       confusion_matrix.csv
                       predictions.csv             per image, with class probabilities
                       embeddings.npy, labels.npy  for the geometry analysis
```

Plus one accumulating summary, so the numbers never have to be dug out of logs:

```
resnet18_comparison_results.csv
```

Interrupting is safe. The summary row is appended after each variant finishes,
and re-running the same command skips variants whose evaluation already
completed — a Colab disconnect costs only the variant in progress.

## Files

| File | Purpose |
|---|---|
| `config_resnet18.yaml` | every parameter, shared by both variants |
| `resnet_model.py` | the ResNet-18 wrapper (the only model code here) |
| `train_resnet18.py` | train one variant |
| `evaluate_resnet18.py` | evaluate a checkpoint on the test set |
| `compare_resnet18.py` | **one command: both variants + summary** |
| `shared_infrastructure.py` | puts `vit_s16_baseline/` on `sys.path` |
| `run_resnet18_colab.ipynb` | Colab runner |

## How this relates to `vit_s16_baseline/`

This folder owns **exactly one thing: the ResNet-18 model**. Everything else —
the training loop, the dataset, checkpointing, logging, Center Loss, the
evaluation script — is imported unmodified from `vit_s16_baseline/`.

That is a deliberate experimental decision, not just convenience. The comparison
only means something if the **architecture is the only difference** between the
runs. A second copy of the training loop would drift over time, and then a gap
in the results could come from the training code rather than from the model.

Consequently `resnet18_baseline/` does not work on its own: it expects
`vit_s16_baseline/` to sit next to it. Nothing in that folder is modified.

### Why the model needs a wrapper

`Trainer` calls `model(images, return_embeddings=True)` whenever Center Loss is
active, because the centre term operates on the penultimate representation.
torchvision's ResNet has no such argument — it goes straight from the pooled
feature map into the classification head, so the embedding is never exposed.

`resnet_model.py` replaces that head with `Identity` and adds an explicit
classifier. The computation is unchanged; the 512-dimensional pooled feature
vector simply becomes observable, playing the same role as the ViT's 384-dim
CLS token. The result is API-compatible with `src/vit.py`, so the shared
Trainer, Center Loss and evaluation code all work untouched.

The architecture comes from `torchvision.models.resnet18` (He et al., CVPR 2016)
and is instantiated with `weights=None` — **the architecture is reused, not any
pre-trained weights.** Every parameter is learned from scratch on the room
dataset, exactly as the ViT was, which is what the thesis requires and what
makes the comparison fair.

## Experimental settings

Deliberately identical to the ViT runs, so the architecture is the only variable:

| Setting | Value |
|---|---|
| Split | 3200 train / 300 val / 100 test per class |
| Optimizer | AdamW, lr 3e-4, weight decay 0.05 |
| Schedule | cosine, 5 warmup epochs |
| Epochs | 50 |
| Batch size | 64 |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |
| Center Loss lambda | 0.0005 (the best value from the ViT sweep) |
| Centre learning rate | 0.5 |
| Seed | 42 |
| Mixed precision | on |

Do not tune these for ResNet. A tuned CNN against an untuned transformer would
not answer the question being asked.

| | ResNet-18 | ViT-S/16 |
|---|---|---|
| Parameters | 11,179,590 | 21,691,014 |
| Embedding | 512 (pooled) | 384 (CLS token) |

## Known wart

The shared `train.py` prints a hard-coded line in `train.log`:

```
Model: ViT-S/16 from scratch | trainable params: 11,179,590
```

It says **ViT-S/16 even when a ResNet was built**, because that string is fixed
inside the shared trainer and this folder deliberately does not modify
`vit_s16_baseline/`. The parameter count in the same line is correct and gives
the game away: 11,179,590 is ResNet-18, 21,691,014 is ViT-S/16.

The authoritative record of what actually ran is elsewhere and unambiguous: the
run folder name, `arch: resnet18` in `config_used.yaml` (also stored inside every
checkpoint), and that parameter count. `train_resnet18.py` additionally prints a
warning about this line before handing over.

## After the runs

Compare the embedding geometry across architectures using the shared analysis:

```bash
python ../vit_s16_baseline/analyze_embeddings.py \
    --runs <..._resnet18_crossentropy> <..._resnet18_crossentropy_centerloss> \
    --labels "ResNet-18 (CE)" "ResNet-18 (CE + Center Loss)" \
    --out-dir ../figuri
```

Cell 6 of the Colab notebook does this automatically.
