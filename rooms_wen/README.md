# Centre update rule on the room dataset

Nine runs: three centre modes across three seeds, on the dataset the thesis is
actually about.

## The question

The thesis uses a Centre Loss variant that is not the one published by Wen et
al. Both minimise the same quantity — the distance from each embedding to its
class centre — but they move the centres differently, and the difference is not
a detail:

| | how a centre moves per step |
|---|---|
| **gradient** (this thesis) | the centre is an ordinary parameter stepped by SGD on the combined objective, so λ multiplies its gradient and the class sum is divided by the batch size `B`. Displacement ∝ `η_c · λ / B` |
| **wen** (Algorithm 1, as published) | a dedicated update with its own rate `α`, independent of λ, normalised per class by `(1 + n_j)`. Displacement ∝ `α / (1 + n_j)` |

Neither is adjusted to make the comparison neater. Each is run exactly as it is
defined, because the question is which of the two, as used, is better.

## What this experiment can and cannot show

The ratio of the two displacements is `α·B / (η_c·λ·(1 + n_j))`.

On FER2013 that ratio varied nine-fold *across classes* — about 63,000 for the
rare `disgust` class against about 7,100 for the common `happy` class — because
`n_j` varied with class frequency. That variation is the rule difference proper,
and it is what produced the effect measured there.

The room dataset is balanced: 3,200 images in every class. With `B = 64` the
expected `n_j` is 10.67 for all six classes, so the ratio is the same number
everywhere, about **10,971×**. On this dataset the two rules differ by a uniform
factor on the centre displacement, not by the per-class behaviour that
distinguishes them.

So a difference found here is evidence about **the two rules as specified**,
which is the question being asked. It is *not* evidence about per-class
normalisation, which balanced data cannot show. Each run logs its own
displacement ratio so this number sits with the results rather than being
reconstructed later.

## The arms

| mode | λ | rate | what it is |
|---|---|---|---|
| `none` | — | — | cross-entropy only |
| `gradient` | 0.0005 | η_c = 0.5 | the variant used in this thesis |
| `wen` | 0.0005 | α = 0.5 | Algorithm 1 as published |

Seeds 42, 43, 44. The seeds are the point: every comparison this project has
reported so far came from a single training run, so a difference of a few points
could not be separated from what a different initialisation would have produced
anyway.

**`none` at seed 42 is the control.** It is the same configuration as
`2026-07-11_17-49_vit_s16_3200_baseline` (validation 0.5683, test 0.5633),
trained by a different loop. If it does not land close, this loop differs
somewhere and nothing else here can be trusted yet.

## Protocol

3,200 training images per class, 300 validation, 100 test — 19,200 / 1,800 /
600. The checkpoint is selected on validation; the test set is not touched until
the final evaluation. Verified: the training logs of the runs being reproduced
contain no reference to the test set, and hashing the whole dataset found no
duplicate group spanning any partition and the test set, with exactly one
duplicate pair spanning training and validation out of 1,800 images.

ViT-S/16 from scratch, 21,691,014 parameters, 256×256, 50 epochs, batch 64,
AdamW at 3e-4 with weight decay 0.05, five epochs of linear warmup then cosine
decay, label smoothing 0.1, gradient clipping at 1.0, mixed precision. Every one
of these is copied from the configuration of the run being reproduced.

## Files

| file | what it does |
|---|---|
| `config_rooms_wen.yaml` | one configuration; the runner overrides `run_name`, `center_mode` and `seed` |
| `rooms_wen_model.py` | builds the ViT by **importing** the shared architecture, not copying it |
| `rooms_wen_data.py` | room dataloaders, augmentations identical to the shared pipeline |
| `train_rooms_wen.py` | the training loop, with the three modes |
| `evaluate_rooms_wen.py` | test evaluation; writes predictions with a pixel hash per image |
| `compare_rooms_wen.py` | runs the nine arms and summarises them |
| `run_rooms_wen_colab.ipynb` | the Colab notebook |

The Vision Transformer and the centre-loss module are imported from
`vit_s16_baseline/` and `fer_wen/` rather than duplicated. Both have been
checked numerically against their references — Algorithm 1 against the formula
in the paper, and the gradient mode against the shared implementation — and a
copy is a thing that can drift out of step.

Nothing in this folder writes to any other experiment's directory.

## Running it

Open `run_rooms_wen_colab.ipynb` from GitHub, so the clone cell is the current
one, and run the cells in order.

Nine runs at roughly two hours each. Colab will disconnect before that finishes.
Every completed arm is written to Drive, and re-running the training cell skips
arms that are already finished — but only after checking that the saved
configuration still matches the current one, so an old run cannot be filed under
the current settings. `--only none` runs one mode at a time.

## Predictions carry a pixel hash

`predictions.csv` has a `sha1` column holding the SHA-1 of the decoded RGB
pixels. Paired tests need the two runs aligned image by image; aligning on row
order assumes both loops enumerated the directory identically, and aligning on
the file path assumes both runs saw the data at the same location. The hash
identifies the image itself, and is the same identity the deduplication analysis
uses.
