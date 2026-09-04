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
| **wen** (Equation (4) and Algorithm 1 of Wen et al.) | a dedicated update with its own rate `α`, independent of λ, normalised per class by `(1 + n_j)`. Displacement ∝ `α / (1 + n_j)` |

Neither is adjusted to make the comparison neater. Each is run exactly as it is
defined, because the question is which of the two, as used, is better.

## What this experiment can and cannot show

The ratio of the two displacements is `α·B / (η_c·λ·(1 + n_j))`.

On FER2013 the ratio varies with class frequency. The experiment ran on the
deduplicated split (27,182 training images, 366 `disgust` against 7,035
`happy`), where at each class's expected count in a batch of 64 it is about
**68,753** for `disgust` against about **7,288** for `happy` — a nine-fold
spread. On the original split the same figures are 64,910 and 7,492. The FER result is *consistent with*
that mechanism; that experiment did not isolate it, so it is not evidence that
the per-class normalisation is what produced the effect.

The room dataset is balanced: 3,200 images in every class. That removes the
systematic difference between classes, but it does not make the ratio constant.
`n_j` is hypergeometric, with mean 10.67 and standard deviation 2.98 at
`B = 64`, so the ratio is about **10,971×** at the mean and moves between
roughly **8,741×** and **14,729×** within one standard deviation, batch by
batch and class by class.

So balancing removes the systematic frequency difference between classes; it
does not remove the per-batch fluctuation, and the Wen rule still normalises
per class. **This experiment does not separate the normalisation from the large
difference in effective rate.** It compares the two complete rules at the
hyperparameters each is used with, which is the question being asked. Each run
logs both the theoretical ratio and the centre displacements it actually
produced.

## The arms

| mode | λ | rate | what it is |
|---|---|---|---|
| `none` | — | — | cross-entropy only |
| `gradient` | 0.0005 | η_c = 0.5 | the variant used in this thesis |
| `wen` | 0.0005 | α = 0.5 | the centre update of Equation (4) / Algorithm 1 |

Seeds 42, 43, 44. The seeds are the point: every comparison this project has
reported so far came from a single training run, so a difference of a few points
could not be separated from what a different initialisation would have produced
anyway.

**`none` at seed 42 is the control.** It is the same configuration as
`2026-07-11_17-49_vit_s16_3200_baseline` (validation 0.5683, test 0.5633),
trained by a different loop. This is a consistency check: it should land close,
and an important difference must be investigated before the other arms are
interpreted. Bit-for-bit equality is not expected — the historical run used a
different runtime, GPU and PyTorch build, and PyTorch does not guarantee
identical results across those.

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
| `verify_rooms_wen.py` | the pre-flight checks; run it before starting |
| `run_rooms_wen_colab.ipynb` | the Colab notebook |

The Vision Transformer and the centre-loss module are imported from
`vit_s16_baseline/` and `fer_wen/` rather than duplicated. Both have been
checked numerically against their references — Algorithm 1 against the formula
in the paper, and the gradient mode against the shared implementation — and a
copy is a thing that can drift out of step.

Nothing in this folder writes to any other experiment's directory.

## Running it

Run `python verify_rooms_wen.py` first. It checks the properties the comparison
depends on — that the three arms see the same images in the same order, that a
skipped mixed-precision step does not move the Wen centres, that the scheduler
produces the intended learning rates — and builds its own tiny dataset, so it
needs neither the real split nor a GPU.

Then open `run_rooms_wen_colab.ipynb` from GitHub and run the cells in order.

**The notebook pins one commit.** Nine runs will span several Colab sessions,
and if `main` moved between them the arms would be produced by different code
while the configuration looked identical. `PINNED_COMMIT` in the clone cell
fixes that; the cell refuses to continue if HEAD is not that commit or the tree
is not clean.

Nine runs at roughly two hours each. Colab will disconnect before that finishes.
Every completed arm is written to Drive, and re-running the training cell skips
arms that are already finished. An arm is only reused when its whole saved
configuration matches, when it records the commit it was built from and that
tree was clean, and when its checkpoint still carries the validation accuracy.
The summary is rebuilt from what is on disk, so it covers all nine arms whether
they were produced in this session, after a reconnection, or through several
`--only` invocations; if the arms turn out to come from different commits or
different GPUs, it says so before you compare them.

## What each run records

Beyond the usual metrics, every epoch logs the mean and maximum norm of the
centres, the mean and maximum distance they moved, the raw centre term and its
λ-weighted contribution, and the number of mixed-precision steps the scaler
skipped. `logs/provenance.json` records the commit, whether the tree was dirty,
and the PyTorch, CUDA, cuDNN and GPU versions.

The displacement ratio written at the start of a run is a property of the
hyperparameters at `E[n_j]`, not a measurement: once the arms diverge they see
different embeddings and `n_j` varies per batch. The measured displacements are
the ones in the per-epoch log.

## Predictions carry a pixel hash

`predictions.csv` has a `sha1` column holding the SHA-1 of the decoded RGB
pixels. Paired tests need the two runs aligned image by image; aligning on row
order assumes both loops enumerated the directory identically, and aligning on
the file path assumes both runs saw the data at the same location. The hash
identifies the image itself, and is the same identity the deduplication analysis
uses.
