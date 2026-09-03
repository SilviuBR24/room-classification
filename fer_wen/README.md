# Center Loss: does the update rule matter?

One question, measured on FER2013: **the centres can be updated two different
ways — does the choice change the outcome?**

| | Centres | Displacement normalised by | Rate depends on λ |
|---|---|---|---|
| **`gradient`** | learnable parameters, stepped by SGD on the combined objective | mini-batch size *B* | **yes**, effective rate = η·λ |
| **`wen`** | a buffer, updated by Algorithm 1 | **1 + n_j**, per class | no, α is independent |

The first is what most public implementations do, and what the rest of this
project does. The second is what the paper specifies (Wen, Zhang, Li, Qiao,
*A Discriminative Feature Learning Approach for Deep Face Recognition*,
ECCV 2016, Equations 2–5 and Algorithm 1).

## Why FER2013

Class imbalance makes the frequency-dependent consequence of the two rules
easier to observe. FER2013 has 7,215 training images for *happy* and 436 for
*disgust*, about 16 to 1.

To be precise about what imbalance does and does not change: the overall
difference in update rate would be present on any dataset, balanced or not.
What imbalance adds is a systematic relationship between how often a class
appears and how far its centre moves — and that relationship is what can be
measured here, and could not be identified on the room dataset where every
class had exactly 3,200 images. Note also that a balanced dataset does not give
balanced mini-batches, so even there the two rules would not be strictly
identical.

Measured on a single deliberately imbalanced batch, the two rules order the
classes in opposite directions:

| class | samples in batch | `gradient` displacement | `wen` displacement |
|---|---|---|---|
| A | 40 | 0.00227 | 7.07 |
| B | 15 | 0.00133 | 10.66 |
| C | 8 | 0.00103 | 14.60 |
| D | **1** | **0.00036** | **22.77** |
| E | 0 | 0 | 0 |

Under the gradient rule the rarest class moves its centre six times *less* than
the most frequent. Under Algorithm 1 it moves three times *more*, because each
centre is drawn towards the mean of its own class regardless of how many samples
carry it.

## Read the result carefully

**Two things differ between the modes, not one.** Both follow from the same
design choice — whether λ enters the centre update — but they are separate
effects and the comparison cannot separate them:

1. **Per-class normalisation.** Dividing by `1 + n_j` instead of by `B`.
2. **Overall rate.** The gradient mode's effective centre rate is η·λ =
   0.5 × 0.0005 = 0.00025, while Algorithm 1 applies α = 0.5 directly.

The ratio of the two is often quoted as α/(η·λ) = 2,000, but that compares the
rate parameters, not the displacements the rules actually produce. Including
both normalisations, the ratio is

    α·B / (η·λ·(1 + n_j))  =  128,000 / (1 + n_j)   at B = 64

which depends on the class:

| samples of the class in the batch | ratio of displacements |
|---|---|
| 40 | 3,122× |
| 15 | 8,000× |
| 8 | 14,222× |
| 1 | 64,000× |

This also means a single reduced α cannot equalise the two rules across all
classes at once: whatever value matched a frequent class would still leave a
rare one far apart.

A result showing that Algorithm 1 behaves differently therefore says the two
implementations differ, not that the per-class normalisation alone is
responsible. Isolating the normalisation would need a third run with α reduced
to match the effective rate. That run is not included here.

## Run it

```bash
python compare_wen.py --config config_wen.yaml
```

Roughly 35 minutes per variant on a T4, so about 70 minutes for both. On Colab,
open `run_wen_colab.ipynb`.

Order matters: the `gradient` variant runs first because it is the **control**.
This folder has its own training loop, so that run has to reproduce what the
shared loop already produced in `fer_baseline/` (test accuracy 0.6807). If it
does not, the loop differs in some way beyond the update rule and the second
result cannot be attributed to Algorithm 1. `compare_wen.py` prints that check
explicitly rather than leaving it to the reader.

## What is held constant

Everything except the update rule: λ = 0.0005, α = η = 0.5, the architecture,
the resolution, the split, the optimiser, the schedule, the number of epochs,
the batch size, the label smoothing, the gradient clipping and the seed,
including the cuDNN determinism settings.

Both modes also use the same scalar loss formula. Given identical centre
values they produce identical losses and identical gradients to the
embeddings — verified as exactly zero difference. After the first centre
update the centres differ, and from then on so do the embedding gradients.
That divergence is the intended consequence of the rule under test, not a
confound: measured after one differing update, the centres differ by up to
0.558 and the next batch's embedding gradients by up to 0.140.

## Files

| File | Purpose |
|---|---|
| `wen_center_loss.py` | both update rules, selected by `mode` |
| `fer_wen_model.py` | ResNet-18 with an observable embedding |
| `fer_wen_data.py` | dataset and transforms |
| `config_wen.yaml` | every parameter, shared by both variants |
| `train_wen.py` | train one variant |
| `evaluate_wen.py` | evaluate on the test set, plus centre alignment |
| `compare_wen.py` | **one command: both variants + control check** |
| `run_wen_colab.ipynb` | Colab runner |

## Isolation

The training and evaluation implementation in this folder is standalone: no
`.py` file here imports from `vit_s16_baseline/`, `resnet18_baseline/` or
`fer_baseline/`. That is deliberate and also necessary, since the shared
training loop steps the centres through an optimiser on the combined
objective — precisely the behaviour under test — and so cannot express
Algorithm 1.

The Colab notebook is the exception, and only for convenience: it installs the
repository-wide dependency list from `vit_s16_baseline/requirements.txt` and
reuses the existing embedding-visualisation utility from `fer_baseline/`.
Neither touches the training or evaluation path.

The cost is that the claim made elsewhere in this work — that one training loop
ran every experiment — does not cover this folder. The control run is what
replaces that guarantee, and the dataset and transform code here is a faithful
copy of the shared one so the control can succeed: the same crop range, the same
flip probability, the same normalisation, the same file ordering, the same
explicit class order.

## What to look at in the results

Accuracy is the least informative measurement here. At 3,589 test images the
standard error near 68 percent is about 0.008, so only a difference larger than
that is resolved.

The direct measurement is `centre_alignment.csv`, written into each run's
evaluation folder. It reports, per class, the cosine similarity between the
learned centre and the empirical class centroid, alongside that class's training
count. Under the gradient rule these two columns are strongly correlated — on
the existing `fer_baseline` run, r = +0.843, with *disgust* at 0.248 against
*happy* at 0.990. If Algorithm 1's per-class normalisation does what it is meant
to, that correlation should weaken.
