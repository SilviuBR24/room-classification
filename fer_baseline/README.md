# Facial expression comparison (FER2013)

The Center Loss comparison repeated on a second task, as suggested by the
supervisor: **ResNet-18 with plain cross-entropy versus ResNet-18 with
cross-entropy + Center Loss**, on FER2013.

## Why this task

The room experiments produced a conditional result. Center Loss contracts the
representation of both architectures by roughly half, but improves the
scale-invariant separation measures only for ResNet-18; on the data-starved
transformer the space is rescaled rather than reorganised. The conclusion drawn
was that Center Loss can only organise a representation that is already
adequate.

FER2013 tests that conclusion on a second, unrelated dataset. A positive result
supports the interpretation; a negative one is equally reportable and would mean
the room outcome was not specific to indoor scenes.

**What this experiment does not claim.** Center Loss was proposed for face
*identity* recognition — telling people apart, across thousands of classes each
of which is visually coherent. Expression recognition is a different problem:
seven classes, wide variation within each one across different identities, and a
discriminative signal that is a subtle change of facial geometry rather than a
stable appearance. FER2013 is a related task on face images, not the original
domain of the method, and the experiment should be read as a further test of the
hypothesis rather than as a reproduction of the setting Center Loss was designed
for.

## Get the data

```bash
python download_fer2013.py
python make_fer_zip.py          # only needed for Colab
```

The first script writes `datasets/fer2013/{train,val,eval}/<class>/*.png` and
re-checks the extracted counts against the published 28,709 / 3,589 / 3,589
split. The second packs that tree for upload to Drive.

The dataset comes from a public Parquet mirror. That format is deliberate:
several mirrors of FER2013 ship PyTorch `.pt` archives, which are pickles and
execute arbitrary code when loaded. Parquet carries no such risk.

## Choose the resolution, then run

```bash
python calibrate_fer.py --data-root ../../datasets/fer2013
python compare_fer.py --config config_fer.yaml
```

FER2013 images are 48×48, and ResNet-18 divides its input by 32 on the way to
the pooling layer:

| input | after the stem | reaching the pooling layer |
|---|---|---|
| 48×48 | 12×12 | 2×2 |
| 96×96 | 24×24 | 3×3 |
| 224×224 | 56×56 | 7×7 (the design point) |

Upsampling adds no information. What it does is stop the fixed 7×7 stride-2
stem and max-pool from discarding most of the image before any residual block
runs. Whether that is worth the extra computation is measured rather than
assumed — `calibrate_fer.py` trains briefly at each candidate resolution and
reports what each reached. Set `model.image_size` from its result.

The alternative would be to replace the stem with a 3×3 stride-1 convolution,
as CIFAR-style ResNets do. That was rejected: it would no longer be the same
ResNet-18 used for the room comparison, and comparability across the two tasks
is the point of this experiment.

## What differs from the room experiments

Only the task. Split roles, optimiser, learning-rate schedule, epochs, batch
size, label smoothing, gradient clipping, Center Loss coefficient, centre
learning rate and seed are all identical, and the training loop is literally the
same code. Resolution is the one unavoidable exception, since the images are a
different size.

**The classes are imbalanced.** This is the other substantive difference:

| class | training images |
|---|---|
| happy | 7,215 |
| neutral | 4,965 |
| sad | 4,830 |
| fear | 4,097 |
| angry | 3,995 |
| surprise | 3,171 |
| **disgust** | **436** |

A ratio of about 16 to 1, where the room dataset had exactly 3,200 per class.
Overall accuracy is therefore not sufficient on its own: a model that never
predicts *disgust* forfeits only about 1.5% of it. Per-class accuracy and
macro-averaged F1 are reported alongside, and `compare_fer.py` prints both.

The imbalance also interacts with the implemented Center Loss variant. The
centre update is normalised by mini-batch size rather than by the number of
samples of that class in the batch, so with a batch of 64 the *disgust* centre
receives an update roughly sixteen times less often than the *happy* centre.
On the balanced room dataset that deviation from the original algorithm could
not show itself; here it can be measured.

## Files

| File | Purpose |
|---|---|
| `download_fer2013.py` | fetch and unpack the dataset |
| `make_fer_zip.py` | pack it for Colab, with Linux-safe path separators |
| `config_fer.yaml` | every parameter, shared by both variants |
| `fer_model.py` | the model wrapper (the only model code here) |
| `calibrate_fer.py` | short trials to pick the input resolution |
| `train_fer.py` | train one variant |
| `evaluate_fer.py` | evaluate a checkpoint on the test set |
| `compare_fer.py` | **one command: both variants + summary** |
| `shared_infrastructure.py` | puts `vit_s16_baseline/` on `sys.path` |

## How this relates to the other folders

This folder owns **one thing: its own model wrapper.** The training loop,
dataset, checkpointing, logging, Center Loss and the evaluation script are
imported unmodified from `vit_s16_baseline/`, so the facial expression runs go
through the same code as every other experiment in this work.

It does **not** import from `resnet18_baseline/`, even though the two model
wrappers are nearly identical. Coupling them would mean a change to one could
break the other, and the small duplication is a fair price for that isolation.

Consequently this folder does not work on its own: it expects
`vit_s16_baseline/` to sit next to it. Nothing in that folder is modified.

## Known wart

The shared `train.py` prints a hard-coded line in `train.log`:

```
Model: ViT-S/16 from scratch | trainable params: 11,180,103
```

It says **ViT-S/16 even though a ResNet was built**, because that string is
fixed inside the shared trainer and this folder deliberately does not modify
`vit_s16_baseline/`. The parameter count in the same line gives it away, and
`arch: resnet18_fer` in `config_used.yaml` — copied into the run folder and
stored inside every checkpoint — is the authoritative record.
