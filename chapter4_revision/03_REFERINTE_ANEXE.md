# The eleven annex references

The regulation is explicit, in both language versions:

> *Fiecare anexă se va menționa cel puțin o dată în textul lucrării, ca
> referință.*
> *Each annex will be mentioned at least once in the text as a reference.*

**The bachelor thesis did not do this.** Its eighty-five pages carry eight
annexes starting at page 66, and the body — pages 1 to 65 — does not contain the
word "Anexa" once. Checked against the final PDF, not from memory.

Below, one sentence per annex, each placed in the subsection that already
discusses what the annex implements. They are written to read as part of the
surrounding prose rather than as a list of pointers.

**Before any of them will compile**, add this to the thesis preamble:

```latex
\usepackage{listings}
\usepackage{xcolor}
```

and give each annex a label when the annex file is merged, so `\ref{}` resolves.
Until then, write the numbers literally (`Annex~1`) and convert to `\ref{}`
afterwards if you prefer.

---

## 1 — `make_split.py`

**Where:** §3.2.3, `\label{subsec:dataset}`, line 1510, at the end of the
paragraph that describes the 3,200 / 300 / 1,500 / 100 partitioning.

> The partitioning is deterministic: a fixed seed drives one shuffle per
> category, and the labelled and unlabelled boundaries are index ranges of that
> shuffle, so reducing the training subset moves images into validation without
> changing which images are unlabelled. The script that produces it is given in
> Annex~1.

## 2 — `dataset.py`

**Where:** §3.2.3, `\label{subsec:dataset}`, after the augmentation description.

> The class order is fixed in the configuration rather than inferred from the
> directory listing, so that the mapping from category to index is a property of
> the experiment and not of the filesystem; the loader and the augmentation
> pipeline are listed in Annex~2.

## 3 — `vit.py`

**Where:** §3.2.4, `\label{subsec:model_architectures}`, line 1574, after the
parameter count.

> The architecture is implemented from first principles rather than imported
> from a model library: patch embedding, the classification token, the encoder
> blocks and the classification head are written out in full in Annex~3.

## 4 — `center_loss.py`

**Where:** §3.2.5, `\label{subsec:loss_functions}`, line 1635, immediately after
the equation that defines the objective.

> The implementation of this objective, including the treatment of the class
> centres as parameters of the combined loss, is given in Annex~4.

## 5 — `trainer.py`

**Where:** §3.2.6, `\label{subsec:training_procedure}`, line 1698, after the
schedule is described.

> The training engine, which applies this schedule, maintains mixed precision
> and selects the reported checkpoint on validation accuracy alone, is listed in
> Annex~5.

## 6 — `pseudo_label.py`

**Where:** §3.2.8, `\label{subsec:self_training}`, line 1762, where the two
selection rules are introduced.

> Both selection rules -- a fixed confidence threshold and a target coverage
> computed per round -- are implemented in Annex~6.

## 7 — `semi_trainer.py`

**Where:** §3.2.8, `\label{subsec:self_training}`, after the description of the
labelling round.

> The engine that interleaves labelling with training, so that unannotated
> images are annotated by the network as it learns rather than in a separate
> offline pass, is listed in Annex~7.

## 8 — `evaluate.py`

**Where:** §3.2.9, `\label{subsec:performance_evaluation}`, line 1844, where the
metrics are introduced.

> Accuracy, per-class accuracy, the confusion matrices and the saved embeddings
> are all produced by the evaluation script given in Annex~8.

## 9 — `analyze_embeddings.py`

**Where:** §4.6, `\label{sec:results_geometry}`, line 2340, in the paragraph that
introduces the four measures.

> All four quantities are computed in the original embedding space, not in any
> projection of it; the script that computes them, together with the projections
> used for the figures, is listed in Annex~9.

## 10 — `mnist_center_loss.py`

**Where:** §4.2, `\label{sec:results_mnist}`, line 2028, at the start of the
validation section.

> The validation network, its two-dimensional embedding layer and the
> measurements reported below are given in Annex~10.

## 11 — `run_colab.ipynb`

**Where:** §4.1, `\label{sec:results_protocol}`, line 2008, after the statement
of the shared configuration.

> The notebook that ran these experiments -- obtaining the code, restoring the
> partitioned dataset, and driving the coefficient sweep with validation-based
> selection -- is reproduced in Annex~11, so that the reported runs can be
> repeated.

---

## Check afterwards

After inserting all eleven, this should return eleven matches:

```bash
grep -c "Annex~" Lupu_Silviu-George_COMPLET_cap1-4.tex
```

If any annex is added or removed later, its sentence has to move with it. An
annex without a reference is the one defect in this section that the regulation
names explicitly.
