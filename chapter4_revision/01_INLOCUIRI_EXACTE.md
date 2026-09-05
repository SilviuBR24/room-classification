# Chapter 4 — exact replacements

Nothing in the thesis has been edited. Each item below gives the text to find
and the text to put in its place. Line numbers refer to
`Versions/Chapter_4/Lupu_Silviu-George_COMPLET_cap1-4.tex` as of the audit; use
the search text, not the line number, in case the file has since shifted.

Grouped by why the change is needed.

---

## 1. WRONG INSTRUMENT — the significance convention

**Where:** §4.1, Experimental Protocol and Reporting Conventions, around line 2019.

**Why:** the standard error of a *single* accuracy is not the right yardstick
for the *difference* between two models scored on the same 600 images. Those
two models make correlated errors, so the paired quantity has its own, smaller
standard error, and the correct test is McNemar's. The convention as written can
also mislead in the other direction: it would call a 3.3-point difference
conclusive when the paired test gives p = 0.098.

**FIND:**

```
categories. This size determines how precisely any reported accuracy can be
known. For an observed accuracy \(\hat{p}\), the standard error of the estimate
is \(\sqrt{\hat{p}(1-\hat{p})/600}\), which is approximately \(2.0\) percentage
points near \(\hat{p}=0.56\) and approximately \(1.6\) percentage points near
\(\hat{p}=0.82\). Differences smaller than this margin are reported as
inconclusive rather than as improvements, and this convention is applied
consistently throughout the chapter.
```

**REPLACE WITH:**

```
categories. This size determines how precisely any reported accuracy can be
known. For an observed accuracy \(\hat{p}\), the standard error of the estimate
is \(\sqrt{\hat{p}(1-\hat{p})/600}\), which is approximately \(2.0\) percentage
points near \(\hat{p}=0.56\) and approximately \(1.6\) percentage points near
\(\hat{p}=0.82\). That figure describes a single accuracy. Where two models are
compared, they are evaluated on the same images and therefore make correlated
errors, so the comparison is treated as paired: only the images on which the two
models disagree carry information, and the exact McNemar test is applied to
them. Reported \(p\)-values are two-sided. Where a family of comparisons is made
against a common baseline, the Holm correction is applied within that family and
both the raw and the corrected value are given.

A second and larger source of uncertainty is not captured by either figure. Two
runs of an identical configuration -- the same architecture, the same
hyperparameters and the same seed, differing only in the order in which the
training images were presented and in the hardware on which they were trained --
produced test accuracies of \(56.33\) and \(59.83\) percent, a difference of
\(3.50\) percentage points on which the two models disagreed for \(109\) of the
\(600\) images \((p = 0.055)\). Their validation accuracies, measured on three
times as many images, agreed to within \(0.55\) percentage points. Run-to-run
variation of this magnitude exceeds every difference attributed to a method in
this chapter, and it is the reason the conclusions drawn here rest on the
geometry of the learned representation rather than on classification accuracy.
```

---

## 2. TOO STRONG — the monotone ordering argument

**Where:** §4.3, Vision Transformer Results, around line 2138.

**Why:** the argument treats a monotone ordering across four runs as evidence.
The span of the entire sweep is 3.3 percentage points, and two runs of the same
configuration differ by 3.5. The ordering is therefore well inside what
run-to-run variation alone produces, and cannot carry the weight the sentence
gives it.

**FIND:**

```
Two qualifications are necessary. First, the individual differences between
adjacent coefficients are small relative to the standard error of roughly two
percentage points, so no single pairwise comparison in this table is
individually conclusive. What is informative is the consistency of the
direction: the ordering is monotone across four independent runs and agrees
between the validation and test partitions. Second, as established in
```

**REPLACE WITH:**

```
Three qualifications are necessary. First, no single pairwise comparison in this
table is individually conclusive: the paired McNemar tests against the
Cross-Entropy baseline give \(p = 0.509\), \(p = 0.235\) and \(p = 0.098\) for
the three coefficients in ascending order, and after Holm correction across the
three comparisons none is below \(0.29\). Second, the ordering is monotone and
agrees between the two partitions, but that observation cannot be treated as
evidence in its own right: the whole sweep spans \(3.3\) percentage points,
while two runs of an identical configuration were found to differ by \(3.5\)
(Section~\ref{sec:results_protocol}). The monotone ordering is therefore
consistent with run-to-run variation alone. Third, as established in
```

---

## 3. STALE — a sentence written for an earlier draft

**Where:** §4.7, Limitations, around line 2478.

**Why:** this *is* the final version, and the seed-repetition study is no longer
absent.

**FIND:**

```
Finally, the reduced-label configuration described in
Subsection~\ref{subsec:dataset} is not reported here. Its results, together with
the seed-repetition study and the embedding analysis of the self-labelled
models, are deferred to the final version of this work.
```

**REPLACE WITH:**

```
Finally, the reduced-label configuration described in
Subsection~\ref{subsec:dataset} and the embedding analysis of the self-labelled
models are outside the scope of this study, and are left as directions for
further work.
```

---

## 4. NOW OUT OF DATE — the single-seed limitation

**Where:** §4.7, Limitations, the paragraph beginning "The test partition of six
hundred images".

**Why:** the paragraph says the variation attributable to initialisation and
data ordering "was therefore not measured". It has since been measured. The
limitation is still real, but its shape has changed: the variation is now
quantified, and it is larger than the effects the chapter discusses.

**FIND:**

```
The test partition of six hundred images gives a standard error of roughly two
percentage points, so differences of that magnitude cannot be resolved. Every
configuration was trained once, with a single seed, and the variation
attributable to initialisation and data ordering was therefore not measured.
Repeating the principal comparisons across several seeds and reporting the mean
and spread would be required before the smaller effects reported here could be
claimed with confidence.
```

**REPLACE WITH:**

```
The test partition of six hundred images gives a standard error of roughly two
percentage points, so differences of that magnitude cannot be resolved from a
single pair of runs. The coefficient sweep and the self-labelling study were
each trained once, with a single seed. The variation that a different
initialisation and a different presentation order produce has since been
measured directly, and it is larger than any effect those studies report: two
runs of an identical configuration differed by \(3.50\) percentage points on the
test partition. The smaller differences reported in this chapter therefore
describe the particular runs that produced them, and are not claimed as
properties of the training procedure.
```

---

## Summary

| # | kind | risk if left |
|---|---|---|
| 1 | wrong instrument | an examiner who knows McNemar will ask why it was not used |
| 2 | too strong | the one claim in the chapter that the data does not support |
| 3 | stale | reads as an unfinished draft |
| 4 | out of date | understates work that was actually done |

Item 2 is the only one that would be a substantive error if left. Items 1 and 4
are upgrades: they replace a defensible but weaker treatment with a stronger
one, using measurements that did not exist when the text was written.
