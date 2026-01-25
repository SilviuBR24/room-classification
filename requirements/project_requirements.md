# Project Requirements (Technical / Research)

Scope: What the dissertation project must implement, demonstrate, and analyze.

## 1) Problem definition
- Task: classify an indoor image into one of 6 room categories:
  - bathroom, kitchen, living room, bedroom, hall, dining room
- Dataset: public dataset (must be named and cited in the thesis)

## 2) Baseline system (Transformer)
- Train a known vision transformer architecture (e.g., ViT/DeiT/Swin or equivalent).
- Perform tuning/parameterization to maximize performance:
  - learning rate and schedule
  - optimizer configuration
  - batch size
  - augmentations
  - model variant / input resolution
- Produce a stable baseline with reproducible training/evaluation.

### Acceptance evidence
- Training configuration table
- Final accuracy on test split
- Confusion matrix for baseline

## 3) Center Loss integration
- Add Center Loss to the training objective to improve embedding compactness and class separability.
- Define the combined objective:
  - `L_total = L_CE + lambda * L_center`
- Justify and document the chosen `lambda` and center update strategy.

### Acceptance evidence
- Baseline vs. Center Loss: accuracy comparison
- Confusion matrices comparison
- Embedding visualizations showing improved compactness/separability (e.g., t-SNE / UMAP)

## 4) Semi-supervised scenario (online pseudo-labeling)
- Simulate limited labeled data.
- Use unlabeled data that is annotated online by the current model (pseudo-labeling).
- Apply safeguards against label noise:
  - confidence thresholding
  - optional class balancing / filtering
  - stopping criteria

### Acceptance evidence
- Performance vs. labeled fraction (plot/table)
- Analysis of pseudo-label quality (e.g., threshold sensitivity)
- Confusion matrices in the semi-supervised setting

## 5) Evaluation and analysis
Mandatory evaluation artifacts:
- Overall accuracy
- Confusion matrices (baseline, +center loss, semi-supervised)
- Embedding/separability plots

Optional (if time allows, but strongly recommended):
- per-class precision/recall/F1
- ablations: lambda, threshold, augmentations

## 6) Implementation requirements
- Python implementation with reproducibility:
  - fixed seeds
  - saved configs (YAML/JSON)
  - logged metrics
  - saved checkpoints
- Clear repo structure (train/eval scripts, requirements.txt, README).

## Definition of Done (DoD)
- A transformer baseline is trained and evaluated with reproducible results.
- Center Loss is implemented and empirically analyzed with embedding plots.
- Semi-supervised pseudo-labeling experiments are executed and analyzed.
- The thesis documents methods, results, and limitations credibly.
