# ViT-S/16 Indoor-Room Classifier — Supervised Baseline + Center Loss

A from-scratch Vision Transformer (ViT-S/16) in PyTorch for 6-class indoor
room classification (`bathroom, bedroom, dining_room, entrance_hall, kitchen,
living_room`) on a filtered Places365 subset.

> **Scope:** supervised baseline **and Center Loss** (Part 2), with robust
> checkpointing, logging, resume, and evaluation. Center Loss is opt-in via
> `use_center_loss` in the config. Pseudo-labeling / self-labeling (Part 3) is
> **not implemented yet** — the model exposes CLS embeddings
> (`model(x, return_embeddings=True)`), which Center Loss uses and Part 3 will.
>
> **Data protocol:** training selects the best model on a **validation** set
> (`data.val_dir`); `evaluate.py` reports on a held-out **test** set
> (`data.eval_dir`). If `val_dir` is omitted, validation falls back to the test
> set (legacy single-split — avoid it for final numbers, it leaks the test set
> into model selection).
>
> **Center Loss note:** the class-center update is coupled to the loss weight,
> so the *effective* center learning rate is `center_loss_lr * center_loss_weight`.
> A `center_loss_weight` (lambda) sweep is therefore partly confounded by how
> fast the centers move.

## Project layout
```
code/
├── train.py          # entry point: training + resume
├── evaluate.py       # entry point: evaluate a checkpoint
├── config.yaml       # all hyperparameters / paths
├── requirements.txt  # extra deps (torch is preinstalled on Colab)
├── src/
│   ├── dataset.py    # fixed-class-mapping dataset + transforms
│   ├── vit.py        # from-scratch ViT (patch embed, MHSA, blocks, head)
│   ├── center_loss.py # Center Loss: learnable per-class centers (Part 2)
│   ├── trainer.py    # epoch loop, AMP, scheduler, checkpointing
│   ├── checkpoint.py # save/load: last / best / epoch / interrupted
│   ├── logger.py     # train.log + metrics.csv
│   └── utils.py      # seeding, device, config IO, AMP, metrics
├── checkpoints/  logs/  runs/  outputs/   # local defaults / placeholders
```

## Dataset layout (train and eval the same)
```
<train_dir>/
    bathroom/  bedroom/  dining_room/  entrance_hall/  kitchen/  living_room/
```

## Google Colab — quickstart
```python
from google.colab import drive
drive.mount('/content/drive')

# Put the code/ folder on Drive (or upload it), then:
%cd /content/drive/MyDrive/code
!pip install -r requirements.txt          # torch/torchvision already present

# Edit config.yaml so data.train_dir / data.eval_dir / paths.output_root
# point at your Drive folders, then train:
!python train.py --config config.yaml
```
Set the runtime to **GPU** (Runtime → Change runtime type → GPU). Point
`paths.output_root` at Google Drive so runs survive disconnects.

## Resume after an interruption
Each run lives in `<output_root>/<timestamp>_<run_name>/`. To continue from
where it stopped (the **next** epoch, same run folder):
```bash
python train.py --resume <output_root>/<run>/checkpoints/last_checkpoint.pt
```
If the session crashed mid-epoch, use `interrupted_checkpoint.pt` instead.

## Evaluate the best model
```bash
python evaluate.py --checkpoint <output_root>/<run>/checkpoints/best_model.pt
# add --save-embeddings to dump CLS embeddings for later t-SNE plots
```
Outputs (confusion matrix PNG/CSV, classification report, predictions.csv,
metrics.txt) are written to `<run>/outputs/eval_<timestamp>/`.

## Key assumptions
- Folder-per-class dataset; class index = position in `config.data.class_names`
  (which is alphabetical, so it matches ImageFolder ordering).
- Images are RGB 256×256; eval transform is a deterministic resize.
- ImageNet mean/std are used purely as input normalization (no pretraining).
- AMP is active only on CUDA; on CPU it is a no-op (same code path).
- Resume restores model/optimizer/scheduler/AMP/RNG state; full bit-for-bit
  GPU determinism is not guaranteed (standard for CUDA).
- `eval_every` controls evaluation frequency; "best" tracks eval accuracy.
