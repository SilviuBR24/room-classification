# ViT-S/16 Indoor-Room Classifier — Supervised Baseline (Step 1)

A from-scratch Vision Transformer (ViT-S/16) in PyTorch for 6-class indoor
room classification (`bathroom, bedroom, dining_room, entrance_hall, kitchen,
living_room`) on a filtered Places365 subset.

> **Scope of this step:** supervised baseline only — robust checkpointing,
> logging, resume, and evaluation. **Center Loss and pseudo-labeling are NOT
> implemented yet.** The model already exposes CLS embeddings
> (`model(x, return_embeddings=True)`), so those stages plug in without
> architectural changes.

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
