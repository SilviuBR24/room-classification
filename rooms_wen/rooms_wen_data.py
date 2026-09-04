"""Room dataloaders for the update-rule comparison.

The augmentations reproduce the shared pipeline exactly -- RandomResizedCrop
with scale (0.7, 1.0) and ratio (0.75, 1.3333), then a horizontal flip -- so
that the baseline arm of this experiment can reproduce the run already obtained
by the shared loop. Anything different here would make that control fail for a
reason that has nothing to do with the centre update.

The class order is taken from the config rather than from the directory
listing. Alphabetical order happens to coincide for these six names, but
relying on that would make the label mapping a property of the filesystem
instead of a property of the experiment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_transforms(image_size: int, mean: List[float], std: List[float],
                     train: bool) -> Callable:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0),
                                         ratio=(0.75, 1.3333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class RoomDataset(Dataset):
    """Images under `root/<class>/`, with the class order fixed by the caller."""

    def __init__(self, root: str | Path, class_names: List[str],
                 transform: Callable | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise SystemExit(f"dataset directory not found: {self.root}")
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform

        self.samples: List[Tuple[Path, int]] = []
        for cls in class_names:
            d = self.root / cls
            if not d.is_dir():
                raise SystemExit(f"missing class directory: {d}")
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append((f, self.class_to_idx[cls]))
        if not self.samples:
            raise SystemExit(f"no images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> Dict[str, int]:
        idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        counts: Dict[str, int] = {c: 0 for c in self.class_to_idx}
        for _, label in self.samples:
            counts[idx_to_class[label]] += 1
        return counts

    def paths(self) -> List[str]:
        return [str(p) for p, _ in self.samples]


def build_dataloaders(cfg: Dict[str, Any], pin: bool
                      ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    names = list(d["class_names"])
    size = int(m["image_size"])

    train_ds = RoomDataset(d["train_dir"], names,
                           build_transforms(size, d["norm_mean"], d["norm_std"], True))
    val_ds = RoomDataset(d["val_dir"], names,
                         build_transforms(size, d["norm_mean"], d["norm_std"], False))

    workers = int(t.get("num_workers", 2))
    common = dict(batch_size=int(t["batch_size"]), num_workers=workers,
                  pin_memory=pin, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, train_ds.class_to_idx


def build_eval_loader(cfg: Dict[str, Any], pin: bool
                      ) -> Tuple[DataLoader, RoomDataset]:
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    ds = RoomDataset(d["eval_dir"], list(d["class_names"]),
                     build_transforms(int(m["image_size"]), d["norm_mean"],
                                      d["norm_std"], False))
    workers = int(t.get("num_workers", 2))
    loader = DataLoader(ds, batch_size=int(t["batch_size"]), shuffle=False,
                        num_workers=workers, pin_memory=pin)
    return loader, ds
