"""
fer_wen_data.py
===============
Dataset and transforms for this experiment.

A private copy, byte-for-byte equivalent in behaviour to the shared dataset
used by every other experiment in this project. That equivalence is not
cosmetic: the control run in this folder has to reproduce the result obtained
by the shared training loop, and it can only do so if the data pipeline is the
same. Any difference here -- a different crop range, a different file ordering,
a different interpolation -- would show up as a difference in accuracy and would
be mistaken for an effect of the update rule.

What is replicated exactly:
  - class order taken from an explicit configured list, never from directory
    enumeration, so class indices are stable across machines;
  - file names sorted within each class folder;
  - images decoded with Pillow and converted to RGB;
  - training transform: RandomResizedCrop(size, scale=(0.7, 1.0),
    ratio=(0.75, 1.3333)), horizontal flip p=0.5, ToTensor, Normalize;
  - evaluation transform: Resize((size, size)), ToTensor, Normalize.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_transforms(image_size: int, mean: List[float], std: List[float],
                     train: bool) -> Callable:
    """Return the transform pipeline. Identical to the shared project's."""
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


class FERDataset(Dataset):
    """Folder-per-class image dataset with an explicitly fixed class order."""

    def __init__(self, root: Union[str, Path], class_names: List[str],
                 transform: Callable | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {self.root}")
        self.class_names = list(class_names)
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        missing: List[str] = []
        for name in self.class_names:
            class_dir = self.root / name
            if not class_dir.is_dir():
                missing.append(name)
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    self.samples.append(
                        (str(class_dir / fname), self.class_to_idx[name]))

        if missing:
            raise FileNotFoundError(
                f"Missing class folder(s) under {self.root}: {missing}. "
                f"Expected one folder per class: {self.class_names}")
        if not self.samples:
            raise ValueError(
                f"No images found under {self.root} with extensions {IMG_EXTENSIONS}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image {path}: {exc}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> Dict[str, int]:
        counts = {name: 0 for name in self.class_names}
        inv = {v: k for k, v in self.class_to_idx.items()}
        for _, label in self.samples:
            counts[inv[label]] += 1
        return counts


def build_dataset(root: Union[str, Path], cfg: Dict[str, Any],
                  train: bool) -> FERDataset:
    """Create a dataset from the full config, reading image_size from `model`."""
    data_cfg = cfg["data"]
    transform = build_transforms(
        image_size=int(cfg["model"]["image_size"]),
        mean=data_cfg["norm_mean"],
        std=data_cfg["norm_std"],
        train=train,
    )
    return FERDataset(root=root, class_names=data_cfg["class_names"],
                      transform=transform)


def build_dataloaders(cfg: Dict[str, Any], pin: bool
                      ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Training and validation loaders. The test set is never loaded here."""
    t = cfg["training"]
    train_ds = build_dataset(cfg["data"]["train_dir"], cfg, train=True)
    val_ds = build_dataset(cfg["data"]["val_dir"], cfg, train=False)

    workers = int(t.get("num_workers", 2))
    train_loader = DataLoader(
        train_ds, batch_size=int(t["batch_size"]), shuffle=True,
        num_workers=workers, pin_memory=pin, drop_last=False,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(t["batch_size"]), shuffle=False,
        num_workers=workers, pin_memory=pin, drop_last=False,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader, train_ds.class_to_idx
