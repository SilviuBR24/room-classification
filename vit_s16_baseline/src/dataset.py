"""
src/dataset.py
==============
Folder-based dataset (ImageFolder-style) with a FIXED, explicit class order.

Why not plain torchvision.ImageFolder?
    ImageFolder derives class indices by alphabetically sorting folder names.
    That would silently break if a folder were renamed or missing. Here the
    class -> index mapping comes straight from the config's `class_names`
    list, so the mapping is guaranteed identical across train / eval / resume.

Expected directory layout (same for train and eval):
    root/
        bathroom/      img001.jpg ...
        bedroom/       ...
        dining_room/   ...
        entrance_hall/ ...
        kitchen/       ...
        living_room/   ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Image extensions we accept.
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------
def build_transforms(
    image_size: int,
    mean: List[float],
    std: List[float],
    train: bool,
) -> Callable:
    """Return the transform pipeline.

    Train: RandomResizedCrop + horizontal flip (light, label-preserving aug).
    Eval:  deterministic resize to image_size (images are already 256x256,
           so this is effectively identity + tensor/normalize).
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.7, 1.0), ratio=(0.75, 1.3333)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class RoomClassificationDataset(Dataset):
    """Image dataset with an externally supplied, fixed class mapping."""

    def __init__(
        self,
        root: Union[str, Path],
        class_names: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {self.root}")

        self.class_names = list(class_names)
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }
        self.transform = transform

        # Build (path, label) list, validating each class folder.
        self.samples: List[Tuple[str, int]] = []
        missing: List[str] = []
        for name in self.class_names:
            class_dir = self.root / name
            if not class_dir.is_dir():
                missing.append(name)
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    self.samples.append((str(class_dir / fname), self.class_to_idx[name]))

        if missing:
            raise FileNotFoundError(
                f"Missing class folder(s) under {self.root}: {missing}. "
                f"Expected one folder per class: {self.class_names}"
            )
        if len(self.samples) == 0:
            raise ValueError(
                f"No images found under {self.root} with extensions {IMG_EXTENSIONS}."
            )

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


# ----------------------------------------------------------------------
# DataLoader builders
# ----------------------------------------------------------------------
def build_dataset(
    root: Union[str, Path],
    cfg: Dict[str, Any],
    train: bool,
) -> RoomClassificationDataset:
    """Create a dataset for `root` using config-defined classes / normalization."""
    data_cfg = cfg["data"]
    image_size = cfg["model"]["image_size"]
    transform = build_transforms(
        image_size=image_size,
        mean=data_cfg["norm_mean"],
        std=data_cfg["norm_std"],
        train=train,
    )
    return RoomClassificationDataset(
        root=root, class_names=data_cfg["class_names"], transform=transform
    )


def build_dataloaders(
    cfg: Dict[str, Any], device: torch.device
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Build train and eval DataLoaders plus the class_to_idx mapping."""
    train_ds = build_dataset(cfg["data"]["train_dir"], cfg, train=True)
    eval_ds = build_dataset(cfg["data"]["eval_dir"], cfg, train=False)

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    pin = device.type == "cuda"
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=persistent,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,  # keep order so predictions align with dataset.samples
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=persistent,
    )
    return train_loader, eval_loader, train_ds.class_to_idx
