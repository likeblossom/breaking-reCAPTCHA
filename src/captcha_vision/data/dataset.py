"""
Dataset utilities for reCAPTCHA tile classification.

Images have mixed sizes (100×100 – 224×224) and mixed modes (RGB, RGBA).
Everything is normalised to 224×224 RGB.  A WeightedRandomSampler handles
the severe class imbalance (Mountain: 14 vs Car: 3892).

Single-scan design
------------------
``build_dataloaders`` calls ``ImageFolder`` exactly once.  Train and val
subsets are created with ``_TransformDataset``, a lightweight wrapper that
holds references to the same in-memory sample list and applies different
transforms per access.  Avoids three redundant directory scans.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------

def _rgba_to_rgb(path: str) -> Image.Image:
    """Load an image as RGB, compositing RGBA onto white if needed."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Transform-aware subset — avoids re-scanning the directory
# ---------------------------------------------------------------------------

class _TransformDataset(Dataset):
    """
    A view into an ``ImageFolder`` that applies a given transform.

    The parent ``ImageFolder`` is loaded once without any transform so its
    ``samples`` list can be shared across the train and val subsets.
    Each subset wraps the same source object but carries its own transform,
    which is applied on-the-fly in ``__getitem__``.
    """

    def __init__(
        self,
        source: datasets.ImageFolder,
        indices: list[int],
        transform: transforms.Compose,
    ) -> None:
        self._source = source
        self._indices = indices
        self._transform = transform

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self._source.samples[self._indices[idx]]
        img = self._source.loader(path)
        return self._transform(img), label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(train: bool) -> transforms.Compose:
    """
    Return the transform pipeline for train or eval mode.

    Training augmentations are tuned for reCAPTCHA tiles, especially the 4x4
    variant where each tile is a small crop of a larger scene:

      - ``RandomResizedCrop`` with scale down to 0.4 simulates the partial-
        object fragments the model sees on 4x4 grids.
      - ``RandomPerspective`` mimics the slight keystone distortion in browser-
        rendered grid cells.
      - ``GaussianBlur`` approximates JPEG compression artefacts common in
        reCAPTCHA images.
      - ``RandomErasing`` acts as spatial dropout, forcing the model to use
        distributed features rather than relying on one salient patch.

    Eval/test: Deterministic resize + normalise only.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    IMAGE_SIZE, scale=(0.4, 1.0), ratio=(0.75, 1.33),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(15),
                transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
                ),
                transforms.RandomGrayscale(p=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                transforms.ToTensor(),
                transforms.Normalize(_MEAN, _STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.25)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )


# ---------------------------------------------------------------------------
# Class weights (for loss function)
# ---------------------------------------------------------------------------

def build_class_weights(
    dataset: datasets.ImageFolder,
    indices: list[int],
    smoothing: str = "sqrt",
) -> torch.Tensor:
    """
    Compute class weights for CrossEntropyLoss / FocalLoss.

    Raw inverse-frequency produces extreme ratios on highly imbalanced datasets
    (e.g. 276× for Car vs Mountain here), which causes the model to collapse on
    the dominant class.  ``smoothing`` controls how aggressively the imbalance
    is corrected:

    ``"sqrt"``
        Square-root of inverse frequency.  Reduces a 276× ratio to ~17×.
        Good default; no new hyperparameters.
    ``"inv"``
        Raw inverse frequency.  Maximum correction; use only when classes are
        mildly imbalanced (<10×).
    ``"cbf"``
        Effective-number-of-samples weighting (β=0.9999).  From the
        Class-Balanced Loss paper (Cui et al., 2019).  Principled smoothing
        that also works well at extreme ratios.

    In all cases weights are mean-normalised so the loss magnitude stays
    comparable to the unweighted baseline.
    """
    num_classes = len(dataset.classes)
    counts = torch.zeros(num_classes)
    for idx in indices:
        counts[dataset.samples[idx][1]] += 1
    counts = counts.clamp(min=1.0)

    if smoothing == "inv":
        weights = 1.0 / counts
    elif smoothing == "cbf":
        beta = 0.9999
        weights = (1.0 - beta) / (1.0 - beta ** counts)
    else:  # "sqrt" — default
        weights = 1.0 / counts.sqrt()

    return weights / weights.mean()


def build_class_weights_for_split(
    data_dir: str | Path,
    val_split: float = 0.15,
    seed: int = 42,
    smoothing: str = "sqrt",
) -> torch.Tensor:
    """
    Convenience wrapper: scan ``data_dir/train`` and return class weights
    for the training split.  Useful when the caller doesn't hold a reference
    to the internal ``ImageFolder`` produced by ``build_dataloaders``.
    """
    train_dir = Path(data_dir) / "train"
    index_ds = datasets.ImageFolder(train_dir, loader=_rgba_to_rgb)
    n = len(index_ds)
    all_labels = [s[1] for s in index_ds.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, _ = next(sss.split(range(n), all_labels))
    return build_class_weights(index_ds, train_idx.tolist(), smoothing=smoothing)


# ---------------------------------------------------------------------------
# Weighted sampler (for DataLoader)
# ---------------------------------------------------------------------------

def _make_weighted_sampler(
    dataset: datasets.ImageFolder,
    indices: list[int],
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler so each class is seen equally often."""
    num_classes = len(dataset.classes)
    counts = torch.zeros(num_classes)
    for idx in indices:
        counts[dataset.samples[idx][1]] += 1
    class_weights = 1.0 / counts.clamp(min=1.0)
    sample_weights = torch.tensor(
        [class_weights[dataset.samples[i][1]] for i in indices]
    )
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build train, validation, and test DataLoaders.

    ``data_dir/train`` is scanned exactly once.  Train and val subsets share
    the same ``ImageFolder`` object via ``_TransformDataset``; their transforms
    differ but no second directory scan is performed.

    The split uses ``StratifiedShuffleSplit`` so every class — including tiny
    ones like Mountain (14 samples) — appears in both train and val.

    Returns
    -------
    train_loader, val_loader, test_loader, class_names
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"

    index_ds = datasets.ImageFolder(train_dir, loader=_rgba_to_rgb)
    class_names = index_ds.classes
    n = len(index_ds)
    all_labels = [s[1] for s in index_ds.samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(sss.split(range(n), all_labels))
    train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        _TransformDataset(index_ds, train_idx, get_transforms(train=True)),
        batch_size=batch_size,
        sampler=_make_weighted_sampler(index_ds, train_idx),
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        _TransformDataset(index_ds, val_idx, get_transforms(train=False)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = build_test_loader(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, class_names


def build_test_loader(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """Build only the test DataLoader (skips the train/val split overhead)."""
    test_dir = Path(data_dir) / "test"
    test_ds = datasets.ImageFolder(
        test_dir, transform=get_transforms(train=False), loader=_rgba_to_rgb
    )
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
