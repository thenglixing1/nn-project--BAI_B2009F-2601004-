"""Data loading and preprocessing.

The same functions are used by training, testing and prediction, so a new
sample is always transformed exactly like the training data.

Pipeline for every image:
  1. fix EXIF orientation, convert to RGB (handles the CMYK image)
  2. resize to img_size x img_size
  3. (training only, optional) random augmentation
  4. scale to [0, 1] and standardise with the TRAINING-set mean and std
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Random augmentation, only ever applied to the training set.
AUGMENT = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])


def load_image_uint8(path, img_size):
    """Read one image file -> uint8 array of shape (img_size, img_size, 3)."""
    with Image.open(path) as im:
        if im.format == "JPEG" and im.mode in ("RGB", "L"):
            # Decode big JPEGs at reduced size: much faster, still >= 2x target.
            im.draft(im.mode, (img_size * 2, img_size * 2))
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im = im.resize((img_size, img_size), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def normalise(arr_hwc, mean, std):
    """uint8 HxWx3 array -> float tensor 3xHxW, standardised."""
    x = torch.from_numpy(np.ascontiguousarray(arr_hwc)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std


def class_names(splits_csv):
    return sorted(pd.read_csv(splits_csv)["cls"].unique())


def load_split(split, data_dir, splits_csv, img_size, cache_dir=".cache"):
    """Load one split ('train', 'val' or 'test') into memory as uint8 tensors.

    Decoded images are cached in cache_dir so later runs start instantly.
    Returns X (N,3,H,W uint8), y (N, int64) and the sorted class names.
    """
    df = pd.read_csv(splits_csv)
    classes = sorted(df["cls"].unique())
    df = df[df["split"] == split].sort_values("rel_path").reset_index(drop=True)
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)

    key = hashlib.md5(("|".join(df["rel_path"]) + f"|{img_size}").encode()).hexdigest()[:10]
    cache_file = cache_dir / f"{split}_{img_size}_{key}.npz"

    if cache_file.exists():
        X = np.load(cache_file)["X"]
    else:
        X = np.zeros((len(df), img_size, img_size, 3), dtype=np.uint8)
        for i, rel in enumerate(df["rel_path"]):
            X[i] = load_image_uint8(data_dir / rel, img_size)
            if (i + 1) % 500 == 0:
                print(f"  [{split}] decoded {i + 1}/{len(df)} images", flush=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, X=X)

    y = np.array([classes.index(c) for c in df["cls"]], dtype=np.int64)
    X = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    return X, torch.from_numpy(y), classes


def compute_mean_std(X, batch=256):
    """Per-channel mean and std of a uint8 (N,3,H,W) tensor, scaled to [0, 1]."""
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    count = 0
    for i in range(0, len(X), batch):
        b = X[i:i + batch].double() / 255.0
        total += b.sum(dim=(0, 2, 3))
        total_sq += (b ** 2).sum(dim=(0, 2, 3))
        count += b.shape[0] * b.shape[2] * b.shape[3]
    mean = total / count
    std = (total_sq / count - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


class FoodDataset(Dataset):
    def __init__(self, X, y, mean, std, augment=False):
        self.X, self.y = X, y
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.aug = AUGMENT if augment else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.aug is not None:
            x = self.aug(x)
        x = (x.float() / 255.0 - self.mean) / self.std
        return x, self.y[i]