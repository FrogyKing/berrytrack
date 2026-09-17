"""CPU-friendly PyTorch dataset interfaces independent of detector implementations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from berrytrack.data.mot import MOTSequence


class MOTFrameDataset(Dataset):
    def __init__(self, sequence: str | Path):
        self.sequence = MOTSequence.load(sequence)
        self.rows = self.sequence.gt

    def __len__(self) -> int:
        return self.sequence.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        image = np.asarray(Image.open(self.sequence.image_path(index + 1)).convert("RGB")).copy()
        rows = self.rows[self.rows[:, 0] == index + 1]
        boxes = rows[:, 2:6].copy()
        boxes[:, :2] -= 1
        boxes[:, 2:] += boxes[:, :2]
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255, {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "track_ids": torch.tensor(rows[:, 1], dtype=torch.long),
            "labels": torch.ones(len(rows), dtype=torch.long),
            "image_id": torch.tensor(index),
        }


class YoloDetectionDataset(Dataset):
    """Paired images and normalized YOLO labels; no temporal IDs are invented."""

    def __init__(self, images: list[Path]):
        self.images = images
        for image in images:
            if not image.with_suffix(".txt").exists():
                raise FileNotFoundError(image.with_suffix(".txt"))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        path = self.images[index]
        image = np.asarray(Image.open(path).convert("RGB")).copy()
        height, width = image.shape[:2]
        text = path.with_suffix(".txt").read_text().strip()
        labels = np.loadtxt(path.with_suffix(".txt"), ndmin=2) if text else np.empty((0, 5))
        if labels.shape[1] != 5 or not np.isfinite(labels).all():
            raise ValueError(f"Invalid YOLO annotations: {path}")
        if labels.size and ((labels[:, 1:] < 0).any() or (labels[:, 1:] > 1).any()):
            raise ValueError("YOLO boxes must be normalized to [0,1]")
        xy = labels[:, 1:3] * [width, height]
        wh = labels[:, 3:5] * [width, height]
        boxes = np.concatenate((xy - wh / 2, xy + wh / 2), axis=1)
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255, {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels[:, 0] + 1, dtype=torch.long),
            "image_id": torch.tensor(index),
        }


def grouped_split(
    sample_groups: dict[str, str],
    seed: int = 42,
) -> dict[str, list[str]]:
    """Stable 70/15/15 split by capture group, never by frame.

    The caller must supply honest capture-group information: hashing filenames
    cannot discover whether two images show the same physical plant.
    """
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sample, group in sorted(sample_groups.items()):
        bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 70 else ("val" if bucket < 85 else "test")
        splits[split].append(sample)
    return splits
