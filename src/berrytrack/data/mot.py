"""Strict MOTChallenge I/O. Frames and coordinates on disk are one based."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_mot(path: str | Path) -> NDArray[np.float64]:
    """Load MOT rows, padding 6--9-column inputs to ten columns.

    Missing files are errors: unknown annotations must never become empty ground truth.
    An explicitly empty file represents a fully annotated sequence with no objects.
    """
    path = Path(path)
    if not path.read_text().strip():
        return np.empty((0, 10), dtype=np.float64)
    rows = np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64)
    if rows.shape[1] < 6 or rows.shape[1] > 10:
        raise ValueError("MOT rows must contain between six and ten columns")
    if not np.isfinite(rows).all():
        raise ValueError("MOT rows contain non-finite numbers")
    if (rows[:, 0] < 1).any() or (rows[:, 0] != np.floor(rows[:, 0])).any():
        raise ValueError("MOT frame indices must be positive integers")
    if (rows[:, 4:6] <= 0).any():
        raise ValueError("MOT boxes must have positive width and height")
    if rows.shape[1] < 10:
        missing_confidence = rows.shape[1] == 6
        rows = np.pad(rows, ((0, 0), (0, 10 - rows.shape[1])), constant_values=-1)
        if missing_confidence:
            rows[:, 6] = 1.0
    return rows[np.argsort(rows[:, 0], kind="stable")]


def write_mot(path: str | Path, rows: NDArray | list[list[float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rows, dtype=float)
    if arr.size == 0:
        path.write_text("")
        return
    np.savetxt(path, arr, fmt="%.6f", delimiter=",")
    read_mot(path)


@dataclass(frozen=True)
class MOTSequence:
    root: Path
    name: str
    width: int
    height: int
    fps: float
    length: int
    extension: str = ".jpg"
    image_directory: str = "img1"

    @classmethod
    def load(cls, root: str | Path) -> MOTSequence:
        root = Path(root)
        config = configparser.ConfigParser()
        if not config.read(root / "seqinfo.ini"):
            raise FileNotFoundError(root / "seqinfo.ini")
        seq = config["Sequence"]
        result = cls(
            root,
            seq["name"],
            int(seq["imWidth"]),
            int(seq["imHeight"]),
            float(seq["frameRate"]),
            int(seq["seqLength"]),
            seq.get("imExt", ".jpg"),
            seq.get("imDir", "img1"),
        )
        if min(result.width, result.height, result.length, result.fps) <= 0:
            raise ValueError("Invalid sequence dimensions, length, or frame rate")
        return result

    @property
    def gt(self) -> NDArray[np.float64]:
        rows = read_mot(self.root / "gt" / "gt.txt")
        if rows.size and (rows[:, 0] > self.length).any():
            raise ValueError("Ground-truth frame exceeds sequence length")
        keys = rows[:, :2]
        if len(keys) != len(np.unique(keys, axis=0)):
            raise ValueError("Duplicate ground-truth identity within a frame")
        return rows

    @property
    def detections(self) -> NDArray[np.float64]:
        return read_mot(self.root / "det" / "det.txt")

    def image_path(self, frame: int) -> Path:
        if not 1 <= frame <= self.length:
            raise ValueError("Frame outside sequence")
        return self.root / self.image_directory / f"{frame:06d}{self.extension}"


def validate_splits(splits: dict[str, list[str]]) -> None:
    """Reject duplicate sequences and cross-split leakage."""
    seen: set[str] = set()
    for name, sequences in splits.items():
        current = set(sequences)
        if len(current) != len(sequences) or current & seen:
            raise ValueError(f"Repeated sequence in split {name}")
        seen |= current
