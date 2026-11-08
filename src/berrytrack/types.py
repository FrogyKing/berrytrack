"""Shared, detector-independent contracts. Boxes use absolute xyxy pixels."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
Image = NDArray[np.uint8]


@dataclass
class Detections:
    boxes: FloatArray
    scores: FloatArray
    class_ids: NDArray[np.int64] | None = None
    features: FloatArray | None = None

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        n = len(self.boxes)
        if len(self.scores) != n or not np.isfinite(self.boxes).all():
            raise ValueError("Boxes and scores must be finite and have equal lengths")
        if not np.isfinite(self.scores).all() or np.any((self.scores < 0) | (self.scores > 1)):
            raise ValueError("Detection scores must lie in [0, 1]")
        if np.any(self.boxes[:, 2:] <= self.boxes[:, :2]):
            raise ValueError("Boxes must have positive width and height")
        if self.class_ids is None:
            self.class_ids = np.zeros(n, dtype=np.int64)
        else:
            self.class_ids = np.asarray(self.class_ids, dtype=np.int64).reshape(-1)
            if len(self.class_ids) != n:
                raise ValueError("Class ID count differs from box count")
        if self.features is not None:
            self.features = np.asarray(self.features, dtype=np.float32)
            if self.features.ndim != 2 or len(self.features) != n:
                raise ValueError("Features must have shape (number of boxes, dimension)")
            if not np.isfinite(self.features).all():
                raise ValueError("Features must be finite")

    def __len__(self) -> int:
        return len(self.boxes)

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.empty((0, 4), np.float32), np.empty(0, np.float32))


@dataclass
class TrackOutput:
    track_id: int
    bbox: FloatArray
    score: float
    confirmed: bool = True
    class_id: int = 0


class Tracker(Protocol):
    def update(self, detections: Detections, frame: Image | None = None) -> list[TrackOutput]: ...


@dataclass
class RunStatistics:
    frames: int = 0
    detection_seconds: float = 0.0
    association_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    unique_count: int = 0
    appearance_crops: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
