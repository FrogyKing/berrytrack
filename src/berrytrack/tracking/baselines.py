"""Thin adapters around upstream baseline implementations, never reimplementations."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np

from berrytrack.config import TrackerConfig
from berrytrack.types import Detections, Image, TrackOutput

# Missing dependencies must fail with actionable errors instead of mutating uv's environment.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


class UltralyticsTracker:
    """ByteTrack or BoT-SORT (official default: GMC on, ReID off).

    ``fps`` is recorded for callers; all lifetimes are defined in frames for fairness.
    """

    def __init__(self, name: str, config: TrackerConfig, fps: float = 30) -> None:
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.trackers.byte_tracker import BYTETracker

        if name not in {"bytetrack", "botsort"}:
            raise ValueError(f"Unknown upstream tracker: {name}")
        self.name = name
        self.fps = fps
        self.appearance_crops = 0
        self.args = SimpleNamespace(
            tracker_type=name,
            track_high_thresh=config.high_threshold,
            track_low_thresh=config.low_threshold,
            new_track_thresh=config.high_threshold,
            track_buffer=config.max_age,
            match_thresh=config.match_threshold,
            fuse_score=True,
            gmc_method="sparseOptFlow" if config.camera_compensation else "none",
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model="auto",
        )
        self.tracker: Any = BYTETracker(self.args) if name == "bytetrack" else BOTSORT(self.args)

    def update(self, detections: Detections, frame: Image | None = None) -> list[TrackOutput]:
        from ultralytics.engine.results import Boxes

        assert detections.class_ids is not None
        raw = np.column_stack((detections.boxes, detections.scores, detections.class_ids))
        shape = frame.shape[:2] if frame is not None else (1, 1)
        boxes = Boxes(raw.astype(np.float32), orig_shape=shape)
        result = self.tracker.update(boxes, img=frame)
        return [
            TrackOutput(
                int(row[4]), row[:4].astype(np.float32), float(row[5]), class_id=int(row[6])
            )
            for row in result
        ]


class DeepSortTracker:
    """DeepSORT with supplied/cache features or shared compact crop descriptors.

    This is the upstream Kalman/cascade/appearance association implementation.
    The default descriptor is *not* a pretrained person-ReID network and is
    explicitly reported as ``deepsort-compact`` in experiment metadata.
    """

    def __init__(self, config: TrackerConfig, fps: float = 30) -> None:
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self.config = config
        self.fps = fps
        self.appearance_crops = 0
        self._extractor: Any = None
        self.tracker = DeepSort(
            max_age=config.max_age,
            n_init=config.min_hits,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder=None,
            nms_max_overlap=1.0,
        )

    def update(self, detections: Detections, frame: Image | None = None) -> list[TrackOutput]:
        assert detections.class_ids is not None
        if len(detections) and detections.features is None:
            if frame is None:
                raise ValueError("DeepSORT requires cached appearance features or an image")
            if self._extractor is None:
                from berrytrack.embeddings.appearance import CropAppearance

                self._extractor = CropAppearance(self.config.embedding_dim, self.config.seed)
            features = self._extractor.extract(frame, detections.boxes)
            self.appearance_crops += len(detections)
        else:
            features = detections.features
        rows = []
        for box, score, cls in zip(
            detections.boxes, detections.scores, detections.class_ids, strict=True
        ):
            xywh = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
            rows.append((xywh, float(score), int(cls)))
        tracks = self.tracker.update_tracks(rows, embeds=features if len(rows) else [])
        return [
            TrackOutput(
                int(track.track_id),
                np.asarray(track.to_ltrb(orig=True), dtype=np.float32),
                float(track.get_det_conf() or 0.0),
                class_id=int(track.get_det_class() or 0),
            )
            for track in tracks
            if track.is_confirmed() and track.time_since_update == 0
        ]


def create_baseline(name: str, config: TrackerConfig | None = None, fps: float = 30) -> Any:
    config = config or TrackerConfig()
    if name in {"bytetrack", "botsort"}:
        return UltralyticsTracker(name, config, fps)
    if name == "deepsort":
        return DeepSortTracker(config, fps)
    raise ValueError(f"Unknown baseline: {name}")
