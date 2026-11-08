"""Adaptive sparse-context association with lazy appearance and identity memory."""

from dataclasses import dataclass

import numpy as np

from berrytrack.association.cost import adaptive_weights, gated_assignment
from berrytrack.config import TrackerConfig
from berrytrack.embeddings.appearance import CropAppearance
from berrytrack.geometry.boxes import centers, neighborhood_context, pairwise_iou
from berrytrack.geometry.camera import CameraEstimate, CameraMotion
from berrytrack.tracking.kalman import BoxKalman
from berrytrack.types import Detections, FloatArray, Image, TrackOutput


@dataclass
class Identity:
    track_id: int
    filter: BoxKalman
    score: float
    class_id: int
    context: FloatArray
    feature: FloatArray | None = None
    hits: int = 1
    missed: int = 0
    confirmed: bool = False


class BerryTracker:
    def __init__(self, config: TrackerConfig | None = None, appearance=None) -> None:
        self.config = config or TrackerConfig()
        if self.config.low_threshold > self.config.high_threshold:
            raise ValueError("low_threshold must not exceed high_threshold")
        if not any((self.config.appearance, self.config.motion, self.config.context)):
            raise ValueError("At least one association cue must be enabled")
        self.identities: list[Identity] = []
        self.next_id = 1
        self.counted_ids: set[int] = set()
        self.camera = CameraMotion()
        self.appearance = appearance or CropAppearance(self.config.embedding_dim, self.config.seed)
        self.last_camera = CameraEstimate()
        self.last_weights = np.empty((0, 4), np.float32)
        self.appearance_crops = 0

    @property
    def unique_count(self) -> int:
        return len(self.counted_ids)

    def update(self, detections: Detections, frame: Image | None = None) -> list[TrackOutput]:
        cfg = self.config
        assert detections.class_ids is not None
        self.last_camera = (
            self.camera.estimate(frame, detections.boxes)
            if cfg.camera_compensation and frame is not None
            else CameraEstimate()
        )
        for track in self.identities:
            track.filter.predict(self.last_camera.matrix if self.last_camera.reliability else None)
            track.missed += 1
        context = neighborhood_context(detections.boxes, cfg.neighbor_k)
        features: list[FloatArray | None] = [None] * len(detections)
        if cfg.appearance and detections.features is not None:
            features = list(detections.features)
        n, m = len(self.identities), len(detections)
        matches: list[tuple[int, int]] = []
        if n and m:
            boxes = np.stack([t.filter.bbox for t in self.identities])
            scale = np.sqrt(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)).clip(1)
            spatial = np.linalg.norm(
                centers(boxes)[:, None] - centers(detections.boxes)[None], axis=2
            )
            spatial = spatial / scale[:, None]
            mahalanobis = np.stack([t.filter.distance(detections.boxes) for t in self.identities])
            motion = np.clip(mahalanobis / cfg.gate_distance, 0, 1)
            geometry = 0.5 * np.clip(spatial / 4, 0, 1) + 0.5 * (
                1 - pairwise_iou(boxes, detections.boxes)
            )
            local = (
                np.abs(np.stack([t.context for t in self.identities])[:, None] - context[None])
                .mean(2)
                .clip(0, 1)
            )
            uncertainty = np.array(
                [min(t.missed / cfg.max_age, 3) for t in self.identities], np.float32
            )
            # Motion-only gating is deliberately disabled for the appearance-only ablation.
            gate = (mahalanobis <= cfg.gate_distance) if cfg.motion else np.ones((n, m), bool)
            for i, track in enumerate(self.identities):
                gate[i] &= detections.class_ids == track.class_id
                if track.missed > cfg.max_age:
                    gate[i] &= spatial[i] <= cfg.gate_distance * 2
            preliminary = np.where(gate, geometry if cfg.motion else local, np.inf)
            needed: set[int] = set()
            if cfg.appearance:
                if not cfg.uncertainty_trigger or not cfg.motion:
                    needed.update(range(m))
                else:
                    for i, row in enumerate(preliminary):
                        candidates = np.flatnonzero(np.isfinite(row))
                        ordered = candidates[np.argsort(row[candidates])]
                        ambiguous = (
                            len(ordered) > 1
                            and row[ordered[1]] - row[ordered[0]] < cfg.ambiguity_margin
                        )
                        if (
                            ambiguous
                            or self.identities[i].missed > 1
                            or self.last_camera.reliability < 0.2
                        ):
                            needed.update(candidates.tolist())
                    # Competing tracks for the same detection also trigger appearance.
                    for j in range(m):
                        if int(gate[:, j].sum()) > 1:
                            needed.add(j)
            self._extract(needed, features, frame, detections.boxes)
            app_cost = np.full((n, m), 0.5, np.float32)
            available = np.zeros((n, m), bool)
            for i, track in enumerate(self.identities):
                for j, feature in enumerate(features):
                    if track.feature is not None and feature is not None:
                        denom = np.linalg.norm(track.feature) * np.linalg.norm(feature)
                        if denom > 1e-8:
                            app_cost[i, j] = np.clip(
                                (1 - track.feature @ feature / denom) / 2, 0, 1
                            )
                            available[i, j] = True
                    if track.missed > cfg.max_age:
                        gate[i, j] &= available[i, j] and app_cost[i, j] < 0.15
            reliability = available.mean(1).astype(np.float32)
            self.last_weights = adaptive_weights(
                reliability,
                uncertainty,
                self.last_camera.reliability,
                min(1, m / (cfg.neighbor_k + 1)),
                [cfg.appearance, cfg.motion, cfg.motion, cfg.context],
                cfg.adaptive,
            )
            terms = np.stack((app_cost, motion, geometry, local), axis=2)
            pair_weights = np.repeat(self.last_weights[:, None], m, axis=1)
            pair_weights[:, :, 0] *= available
            pair_weights /= np.maximum(pair_weights.sum(2, keepdims=True), 1e-8)
            cost = np.sum(terms * pair_weights, axis=2)
            # High-confidence stage followed by low-confidence recovery, no low-score births.
            high = detections.scores >= cfg.high_threshold
            matches = gated_assignment(cost, gate & high[None], cfg.match_threshold)
            used_i = {i for i, _ in matches}
            used_j = {j for _, j in matches}
            low_gate = gate & ((detections.scores >= cfg.low_threshold) & ~high)[None]
            for i in used_i:
                low_gate[i] = False
            for j in used_j:
                low_gate[:, j] = False
            # Only currently active, confirmed identities can recover weak detections.
            for i, track in enumerate(self.identities):
                if not track.confirmed or track.missed > 1:
                    low_gate[i] = False
            matches += gated_assignment(cost, low_gate, cfg.match_threshold * 0.85)
        matched_detections = {j for _, j in matches}
        for i, j in matches:
            track = self.identities[i]
            track.filter.update(detections.boxes[j])
            track.score = float(detections.scores[j])
            track.missed = 0
            track.hits += 1
            track.context = context[j]
            if features[j] is not None:
                feature = features[j]
                assert feature is not None
                track.feature = (
                    feature if track.feature is None else 0.9 * track.feature + 0.1 * feature
                )
            track.confirmed |= track.hits >= cfg.min_hits
            if track.confirmed:
                self.counted_ids.add(track.track_id)
        new = [
            j
            for j in range(m)
            if j not in matched_detections and detections.scores[j] >= cfg.high_threshold
        ]
        if cfg.appearance:
            self._extract(set(new), features, frame, detections.boxes)
        for j in new:
            track = Identity(
                self.next_id,
                BoxKalman(detections.boxes[j]),
                float(detections.scores[j]),
                int(detections.class_ids[j]),
                context[j],
                features[j],
                confirmed=cfg.min_hits <= 1,
            )
            self.identities.append(track)
            self.next_id += 1
            if track.confirmed:
                self.counted_ids.add(track.track_id)
        self.identities = [
            t
            for t in self.identities
            if t.missed <= cfg.memory_age and (t.confirmed or t.missed <= 1)
        ]
        return [
            TrackOutput(t.track_id, t.filter.bbox, t.score, t.confirmed, t.class_id)
            for t in self.identities
            if t.missed == 0 and t.confirmed
        ]

    def _extract(
        self, needed: set[int], features: list, frame: Image | None, boxes: FloatArray
    ) -> None:
        missing = sorted(j for j in needed if features[j] is None)
        if not missing or frame is None or not self.config.appearance:
            return
        encoded = self.appearance.extract(frame, boxes[missing])
        self.appearance_crops += len(missing)
        for j, feature in zip(missing, encoded, strict=True):
            features[j] = feature
