"""Real video inference shared by CLI and UI; comparisons reuse identical detections."""

import json
import logging
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from berrytrack.config import TrackerConfig
from berrytrack.data.mot import read_mot, write_mot
from berrytrack.evaluation.benchmark import mot_detections
from berrytrack.tracking import create_tracker
from berrytrack.types import RunStatistics
from berrytrack.visualization.render import TrackRenderer


def process_video(
    video: Path,
    output: Path,
    config: TrackerConfig | None = None,
    detector: Any = None,
    cached_detections: Path | None = None,
    compare: str | None = None,
    ground_truth: Path | None = None,
    max_frames: int = 3000,
    max_seconds: int = 600,
    appearance_checkpoint: Path | None = None,
) -> dict:
    config = config or TrackerConfig()
    log = logging.getLogger(__name__)
    if detector is None and cached_detections is None:
        raise ValueError("Provide detector weights or a cached MOT detection file")
    if min(max_frames, max_seconds) < 1:
        raise ValueError("Video processing limits must be positive")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30
    if not np.isfinite(fps) or fps <= 0:
        fps = 30
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 1 or height < 1 or width * height > 3840 * 2160:
        cap.release()
        raise ValueError("Video must have valid dimensions, at most 3840 x 2160 pixels")
    output.mkdir(parents=True, exist_ok=True)
    log.info("Starting %s video tracking, comparison=%s", config.algorithm, compare)
    cache = read_mot(cached_detections) if cached_detections is not None else None
    configs = [config]
    if compare and compare != config.algorithm:
        configs.append(config.model_copy(update={"algorithm": compare}))
    trackers = [create_tracker(c, fps) for c in configs]
    if appearance_checkpoint is not None:
        from berrytrack.embeddings.training import TorchAppearance
        from berrytrack.tracking.berry import BerryTracker

        for tracker in trackers:
            if isinstance(tracker, BerryTracker):
                tracker.appearance = TorchAppearance(appearance_checkpoint)
    renderers = [TrackRenderer() for _ in configs]
    stats = [
        RunStatistics(
            metadata={
                "algorithm": c.algorithm,
                "device": "cpu",
                "tracker_config": c.model_dump(),
            }
        )
        for c in configs
    ]
    seen: list[set[int]] = [set() for _ in configs]
    mot_rows: list[list[list[float]]] = [[] for _ in configs]
    raw_video = output / "render.mp4"
    writer = cv2.VideoWriter(
        str(raw_video), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width * len(configs), height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV could not create the annotated video")
    started = time.perf_counter()
    number = 0
    detection_seconds = 0.0
    try:
        while number < max_frames and time.perf_counter() - started < max_seconds:
            valid, raw_frame = cap.read()
            if not valid:
                break
            frame = np.asarray(raw_frame, dtype=np.uint8)
            number += 1
            detection_start = time.perf_counter()
            detections = (
                mot_detections(cache[cache[:, 0] == number])
                if cache is not None
                else detector.predict(frame)
            )
            detection_seconds += time.perf_counter() - detection_start
            # Ripeness/visibility categories represent the same physical fruit class.
            # A changing maturity classifier must not force a new tracking identity.
            detections.class_ids = np.zeros(len(detections), dtype=np.int64)
            panels = []
            for i, tracker in enumerate(trackers):
                association_start = time.perf_counter()
                tracks = tracker.update(detections, frame)
                stats[i].association_seconds += time.perf_counter() - association_start
                seen[i].update(t.track_id for t in tracks if t.confirmed)
                for track in tracks:
                    x1, y1, x2, y2 = track.bbox
                    mot_rows[i].append(
                        [
                            number,
                            track.track_id,
                            x1 + 1,
                            y1 + 1,
                            x2 - x1,
                            y2 - y1,
                            track.score,
                            -1,
                            -1,
                            -1,
                        ]
                    )
                panels.append(renderers[i].draw(frame, tracks, len(seen[i]), configs[i].algorithm))
            writer.write(np.concatenate(panels, axis=1))
    finally:
        cap.release()
        writer.release()
    if number == 0:
        raise ValueError("Video contains no decodable frames")
    elapsed = time.perf_counter() - started
    export = output / "annotated.mp4"
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(raw_video),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(export),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    result: dict[str, Any] = {
        "video": str(export.resolve()),
        "frames": number,
        "source": "cached detections" if cache is not None else "detector predictions",
        "execution_mode": "local CPU",
        "truncated": number >= max_frames or elapsed >= max_seconds,
        "results": [],
        "statistics_scope": (
            "wall includes decode, shared detector, trackers, render; excludes export transcode"
        ),
    }
    for i, stat in enumerate(stats):
        stat.frames, stat.detection_seconds, stat.elapsed_seconds = (
            number,
            detection_seconds,
            elapsed,
        )
        stat.unique_count = len(seen[i])
        stat.appearance_crops = int(getattr(trackers[i], "appearance_crops", 0))
        mot_path = output / f"{configs[i].algorithm}.txt"
        write_mot(mot_path, mot_rows[i])
        row = {
            "algorithm": configs[i].algorithm,
            "mot": str(mot_path.resolve()),
            **asdict(stat),
            "association_ms": 1000 * stat.association_seconds / number,
            "end_to_end_ms": elapsed * 1000 / number,
            "video_fps": number / elapsed,
        }
        if ground_truth is not None:
            from berrytrack.evaluation.metrics import evaluate_mot

            gt = read_mot(ground_truth)
            gt = gt[gt[:, 0] <= number]
            pred = np.asarray(mot_rows[i], dtype=float).reshape(-1, 10)
            row["metrics"] = evaluate_mot(gt, pred, number)
        result["results"].append(row)
    result["statistics"] = str((output / "statistics.json").resolve())
    (output / "statistics.json").write_text(json.dumps(result, indent=2))
    log.info(
        "Finished %d frames in %.3f seconds; counts=%s",
        number,
        elapsed,
        [s.unique_count for s in stats],
    )
    return result
