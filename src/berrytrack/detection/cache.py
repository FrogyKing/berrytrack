import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from berrytrack.data.mot import MOTSequence, write_mot
from berrytrack.detection.yolo import YoloDetector


def cache_detections(
    sequence: Path, weights: str, output: Path, device: str = "cpu", image_size: int = 640
) -> dict:
    seq = MOTSequence.load(sequence)
    detector = YoloDetector(weights, device=device, image_size=image_size)
    rows = []
    durations = []
    for i in range(1, seq.length + 1):
        frame = cv2.imread(str(seq.image_path(i)))
        if frame is None:
            raise FileNotFoundError(seq.image_path(i))
        started = time.perf_counter()
        det = detector.predict(np.asarray(frame, dtype=np.uint8))
        durations.append(time.perf_counter() - started)
        for box, score in zip(det.boxes, det.scores, strict=True):
            x1, y1, x2, y2 = box
            rows.append([i, -1, x1 + 1, y1 + 1, x2 - x1, y2 - y1, score, -1, -1, -1])
    write_mot(output, rows)
    result = {
        "sequence": seq.name,
        "frames": seq.length,
        "detector": weights,
        "image_size": image_size,
        "threshold": 0.1,
        "device": device,
        "detections": len(rows),
        "mean_detection_ms": float(np.mean(durations) * 1000),
        "source": "detector predictions; not ground truth",
        "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        "cache_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result
