import json

import cv2
import numpy as np
import pytest

from berrytrack.config import TrackerConfig
from berrytrack.data.mot import read_mot, write_mot
from berrytrack.pipeline import process_video


@pytest.mark.integration
def test_video_comparison_exports_decodable_video_and_tracks(tmp_path):
    """Engineering fixture only; never used as a scientific benchmark."""
    video = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (64, 64))
    rows = []
    for i in range(1, 9):
        image = np.zeros((64, 64, 3), np.uint8)
        cv2.circle(image, (20 + i, 25), 5, (100, 100, 240), -1)
        writer.write(image)
        rows.append([i, 1, 16 + i, 21, 10, 10, 0.9, 1, 1, -1])
    writer.release()
    cache = tmp_path / "det.txt"
    write_mot(cache, rows)
    result = process_video(
        video,
        tmp_path / "out",
        TrackerConfig(min_hits=1),
        cached_detections=cache,
        compare="bytetrack",
        ground_truth=cache,
    )
    assert result["frames"] == 8
    assert len(result["results"]) == 2
    assert result["results"][0]["unique_count"] == 1
    assert len(read_mot(result["results"][0]["mot"])) == 8
    assert (
        json.loads((tmp_path / "out/statistics.json").read_text())["execution_mode"] == "local CPU"
    )
    rendered = cv2.VideoCapture(result["video"])
    assert rendered.read()[0]
    assert rendered.get(cv2.CAP_PROP_FRAME_WIDTH) == 128
    rendered.release()


def test_video_requires_detector_or_cache(tmp_path):
    with pytest.raises(ValueError, match="Provide detector"):
        process_video(tmp_path / "missing.mp4", tmp_path / "out")
