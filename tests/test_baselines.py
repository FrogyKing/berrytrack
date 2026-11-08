import numpy as np
import pytest

from berrytrack.config import TrackerConfig
from berrytrack.tracking.baselines import create_baseline
from berrytrack.types import Detections


def detections(offset=0.0, score=0.9):
    return Detections(
        np.array([[10 + offset, 10, 30 + offset, 30]], dtype=np.float32),
        np.array([score], dtype=np.float32),
        features=np.array([[1, 0, 0, 0]], dtype=np.float32),
    )


@pytest.mark.parametrize("name", ["bytetrack", "botsort", "deepsort"])
def test_upstream_baseline_keeps_identity_across_short_gap(name):
    tracker = create_baseline(name, TrackerConfig(min_hits=2, max_age=10))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracker.update(detections(), frame)
    first = tracker.update(detections(1), frame)
    assert len(first) == 1
    identifier = first[0].track_id
    tracker.update(Detections.empty(), frame)
    recovered = tracker.update(detections(2), frame)
    assert len(recovered) == 1
    assert recovered[0].track_id == identifier


def test_bytetrack_recovers_low_score_box_in_second_stage():
    tracker = create_baseline("bytetrack")
    first = tracker.update(detections())
    second = tracker.update(detections(1, 0.2))
    assert first[0].track_id == second[0].track_id
    assert second[0].score == pytest.approx(0.2)


def test_baseline_factory_rejects_unknown():
    with pytest.raises(ValueError):
        create_baseline("unknown")


def test_deepsort_does_not_silently_use_missing_appearance():
    tracker = create_baseline("deepsort")
    boxes = Detections(np.array([[1, 1, 5, 5]]), np.array([0.9]))
    with pytest.raises(ValueError, match="features or an image"):
        tracker.update(boxes)
