import numpy as np

from berrytrack.association.cost import adaptive_weights, gated_assignment
from berrytrack.config import TrackerConfig
from berrytrack.geometry.boxes import neighborhood_context, pairwise_iou
from berrytrack.geometry.camera import CameraMotion
from berrytrack.tracking.berry import BerryTracker
from berrytrack.types import Detections


def detections(boxes, scores=None, features=None):
    return Detections(
        np.array(boxes, np.float32), np.array(scores or [0.9] * len(boxes)), features=features
    )


def test_hungarian_invalid_edges_do_not_steal_matches():
    cost = np.array([[0.1, 0.2], [0.11, 0.99]], np.float32)
    gate = np.array([[False, True], [True, False]])
    assert set(gated_assignment(cost, gate, 0.5)) == {(0, 1), (1, 0)}


def test_identity_survives_occlusion_and_low_confidence():
    tracker = BerryTracker(TrackerConfig(min_hits=1, camera_compensation=False))
    first = tracker.update(detections([[10, 10, 20, 20]]))
    tracker.update(Detections.empty())
    again = tracker.update(detections([[12, 10, 22, 20]]))
    low = tracker.update(detections([[13, 10, 23, 20]], [0.2]))
    assert first[0].track_id == again[0].track_id == low[0].track_id
    assert tracker.unique_count == 1


def test_low_confidence_does_not_initialize_identity():
    tracker = BerryTracker(TrackerConfig(min_hits=1))
    assert tracker.update(detections([[1, 1, 5, 5]], [0.2])) == []
    assert tracker.unique_count == 0


def test_geometry_shapes_and_camera_rejects_textureless_frame():
    boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], np.float32)
    assert neighborhood_context(boxes).shape == (2, 16)
    assert np.allclose(np.diag(pairwise_iou(boxes, boxes)), 1)
    camera = CameraMotion()
    frame = np.zeros((100, 100, 3), np.uint8)
    camera.estimate(frame)
    assert camera.estimate(frame).reliability == 0


def test_weight_simplex_and_disabled_components():
    weights = adaptive_weights(np.ones(2), np.array([0, 1]), 0.5, 1, [True, True, False, False])
    assert np.allclose(weights.sum(1), 1)
    assert np.all(weights[:, 2:] == 0)
    assert weights[1, 0] > weights[0, 0]


def test_distant_object_does_not_reuse_id():
    tracker = BerryTracker(TrackerConfig(min_hits=1, camera_compensation=False))
    first = tracker.update(detections([[0, 0, 10, 10]]))
    second = tracker.update(detections([[500, 500, 510, 510]]))
    assert first[0].track_id != second[0].track_id
