import numpy as np
import pytest
from pydantic import ValidationError

from berrytrack.config import TrackerConfig
from berrytrack.types import Detections


def test_empty_and_invalid_detections():
    assert len(Detections.empty()) == 0
    with pytest.raises(ValueError):
        Detections(np.array([[1, 1, 0, 0]]), np.array([0.8]))
    with pytest.raises(ValueError):
        Detections(np.array([[0, 0, 1, 1]]), np.array([float("nan")]))


def test_config_rejects_unknown_and_invalid_values():
    with pytest.raises(ValidationError):
        TrackerConfig(embedding_dim=31)
    with pytest.raises(ValidationError):
        TrackerConfig(typo=True)
