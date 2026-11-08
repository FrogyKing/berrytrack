import numpy as np
from scipy.optimize import linear_sum_assignment

from berrytrack.types import FloatArray


def adaptive_weights(
    appearance_reliability: FloatArray,
    uncertainty: FloatArray,
    camera_reliability: float,
    context_reliability: float,
    enabled: list[bool],
    adaptive: bool = True,
) -> FloatArray:
    """Per-track simplex weights in appearance, motion, geometry, context order."""
    n = len(uncertainty)
    weights = np.tile(np.array([0.30, 0.30, 0.25, 0.15], np.float32), (n, 1))
    if adaptive:
        weights[:, 0] *= appearance_reliability * (1 + uncertainty)
        weights[:, 1] *= 1 / (1 + uncertainty)
        weights[:, 2] *= 0.35 + 0.65 * camera_reliability
        weights[:, 3] *= context_reliability
    weights *= np.asarray(enabled, dtype=np.float32)
    return (weights / np.maximum(weights.sum(1, keepdims=True), 1e-8)).astype(np.float32)


def gated_assignment(cost: FloatArray, gate: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Only admissible edges enter assignment; rejected edges cannot steal a match.

    Per-row dummy columns represent an unmatched track, avoiding forced bad matches.
    """
    if not cost.size:
        return []
    valid = gate & np.isfinite(cost) & (cost <= threshold)
    rows, cols = cost.shape
    augmented = np.full((rows, cols + rows), threshold + 1e-4, dtype=np.float64)
    augmented[:, :cols] = np.where(valid, cost, 1e6)
    ri, ci = linear_sum_assignment(augmented)
    return [(int(i), int(j)) for i, j in zip(ri, ci, strict=True) if j < cols and valid[i, j]]
