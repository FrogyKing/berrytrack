import numpy as np

from berrytrack.types import FloatArray


def centers(boxes: FloatArray) -> FloatArray:
    return (boxes[:, :2] + boxes[:, 2:]) * 0.5


def pairwise_iou(a: FloatArray, b: FloatArray) -> FloatArray:
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    size = np.maximum(high - low, 0)
    inter = size[..., 0] * size[..., 1]
    sa = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), axis=1)
    sb = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)
    return (inter / np.maximum(sa[:, None] + sb[None, :] - inter, 1e-8)).astype(np.float32)


def neighborhood_context(boxes: FloatArray, k: int = 4) -> FloatArray:
    """Sparse kNN radial/angular moments, normalized by each instance's scale.

    Only a cKDTree query is used; no dense N x N attention tensor is formed.
    Missing neighbors are zeros with an explicit occupancy channel.
    """
    from scipy.spatial import cKDTree

    result = np.zeros((len(boxes), k * 4), dtype=np.float32)
    if len(boxes) < 2:
        return result
    xy = centers(boxes)
    scale = np.sqrt(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)).clip(1)
    count = min(k + 1, len(boxes))
    _, indices = cKDTree(xy).query(xy, k=count)
    for i in range(len(boxes)):
        neighbors = indices[i, 1:]
        offsets = (xy[neighbors] - xy[i]) / scale[i]
        radius = np.linalg.norm(offsets, axis=1)
        features = np.column_stack(
            (
                np.tanh(radius / 8),
                offsets[:, 0] / (radius + 1e-6),
                offsets[:, 1] / (radius + 1e-6),
                np.ones(len(neighbors)),
            )
        )
        result[i, : len(neighbors) * 4] = features.reshape(-1)
    return result
