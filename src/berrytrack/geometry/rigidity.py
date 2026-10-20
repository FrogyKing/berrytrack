"""Anchor-conditioned, locally affine polygon consistency for quasi-static fruit.

Corresponding neighbor identities define a deformable local coordinate system.
A candidate is scored by its agreement with that system, allowing translation,
rotation, scale and moderate shear rather than assuming constant pixel distances.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from berrytrack.geometry.boxes import centers
from berrytrack.types import FloatArray


@dataclass
class RigidityResult:
    cost: FloatArray
    gate: np.ndarray
    predictions: FloatArray
    reliability: FloatArray
    anchors: int = 0


def refine_polygon_cost(
    previous_boxes: FloatArray,
    detection_boxes: FloatArray,
    cost: FloatArray,
    gate: np.ndarray,
    anchors: list[tuple[int, int]],
    prior: FloatArray | None = None,
    neighbors: int = 6,
    weight: float = 0.35,
    gate_sigma: float = 4.0,
    recovery_eligible: np.ndarray | None = None,
) -> RigidityResult:
    """Fit robust local affine maps to reliable *other* identity correspondences.

    A fit needs three non-collinear anchors. Collinear, reflected, highly strained
    or extrapolated neighborhoods receive no constraint. Missing neighbors simply
    reduce support; no fixed nearest-neighbor rank correspondence is assumed.
    All candidate edges remain subject to the caller's class/lifecycle gates.
    """
    n, m = cost.shape
    result = RigidityResult(
        cost.copy(),
        gate.copy(),
        np.full((n, 2), np.nan, np.float32),
        np.zeros(n, np.float32),
        len(anchors),
    )
    if len(anchors) < 3 or not n or not m or weight <= 0:
        return result
    old = centers(previous_boxes).astype(np.float64)
    new = centers(detection_boxes).astype(np.float64)
    source_ids = np.array([i for i, _ in anchors])
    target_ids = np.array([j for _, j in anchors])
    source, target = old[source_ids], new[target_ids]
    tree = cKDTree(source)
    matrix_prior = np.eye(2) if prior is None else np.asarray(prior[:, :2], dtype=float)
    if np.linalg.det(matrix_prior) <= 0:
        return result
    fruit_scale = np.sqrt(np.prod(previous_boxes[:, 2:] - previous_boxes[:, :2], axis=1)).clip(1)
    count = min(len(anchors), neighbors + 1)
    distances, nearby = tree.query(old, k=count)
    for i in range(n):
        indices = nearby[i][source_ids[nearby[i]] != i][:neighbors]
        # Locality limits depth/parallax coupling across distant parts of a canopy.
        indices = indices[np.linalg.norm(source[indices] - old[i], axis=1) < 12 * fruit_scale[i]]
        if len(indices) < 3:
            continue
        p, q = source[indices], target[indices]
        origin, destination = p.mean(0), q.mean(0)
        radius = max(float(np.sqrt(np.mean(np.sum((p - origin) ** 2, axis=1)))), 1)
        x, y = (p - origin) / radius, (q - destination) / radius
        eigenvalues = np.linalg.eigvalsh(x.T @ x)
        if eigenvalues[0] < 0.025 * max(eigenvalues[-1], 1e-8):
            continue
        leverage = float(np.linalg.norm(old[i] - origin) / radius)
        if leverage > 2.5:
            continue
        weights = np.ones(len(p))
        affine = matrix_prior.copy()
        for _ in range(3):
            regularizer = 0.03 * len(p)
            affine = np.linalg.solve(
                x.T @ (weights[:, None] * x) + regularizer * np.eye(2),
                x.T @ (weights[:, None] * y) + regularizer * matrix_prior.T,
            ).T
            residual = np.linalg.norm(x @ affine.T - y, axis=1) * radius
            tolerance = max(2.0, 0.25 * float(fruit_scale[i]))
            weights = np.minimum(1, tolerance / np.maximum(residual, 1e-6))
        # Relative strain permits a softly deformed polygon, never a reflection.
        strain = np.linalg.svd(affine @ np.linalg.inv(matrix_prior), compute_uv=False)
        if np.linalg.det(affine) <= 0 or strain.min() < 0.6 or strain.max() > 1.65:
            continue
        residual_px = float(np.median(residual))
        if residual_px > max(3.0, float(fruit_scale[i]) * 0.6):
            continue
        predicted = destination + affine @ (old[i] - origin)
        sigma = max(2.0, 0.35 * float(fruit_scale[i]) + 2 * residual_px)
        sigma *= 1 + max(0, leverage - 1)
        confidence = min(1, len(indices) / 5) * np.exp(-residual_px / sigma)
        result.predictions[i] = predicted.astype(np.float32)
        result.reliability[i] = confidence
        normalized = np.linalg.norm(new - predicted, axis=1) / sigma
        polygon_cost = 1 - np.exp(-0.5 * normalized**2)
        mixture = weight * confidence
        result.cost[i] = (1 - mixture) * cost[i] + mixture * polygon_cost
        # A reliable local camera model can rescue a plausible active candidate
        # rejected by the less precise global-motion filter. It cannot bypass
        # class constraints or long-gap re-identification evidence.
        if confidence >= 0.85 and recovery_eligible is not None:
            result.gate[i] |= recovery_eligible[i] & (normalized <= 2.0)
        # This veto is evidence-based and active only with a well-supported fit.
        if confidence >= 0.65:
            result.gate[i] &= normalized <= gate_sigma
    return result
