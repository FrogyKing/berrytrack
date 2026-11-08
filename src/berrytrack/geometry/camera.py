from dataclasses import dataclass, field

import cv2
import numpy as np

from berrytrack.types import FloatArray, Image


@dataclass
class CameraEstimate:
    matrix: FloatArray = field(default_factory=lambda: np.eye(2, 3, dtype=np.float32))
    reliability: float = 0.0
    inliers: int = 0
    residual: float = 0.0
    reason: str = "first frame"


class CameraMotion:
    """Robust partial affine camera model, rejected under weak support/parallax."""

    def __init__(self, max_features: int = 1000) -> None:
        self.orb = cv2.ORB.create(max_features)
        self.previous: tuple | None = None

    def estimate(self, frame: Image, boxes: FloatArray | None = None) -> CameraEstimate:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        # Work at bounded resolution; transform translations back to original pixels.
        ratio = min(1.0, 640 / max(width, height))
        gray = cv2.resize(gray, None, fx=ratio, fy=ratio)
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = np.round(box * ratio).astype(int)
                cv2.rectangle(mask, (max(0, x1), max(0, y1)), (x2, y2), 0, -1)
        points, descriptors = self.orb.detectAndCompute(gray, mask)
        old = self.previous
        self.previous = (points, descriptors, gray.shape)
        result = CameraEstimate()
        if old is None:
            return result
        old_points, old_desc, old_shape = old
        if old_shape != gray.shape or descriptors is None or old_desc is None:
            result.reason = "missing descriptors or changed resolution"
            return result
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(old_desc, descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 8:
            result.reason = "insufficient matches"
            return result
        src = np.asarray([old_points[m.queryIdx].pt for m in good], dtype=np.float32)
        dst = np.asarray([points[m.trainIdx].pt for m in good], dtype=np.float32)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=1000
        )
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            result.reason = "invalid RANSAC transform"
            return result
        valid = inliers.ravel().astype(bool)
        n = int(valid.sum())
        fraction = n / len(good)
        scale = float(np.linalg.det(matrix[:, :2])) ** 0.5
        error = np.linalg.norm(src @ matrix[:, :2].T + matrix[:, 2] - dst, axis=1)
        residual = float(np.median(error[valid])) if n else float("inf")
        # Inlier spatial coverage prevents a moving leaf/fruit dominating the camera estimate.
        coverage = float(np.prod(np.ptp(src[valid], axis=0)) / np.prod(gray.shape)) if n else 0
        if n < 8 or fraction < 0.4 or not 0.85 <= scale <= 1.18 or coverage < 0.03:
            result.reason = "weak support, excessive scale, or local/parallax motion"
            return result
        if residual > 2.5 or np.linalg.norm(matrix[:, 2]) > 0.35 * max(gray.shape):
            result.reason = "excessive residual or displacement"
            return result
        matrix[:, 2] /= ratio
        return CameraEstimate(
            matrix.astype(np.float32),
            fraction * min(1.0, coverage * 4),
            n,
            residual / ratio,
            "accepted",
        )
