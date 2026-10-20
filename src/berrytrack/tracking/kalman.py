"""Constant-velocity box filter with covariance-preserving camera compensation."""

import numpy as np

from berrytrack.types import FloatArray


class BoxKalman:
    def __init__(self, bbox: FloatArray) -> None:
        self.mean = np.zeros(8, np.float64)
        self.mean[:2] = (bbox[:2] + bbox[2:]) / 2
        self.mean[2:4] = bbox[2:] - bbox[:2]
        self.covariance = np.diag([16, 16, 16, 16, 100, 100, 25, 25]).astype(float)
        self.transition = np.eye(8)
        self.transition[:4, 4:] = np.eye(4)

    @property
    def bbox(self) -> FloatArray:
        wh = self.mean[2:4].clip(1)
        return np.concatenate((self.mean[:2] - wh / 2, self.mean[:2] + wh / 2)).astype(np.float32)

    def predict(self, camera: FloatArray | None = None, static_scene: bool = False) -> None:
        if camera is not None:
            a = camera[:, :2]
            self.mean[:2] = a @ self.mean[:2] + camera[:, 2]
            self.mean[4:6] = a @ self.mean[4:6]
            # Axis-aligned envelope of the transformed rectangle.
            self.mean[2:4] = np.abs(a) @ self.mean[2:4]
            self.mean[6:8] = np.abs(a) @ self.mean[6:8]
            jacobian = np.zeros((8, 8))
            jacobian[:2, :2] = a
            jacobian[2:4, 2:4] = np.abs(a)
            jacobian[4:6, 4:6] = a
            jacobian[6:8, 6:8] = np.abs(a)
            self.covariance = jacobian @ self.covariance @ jacobian.T
        transition = self.transition
        if static_scene and camera is not None:
            # Fruits are quasi-static in world coordinates. Camera compensation
            # explains image motion; retain only a damped residual for parallax/wind.
            transition = self.transition.copy()
            transition[:, 4:] *= 0.2
        self.mean = transition @ self.mean
        q = max(1.0, float(np.mean(self.mean[2:4])) * 0.03)
        self.covariance = transition @ self.covariance @ transition.T + np.eye(8) * q**2
        self.mean[2:4] = self.mean[2:4].clip(1)

    def update(self, bbox: FloatArray) -> None:
        measurement = np.r_[(bbox[:2] + bbox[2:]) / 2, bbox[2:] - bbox[:2]]
        r = np.eye(4) * max(1.0, float(np.mean(self.mean[2:4])) * 0.05) ** 2
        covariance = self.covariance[:4, :4] + r
        gain = np.linalg.solve(covariance, self.covariance[:4]).T
        self.mean += gain @ (measurement - self.mean[:4])
        # Joseph update maintains positive semidefiniteness under numerical error.
        identity = np.eye(8)
        identity[:, :4] -= gain
        self.covariance = identity @ self.covariance @ identity.T + gain @ r @ gain.T

    def distance(self, boxes: FloatArray) -> FloatArray:
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        delta = center - self.mean[:2]
        covariance = self.covariance[:2, :2] + np.eye(2) * 4
        return np.sqrt(np.maximum(np.sum(delta * np.linalg.solve(covariance, delta.T).T, 1), 0))
