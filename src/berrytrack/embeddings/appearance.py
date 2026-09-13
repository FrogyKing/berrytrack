import cv2
import numpy as np

from berrytrack.types import FloatArray, Image


class CropAppearance:
    """Untrained HSV + low-frequency DCT descriptor with a seeded random projection.

    This is a CPU fallback, not a claimed learned fruit ReID model. A precomputed
    detector-backbone feature matrix can instead be passed in Detections.features.
    """

    def __init__(self, dimension: int = 64, seed: int = 42) -> None:
        self.dimension = dimension
        rng = np.random.default_rng(seed)
        self.projection = rng.normal(size=(144, dimension)).astype(np.float32)
        self.projection /= np.sqrt(144)
        self.crops_computed = 0

    def extract(self, frame: Image, boxes: FloatArray) -> FloatArray:
        features = np.zeros((len(boxes), self.dimension), dtype=np.float32)
        height, width = frame.shape[:2]
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = np.round(box).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(frame[y1:y2, x1:x2], (16, 16))
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32) / 255
            # 3 channels x 16 low-frequency coefficients + 3 x 32-bin histograms.
            dct = np.concatenate([cv2.dct(hsv[:, :, c])[:4, :4].ravel() for c in range(3)])
            hist = np.concatenate([np.histogram(hsv[:, :, c], 32, (0, 1))[0] for c in range(3)])
            vector = np.concatenate((dct / 16, hist / 256)).astype(np.float32)
            features[i] = vector @ self.projection
            self.crops_computed += 1
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return (features / np.maximum(norms, 1e-8)).astype(np.float32)
