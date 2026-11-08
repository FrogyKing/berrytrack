"""YOLO adapter with optional reuse of detector pyramid features via ROI pooling."""

import os
from pathlib import Path
from typing import Any, cast

import numpy as np

from berrytrack.types import Detections, Image

os.environ.setdefault("YOLO_AUTOINSTALL", "false")


class YoloDetector:
    def __init__(
        self,
        weights: str | Path,
        confidence: float = 0.1,
        image_size: int = 640,
        device: str = "cpu",
        reuse_features: bool = False,
        dimension: int = 64,
    ) -> None:
        from ultralytics import YOLO

        self.model: Any = YOLO(str(weights))
        self.confidence, self.image_size, self.device = confidence, image_size, device
        self.reuse_features, self.dimension = reuse_features, dimension
        self._pyramid = None
        self._input_size: tuple[int, int] | None = None
        self._projection = None
        if reuse_features:
            # Initialize the inference copy before installing bound-method hooks.
            self.model.predict(
                np.zeros((64, 64, 3), np.uint8), imgsz=64, device=device, verbose=False
            )
            # The Detect head receives the same pyramid used for box prediction.
            network: Any = cast(Any, self.model).predictor.model.model
            network.model[-1].register_forward_pre_hook(self._capture)
            network.model[0].register_forward_pre_hook(self._capture_input)

    def _capture_input(self, module, inputs) -> None:
        self._input_size = (int(inputs[0].shape[-2]), int(inputs[0].shape[-1]))

    def _capture(self, module, inputs) -> None:
        self._pyramid = inputs[0][0].detach().clone()

    def predict(self, frame: Image) -> Detections:
        self._pyramid = None
        result = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
            max_det=1000,
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        features = None
        if self.reuse_features and len(boxes):
            features = self._roi_features(boxes, frame.shape[:2])
        return Detections(
            boxes,
            result.boxes.conf.cpu().numpy().astype(np.float32),
            result.boxes.cls.cpu().numpy().astype(np.int64),
            features,
        )

    def _roi_features(self, boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        import torch
        from torchvision.ops import roi_align

        if self._pyramid is None:
            raise RuntimeError("Detector backbone hook did not capture a feature pyramid")
        fmap = self._pyramid
        # Ultralytics letterboxes with a stride-aligned minimum rectangle for one image.
        h, w = shape
        ratio = self.image_size / max(h, w)
        if self._input_size is None:
            raise RuntimeError("Detector input dimensions were not captured")
        padded_h, padded_w = self._input_size
        transformed = boxes.copy()
        transformed[:, [0, 2]] = (
            (transformed[:, [0, 2]] * ratio + (padded_w - round(w * ratio)) / 2)
            * fmap.shape[3]
            / padded_w
        )
        transformed[:, [1, 3]] = (
            (transformed[:, [1, 3]] * ratio + (padded_h - round(h * ratio)) / 2)
            * fmap.shape[2]
            / padded_h
        )
        rois = torch.as_tensor(
            np.column_stack((np.zeros(len(boxes)), transformed)),
            device=fmap.device,
            dtype=fmap.dtype,
        )
        with torch.inference_mode():
            pooled = roi_align(fmap, rois, output_size=1, spatial_scale=1.0, aligned=True).flatten(
                1
            )
            if self._projection is None:
                rng = torch.Generator(device="cpu").manual_seed(42)
                projection = torch.randn(pooled.shape[1], self.dimension, generator=rng)
                self._projection = (
                    projection.to(device=fmap.device, dtype=fmap.dtype) / pooled.shape[1] ** 0.5
                )
            features = torch.nn.functional.normalize(pooled @ self._projection, dim=1)
        return features.cpu().numpy().astype(np.float32)
