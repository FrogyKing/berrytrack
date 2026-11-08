from collections import defaultdict, deque

import cv2
import numpy as np

from berrytrack.types import Image, TrackOutput


class TrackRenderer:
    def __init__(self, history: int = 24) -> None:
        self.history = history
        self.trajectories: dict[int, deque] = defaultdict(lambda: deque(maxlen=history))

    def draw(self, frame: Image, tracks: list[TrackOutput], count: int, label: str) -> Image:
        image = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = np.round(track.bbox).astype(int)
            rng = np.random.default_rng(track.track_id)
            color = tuple(int(c) for c in rng.integers(70, 255, 3))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"#{track.track_id}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            points = self.trajectories[track.track_id]
            points.append(((x1 + x2) // 2, (y1 + y2) // 2))
            if len(points) > 1:
                cv2.polylines(image, [np.array(points, np.int32)], False, color, 1, cv2.LINE_AA)
        # Bound rendering memory even for very long videos.
        visible = {t.track_id for t in tracks}
        if len(self.trajectories) > 1000:
            self.trajectories = defaultdict(
                lambda: deque(maxlen=self.history), {i: self.trajectories[i] for i in visible}
            )
        cv2.rectangle(image, (0, 0), (image.shape[1], 38), (24, 20, 40), -1)
        cv2.putText(
            image,
            f"{label} | unique identities: {count}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 245, 250),
            2,
            cv2.LINE_AA,
        )
        return image
