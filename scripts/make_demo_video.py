"""Render a local reference demo from licensed data; no dataset enters Git."""

import argparse
from pathlib import Path

import cv2

from berrytrack.data.mot import MOTSequence


def make_video(sequence: Path, output: Path) -> None:
    seq = MOTSequence.load(sequence)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter.fourcc(*"mp4v"), seq.fps, (seq.width, seq.height)
    )
    if not writer.isOpened():
        raise RuntimeError("Video encoder unavailable")
    try:
        for i in range(1, seq.length + 1):
            frame = cv2.imread(str(seq.image_path(i)))
            if frame is None:
                raise FileNotFoundError(seq.image_path(i))
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=Path("data/processed/butom-st21-235"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo/input.mp4"))
    args = parser.parse_args()
    make_video(args.sequence, args.output)
