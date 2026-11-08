"""Render a single-tracker blueberry GIF from real frames and reference boxes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from berrytrack.config import TrackerConfig
from berrytrack.data.mot import MOTSequence, read_mot, write_mot
from berrytrack.evaluation.benchmark import run_sequence


def font(size: int, bold: bool = False):
    candidates = (
        ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold
        else ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError("Arial or DejaVu Sans is required")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(sequence: Path, output: Path) -> None:
    seq = MOTSequence.load(sequence)
    provenance = json.loads((sequence / "reference_annotations.json").read_text())
    if provenance.get("label_kind") != "assistant_visual_review":
        raise ValueError("Expected the explicitly assistant-authored blueberry reference")
    if (seq.width, seq.height, seq.length) != (260, 260, 8):
        raise ValueError("This presentation is pinned to the eight-frame blueberry ROI")
    work = Path("artifacts/blueberry-demo")
    work.mkdir(parents=True, exist_ok=True)
    detections = seq.gt.copy()
    detections[:, 1] = -1  # Reference identities never enter tracker inputs.
    detections[:, 6] = 1
    cache = work / "reference-boxes.txt"
    write_mot(cache, detections)
    predictions = work / "berrytrack.txt"
    config = TrackerConfig()
    run_sequence(seq, config, detections, predictions)
    rows = read_mot(predictions)
    output.mkdir(parents=True, exist_ok=True)
    frames, seen = [], set()
    palette = ["#56e7b2", "#f9c76b", "#8dcaff", "#f8a9cc", "#b7a5ff", "#80ece5", "#e0ee8c"]
    for index in range(1, seq.length + 1):
        frame = Image.open(seq.image_path(index)).convert("RGB")
        canvas = Image.new("RGB", (960, 640), "#0b1120")
        draw = ImageDraw.Draw(canvas)
        draw.text((26, 20), "BerryTrack", font=font(34, True), fill="#f1f5f9")
        draw.text((580, 35), "GEOMETRY-AWARE FRUIT TRACKING", font=font(15, True), fill="#56e7b2")
        canvas.paste(frame.resize((520, 520), Image.Resampling.LANCZOS), (24, 92))
        current = rows[rows[:, 0] == index]
        for row in current:
            track_id = int(row[1])
            seen.add(track_id)
            x, y, w, h = row[2:6] * 2
            color = palette[(track_id - 1) % len(palette)]
            box = (24 + x, 92 + y, 24 + x + w, 92 + y + h)
            draw.rectangle(box, outline=color, width=3)
            bx, by = box[0], max(93, box[1] - 25)
            draw.rounded_rectangle((bx, by, bx + 44, by + 24), radius=4, fill=color)
            draw.text((bx + 4, by + 1), f"#{track_id}", font=font(18, True), fill="#0b1120")
        draw.text((580, 115), "Blueberry\ntracking", font=font(42, True), fill="#f1f5f9", spacing=5)
        draw.text((580, 253), str(len(seen)), font=font(78, True), fill="#56e7b2")
        draw.text((580, 346), "confirmed track IDs", font=font(23), fill="#d0daea")
        draw.text(
            (580, 402),
            "Real blueberry frames\nPersistent identity labels",
            font=font(21),
            fill="#d0daea",
            spacing=12,
        )
        draw.line((580, 493, 930, 493), fill="#30415a", width=1)
        draw.text((580, 516), "REFERENCE-BOX DEMO", font=font(17, True), fill="#56e7b2")
        draw.text(
            (580, 548),
            "Assistant-authored annotations\nPlayback 4 FPS; capture FPS unknown",
            font=font(16),
            fill="#a7b5cc",
            spacing=8,
        )
        draw.text(
            (26, 617),
            f"AgriVision DB-2  /  frame {index:02d}/08  /  CC-BY-4.0",
            font=font(14),
            fill="#a7b5cc",
        )
        if index == 4:
            canvas.save(output / "blueberry-demo.png")
        frames.append(canvas)
    if seen != set(range(1, 8)):
        raise ValueError("Unexpected identities; re-audit the blueberry demo")
    raw = work / "render.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter.fourcc(*"mp4v"), 4, (960, 640))
    if not writer.isOpened():
        raise RuntimeError("Video encoder unavailable")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(raw),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / "blueberry-demo.mp4"),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    frames[0].save(
        output / "blueberry-demo.gif", save_all=True, append_images=frames[1:], duration=250, loop=0
    )
    manifest = {
        "source": "https://doi.org/10.6084/m9.figshare.29222462.v1",
        "license": "CC-BY-4.0",
        "sequence": seq.name,
        "frames": 8,
        "source_fps": None,
        "playback_fps": 4,
        "algorithm": "berrytrack",
        "tracker_config": config.model_dump(),
        "confirmed_ids": sorted(seen),
        "label_kind": "assistant_visual_review",
        "human_verified": False,
        "independent_benchmark": False,
        "input_kind": "reference boxes with IDs removed",
        "interpretation": (
            "Single-tracker software demo; not detector performance "
            "or comparative accuracy evidence."
        ),
        "modifications": (
            "Source ROI resized 2x, actual predicted boxes and IDs overlaid, "
            "presentation titles, explicit playback speed. All eight frames retained."
        ),
        "reference_sha256": sha256(sequence / "gt/gt.txt"),
        "detections_sha256": sha256(cache),
        "predictions_sha256": sha256(predictions),
        "source_frames_sha256": {str(f): sha256(seq.image_path(f)) for f in range(1, 9)},
        "reproduce": "uv run scripts/make_blueberry_demo.py",
    }
    (output / "blueberry-demo.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered BerryTrack-only blueberry GIF and H.264 video in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        type=Path,
        default=Path("data/processed/agrivision-assistant-reference-reproduced"),
    )
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    render(args.sequence, args.output)
