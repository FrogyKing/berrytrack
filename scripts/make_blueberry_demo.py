"""Render a long, single-tracker blueberry sequence with actual trajectory trails."""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from berrytrack.config import TrackerConfig
from berrytrack.data.agrivision import mask_component_boxes
from berrytrack.data.mot import write_mot
from berrytrack.tracking.berry import BerryTracker
from berrytrack.types import Detections

PLAYBACK_FPS = 10
TRAIL_FRAMES = 48
PALETTE = ["#56e7b2", "#f9c76b", "#8dcaff", "#f8a9cc", "#b7a5ff", "#80ece5", "#e0ee8c"]


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
    source = json.loads((sequence / "manifest.json").read_text(encoding="utf-8"))
    if (source["camera"], source["video"], source["frame_count"]) != (1, 104, 160):
        raise ValueError("Prepare the long camera1/video104 sequence first")
    if [r["source_frame"] for r in source["frames"]] != list(range(1, 161)):
        raise ValueError("The demo requires all160 consecutive original source frames")
    work = Path("artifacts/blueberry-demo-long")
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    config = TrackerConfig()
    tracker = BerryTracker(config)
    cv2.setNumThreads(1)
    trails = defaultdict(lambda: deque(maxlen=TRAIL_FRAMES))
    observations = defaultdict(list)
    mot_rows, input_rows, source_hashes, active_counts = [], [], [], []
    raw = work / "render.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter.fourcc(*"mp4v"), PLAYBACK_FPS, (960, 840))
    if not writer.isOpened():
        raise RuntimeError("Video encoder unavailable")
    title, body, small = font(34, True), font(20), font(14)
    try:
        for index, record in enumerate(source["frames"], 1):
            image_path = sequence / "img1" / f"{index:06d}.jpg"
            mask_path = sequence / "pseudo_masks" / f"{index:06d}.png"
            image_hash, mask_hash = sha256(image_path), sha256(mask_path)
            if image_hash != record["sha256"] or mask_hash != record["pseudo_mask"]["sha256"]:
                raise ValueError(f"Source integrity mismatch at frame{index}")
            frame = cv2.imread(str(image_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if frame is None or mask is None or frame.shape[:2] != mask.shape:
                raise ValueError(f"Image/mask dimensions do not match at frame{index}")
            boxes = mask_component_boxes(mask)
            detections = Detections(boxes, np.ones(len(boxes), np.float32))
            for x1, y1, x2, y2 in boxes:
                input_rows.append([index, -1, x1 + 1, y1 + 1, x2 - x1, y2 - y1, 1, -1, -1, -1])
            tracks = [t for t in tracker.update(detections, frame) if t.confirmed]
            active_counts.append(len(tracks))
            source_hashes.append(
                {
                    "frame": index,
                    "source_member": record["source_member"],
                    "image_sha256": image_hash,
                    "mask_sha256": mask_hash,
                }
            )
            canvas = Image.new("RGB", (960, 840), "#0b1120")
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 20), "BerryTrack", font=title, fill="#f1f5f9")
            draw.text(
                (470, 35), "BLUEBERRY TRACKING + TRAJECTORIES", font=font(18, True), fill="#56e7b2"
            )
            # Fixed full-frame view: no artificial camera motion, panning or zoom.
            scene = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
                (720, 720), Image.Resampling.LANCZOS
            )
            canvas.paste(scene, (24, 84))
            sx, sy = 720 / frame.shape[1], 720 / frame.shape[0]
            for track in tracks:
                tid = track.track_id
                x1, y1, x2, y2 = map(float, track.bbox)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                points = trails[tid]
                if points and points[-1][0] != index - 1:
                    points.clear()  # Do not draw an invented path through unobserved gaps.
                points.append((index, center))
                observations[tid].append((index, center))
                color = PALETTE[(tid - 1) % len(PALETTE)]
                line = [(24 + p[0] * sx, 84 + p[1] * sy) for _, p in points]
                if len(line) > 1:
                    draw.line(line, fill="#10202a", width=5)
                    draw.line(line, fill=color, width=3)
                    px, py = line[0]
                    draw.ellipse((px - 3, py - 3, px + 3, py + 3), outline=color, width=1)
                bx, by = 24 + x1 * sx, 84 + y1 * sy
                draw.rectangle((bx, by, 24 + x2 * sx, 84 + y2 * sy), outline=color, width=2)
                ly = max(85, by - 18)
                label = f"#{tid}"
                label_width = draw.textlength(label, font=small) + 8
                draw.rectangle((bx, ly, bx + label_width, ly + 18), fill="#0b1120")
                draw.text((bx + 3, ly), label, font=small, fill=color)
                mot_rows.append(
                    [index, tid, x1 + 1, y1 + 1, x2 - x1, y2 - y1, track.score, -1, -1, -1]
                )
            draw.text((772, 104), "Moving\ncamera.", font=font(25, True), fill="#f1f5f9", spacing=8)
            draw.text((772, 206), str(len(tracks)), font=font(60, True), fill="#56e7b2")
            draw.text((772, 280), "active tracks", font=body, fill="#cbd5e1")
            draw.line((772, 329, 936, 329), fill="#30415a", width=1)
            draw.text((772, 352), "TRAIL HISTORY", font=font(16, True), fill="#56e7b2")
            draw.text(
                (772, 385),
                "48 frames\nActual track\npositions",
                font=body,
                fill="#cbd5e1",
                spacing=10,
            )
            draw.text((772, 518), f"{index:03d} / 160", font=font(23, True), fill="#f1f5f9")
            draw.text((772, 558), f"{index / PLAYBACK_FPS:.1f}s / 16.0s", font=body, fill="#cbd5e1")
            draw.text((772, 603), "playback time", font=small, fill="#a7b5cc")
            draw.text(
                (772, 680),
                "Source-mask\nassociation\ndemo",
                font=font(17),
                fill="#a7b5cc",
                spacing=8,
            )
            draw.text(
                (24, 816),
                "AgriVision DB-2  |  160 real consecutive frames  |  "
                "playback 10 FPS; capture FPS unknown",
                font=small,
                fill="#a7b5cc",
            )
            writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
            if index in (1, 40, 80, 120, 160):
                canvas.save(work / f"review-{index:03d}.jpg")
            if index == 80:
                canvas.save(output / "blueberry-demo.png")
            if index % 40 == 0:
                print(f"Tracked {index}/160 frames; active={len(tracks)}", flush=True)
    finally:
        writer.release()
    predictions, inputs = work / "berrytrack.txt", work / "mask-boxes.txt"
    write_mot(predictions, mot_rows)
    write_mot(inputs, input_rows)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(raw),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / "blueberry-demo.mp4"),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(output / "blueberry-demo.mp4"),
            "-filter_complex",
            "fps=5,scale=640:560:flags=lanczos,split[a][b];"
            "[a]palettegen=max_colors=96:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=none:diff_mode=rectangle",
            "-loop",
            "0",
            str(output / "blueberry-demo.gif"),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    motion = []
    for tid, values in observations.items():
        coords = np.asarray([center for _, center in values])
        extent = float(np.linalg.norm(coords - coords[0], axis=1).max())
        motion.append(
            {
                "track_id": tid,
                "observations": len(values),
                "first_frame": values[0][0],
                "last_frame": values[-1][0],
                "max_displacement_from_first_source_px": round(extent, 2),
            }
        )
    manifest = {
        "source": "https://doi.org/10.6084/m9.figshare.29222462.v1",
        "license": "CC-BY-4.0",
        "camera": 1,
        "video": 104,
        "frames": 160,
        "source_frame_range": [1, 160],
        "source_fps": None,
        "playback_fps": PLAYBACK_FPS,
        "duration_seconds": 16,
        "gif_fps": 5,
        "gif_sampling": "Every second source frame; full160 frames in MP4",
        "algorithm": "berrytrack",
        "tracker_config": config.model_dump(),
        "trail_history_frames": TRAIL_FRAMES,
        "trails_break_on_missing_frames": True,
        "input_kind": "Bounding boxes from publisher-inferred binary mask components",
        "component_area_limits": [80, 30000],
        "proposal_score": 1,
        "proposal_score_note": (
            "Assigned uniformly for association; not calibrated detector confidence"
        ),
        "tracking_ground_truth": "unavailable",
        "independent_benchmark": False,
        "interpretation": (
            "Single-tracker qualitative execution demo, not counting or comparative "
            "accuracy evidence. Touching masked fruit can form one proposal; "
            "missed/merged masks and identity errors remain."
        ),
        "modifications": (
            "Fixed full-frame resize, actual BerryTrack predictions, 48-frame trajectory "
            "trails, labels and presentation layout. No artificial camera movement "
            "or interpolated frames."
        ),
        "detections_sha256": sha256(inputs),
        "predictions_sha256": sha256(predictions),
        "source_frames": source_hashes,
        "active_tracks_per_frame": active_counts,
        "track_motion": sorted(motion, key=lambda r: r["observations"], reverse=True),
        "reproduce": [
            "uv run scripts/prepare_blueberry_demo.py",
            "uv run scripts/make_blueberry_demo.py",
        ],
    }
    (output / "blueberry-demo.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Rendered16-second BerryTrack-only blueberry video and GIF: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence", type=Path, default=Path("data/processed/agrivision-camera1-video104-long")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    render(args.sequence, args.output)
