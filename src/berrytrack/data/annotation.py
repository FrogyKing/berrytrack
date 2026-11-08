"""Human verification exchange for videos lacking persistent fruit identities."""

from __future__ import annotations

import json
from pathlib import Path

from berrytrack.data.mot import write_mot


def create_annotation_task(
    video: Path,
    output: Path,
    frame_count: int,
    source_license: str,
) -> None:
    """Create a blank task. IDs and boxes must be supplied by a human annotator.

    Coordinates in this exchange are zero-based xywh. Empty, reviewed frames must
    be marked reviewed explicitly. Use CVAT interpolation/tracks for visual editing,
    then map its verified export to this exchange schema.
    """
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video": str(video),
                "license": source_license,
                "label_kind": "manual_pending",
                "reviewer": "",
                "sequence": video.stem,
                "frames": [
                    {"frame": frame, "reviewed": False, "objects": []}
                    for frame in range(1, frame_count + 1)
                ],
            },
            indent=2,
        )
    )


def export_verified_task(task: Path, mot_file: Path) -> None:
    """Reject incomplete/pseudo annotations; export only human-reviewed identities."""
    data = json.loads(task.read_text())
    if data.get("label_kind") != "manual_verified" or not data.get("reviewer", "").strip():
        raise ValueError("Human verification and named reviewer required")
    rows = []
    frame_ids = []
    for frame in data["frames"]:
        if frame.get("reviewed") is not True:
            raise ValueError("Every evaluation frame must be explicitly reviewed")
        number = frame["frame"]
        frame_ids.append(number)
        seen = set()
        for obj in frame["objects"]:
            identity = obj["track_id"]
            if not isinstance(identity, int) or identity < 1 or identity in seen:
                raise ValueError("IDs must be positive and unique within a frame")
            seen.add(identity)
            x, y, width, height = obj["bbox_xywh"]
            rows.append([number, identity, x + 1, y + 1, width, height, 1, 1, 1, -1])
    if frame_ids != list(range(1, len(frame_ids) + 1)):
        raise ValueError("Annotation coverage must be contiguous starting at frame1")
    write_mot(mot_file, rows)
    mot_file.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "label_kind": "manual_verified",
                "reviewer": data["reviewer"],
                "source_task": str(task),
                "license": data["license"],
                "evaluation_frames": frame_ids,
            },
            indent=2,
        )
    )
