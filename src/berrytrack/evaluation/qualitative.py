"""Export compact, attributed crops around measured identity-switch events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from berrytrack.data.mot import MOTSequence, read_mot

BUTOM_CREATORS = [
    "Michael Halstead",
    "Esra Guclu",
    "Mohamed Farag",
    "Enrico Pallotta",
    "Christian Hund",
    "Ribana Roscher",
    "Maren Bennewitz",
    "Juergen Gall",
    "Chris McCool",
]


def _panel(
    sequence: MOTSequence,
    frame: int,
    gt_id: int,
    prediction_id: int,
    predictions: np.ndarray,
    timestamp: str,
) -> np.ndarray:
    image = cv2.imread(str(sequence.image_path(frame)))
    if image is None:
        raise FileNotFoundError(sequence.image_path(frame))
    truth = sequence.gt
    target = truth[(truth[:, 0] == frame) & (truth[:, 1] == gt_id)]
    if not len(target):
        raise ValueError("Switch target is not annotated in the selected frame")
    box = target[0, 2:6].copy()
    box[:2] -= 1
    box[2:] += box[:2]
    center = (box[:2] + box[2:]) / 2
    radius = max(95, int(max(box[2:] - box[:2]) * 3))
    x0, y0 = np.maximum(0, center.astype(int) - radius)
    x1, y1 = np.minimum((sequence.width, sequence.height), center.astype(int) + radius)
    crop = image[y0:y1, x0:x1].copy()
    offset = np.array([x0, y0])

    def draw_box(coordinates: np.ndarray, color: tuple[int, int, int]) -> None:
        start = np.round(coordinates[:2] - offset).astype(int)
        end = np.round(coordinates[2:] - offset).astype(int)
        cv2.rectangle(crop, tuple(start), tuple(end), color, 1, cv2.LINE_AA)

    draw_box(box, (100, 255, 90))
    chosen = predictions[(predictions[:, 0] == frame) & (predictions[:, 1] == prediction_id)]
    if len(chosen):
        pbox = chosen[0, 2:6].copy()
        pbox[:2] -= 1
        pbox[2:] += pbox[:2]
        draw_box(pbox, (240, 80, 255))
    crop = cv2.resize(crop, (470, 340), interpolation=cv2.INTER_CUBIC)
    panel = np.full((430, 500, 3), (26, 20, 24), dtype=np.uint8)
    panel[70:410, 15:485] = crop
    cv2.putText(
        panel,
        f"MOT frame {frame} | GT {gt_id} -> P {prediction_id}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"source ns {timestamp}",
        (15, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return panel


def export_switch_montage(
    sequence_root: Path,
    results_path: Path,
    output: Path,
    variant: str = "berrytrack",
    seed: int = 42,
    event_limit: int = 2,
) -> dict[str, Any]:
    sequence = MOTSequence.load(sequence_root)
    result_file = json.loads(results_path.read_text(encoding="utf-8"))
    selected = next(
        result
        for result in result_file["results"]
        if result["sequence"] == sequence.name
        and result["variant"] == variant
        and result["seed"] == seed
    )
    events = selected["metrics"]["identity_switch_events"][:event_limit]
    if not events:
        raise ValueError("No measured switches for this run; no failure image fabricated")
    predictions = read_mot(
        results_path.parent / "predictions" / f"{sequence.name}-{variant}-{seed}.txt"
    )
    manifest = json.loads((sequence_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("license") != "CC-BY-4.0" or manifest.get("dataset") != "BUTom-ST21":
        raise ValueError("This exporter currently permits attributed BUTom-ST21 CC-BY-4.0 crops")
    timestamps = {entry["frame"]: entry["source_timestamp_ns"] for entry in manifest["frames"]}
    truth = sequence.gt
    panels = []
    exported = []
    for event in events:
        current = event["frame"]
        prior_candidates = truth[(truth[:, 0] < current) & (truth[:, 1] == event["gt_id"]), 0]
        prior_pred_frames = predictions[predictions[:, 1] == event["previous_id"], 0]
        available = np.intersect1d(prior_candidates, prior_pred_frames)
        if not len(available):
            continue
        previous = int(available.max())
        panels.append(
            np.hstack(
                [
                    _panel(
                        sequence,
                        previous,
                        event["gt_id"],
                        event["previous_id"],
                        predictions,
                        timestamps[previous],
                    ),
                    _panel(
                        sequence,
                        current,
                        event["gt_id"],
                        event["new_id"],
                        predictions,
                        timestamps[current],
                    ),
                ]
            )
        )
        exported.append(
            {
                **event,
                "previous_frame": previous,
                "source_previous_ns": timestamps[previous],
                "source_current_ns": timestamps[current],
            }
        )
    if not panels:
        raise ValueError("No preceding matched frame is available for the switch events")
    header = np.full((90, 1000, 3), (26, 20, 24), dtype=np.uint8)
    cv2.putText(
        header,
        f"{variant}: measured identity switches | oracle detections | tomato",
        (20, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "Green = human-corrected GT; magenta = tracker prediction. Before / after.",
        (20, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    footer = np.full((65, 1000, 3), (26, 20, 24), dtype=np.uint8)
    cv2.putText(
        footer,
        "Michael Halstead et al., BUTom-ST21 / University of Bonn. CC BY 4.0.",
        (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "Source: doi.org/10.60507/FK2/TTPCNV | crops + overlays by BerryTrack",
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    montage = np.vstack([header, *panels, footer])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), montage):
        raise OSError(f"Could not write {output}")
    provenance = {
        "sequence": sequence.name,
        "variant": variant,
        "seed": seed,
        "source": manifest["source"],
        "license": manifest["license"],
        "creators": BUTOM_CREATORS,
        "modifications": "Cropped original images, resized and overlaid GT and predicted boxes",
        "interpretation": "Observed failures from an oracle-box development diagnostic",
        "events": exported,
    }
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/identity-switches.png"))
    parser.add_argument("--variant", default="berrytrack")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export_switch_montage(args.sequence, args.results, args.output, args.variant, args.seed)


if __name__ == "__main__":
    main()
