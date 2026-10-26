"""Exploratory blueberry ROI diagnostic with explicit visual-reference provenance.

This is an execution check, not an independent human-verified tracking benchmark.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from berrytrack.config import TrackerConfig
from berrytrack.data.mot import MOTSequence, write_mot
from berrytrack.evaluation.benchmark import (
    _environment,
    run_sequence,
    sha256,
    source_snapshot,
    write_reports,
)


def run_reference(sequence_root: Path, output: Path) -> dict:
    sequence = MOTSequence.load(sequence_root)
    provenance = json.loads((sequence_root / "reference_annotations.json").read_text())
    if provenance.get("label_kind") != "exploratory_visual_review" or provenance.get(
        "human_verified"
    ):
        raise ValueError("Expected the explicitly nonhuman-reviewed exploratory reference")
    cached = sequence.gt.copy()
    cached[:, 1] = -1
    cached[:, 6] = 1
    cache = output / "cache" / "reference-boxes.txt"
    write_mot(cache, cached)
    modes = {
        "legacy": dict(local_rigidity=False, static_scene=False),
        "polygon": dict(local_rigidity=True, static_scene=False),
        "static_polygon_experimental": dict(local_rigidity=True, static_scene=True),
        "bytetrack": dict(algorithm="bytetrack", local_rigidity=False, static_scene=False),
        "botsort": dict(algorithm="botsort", local_rigidity=False, static_scene=False),
        "deepsort": dict(algorithm="deepsort", local_rigidity=False, static_scene=False),
    }
    results = []
    for name, settings in modes.items():
        row = run_sequence(
            sequence,
            TrackerConfig.model_validate(settings),
            cached,
            output / "predictions" / f"{name}.txt",
        )
        row["variant"] = name
        results.append(row)
    metadata = {
        "executed_at": datetime.now(UTC).isoformat(),
        "experiment": "tiny_visually_inspected_blueberry_roi",
        "oracle_detections": True,
        "interpretation": (
            "EXPLORATORY EXECUTION CHECK ONLY. Eight near-stationary real blueberry ROI frames "
            "with seven identities authored by exploratory visual inspection. Reference boxes "
            "are also tracker inputs with IDs removed. No independent human-verified GT, "
            "full-scene count, detector accuracy or H1-H4 evidence."
        ),
        "annotation_kind": provenance["label_kind"],
        "human_verified": False,
        "independent_benchmark": False,
        "reference_provenance": provenance,
        "source_manifest_sha256": sha256(sequence_root / "manifest.json"),
        "reference_sha256": sha256(sequence_root / "gt" / "gt.txt"),
        "detections_sha256": sha256(cache),
        "environment": _environment(),
        "implementation": source_snapshot(),
        "source_fps": None,
        "playback_fps": 10,
        "timing_note": (
            "8-frame timing is too small and initialization-sensitive for efficiency claims"
        ),
    }
    write_reports(results, output, metadata)
    return {"metadata": metadata, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        type=Path,
        default=Path("data/processed/agrivision-visual-reference-reproduced"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/blueberry-reference"))
    args = parser.parse_args()
    run_reference(args.sequence, args.output)
