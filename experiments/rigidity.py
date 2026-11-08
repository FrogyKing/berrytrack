"""Controlled local-rigidity/static-scene comparison on real cached detections.

Historical experiments are preserved. This runner writes a separate experiment
and never trains a detector, tunes on annotations, or allocates cloud compute.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from berrytrack.config import TrackerConfig
from berrytrack.data.mot import MOTSequence, read_mot, write_mot
from berrytrack.evaluation.benchmark import (
    _environment,
    aggregate_counts,
    run_sequence,
    sha256,
    source_snapshot,
    write_reports,
)

VARIANTS: dict[str, tuple[str, bool, bool]] = {
    "legacy": ("berrytrack", False, False),
    "static_only": ("berrytrack", False, True),
    "rigidity": ("berrytrack", True, False),
    "static_rigidity": ("berrytrack", True, True),
    "bytetrack": ("bytetrack", False, False),
    "botsort": ("botsort", False, False),
    "deepsort": ("deepsort", False, False),
}


def rigidity_execution_evidence(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Distinguish supported geometry execution from merely enabling a switch."""
    enabled = [
        result
        for result in results
        if result["algorithm"] == "berrytrack" and result["config"].get("local_rigidity")
    ]
    missing = [result for result in enabled if "rigidity_fits" not in result]
    supported = [result for result in enabled if result.get("rigidity_fits", 0) > 0]
    status = (
        "no_rigidity_runs"
        if not enabled
        else "not_measured"
        if missing
        else "exercised"
        if supported
        else "not_exercised"
    )
    return {
        "status": status,
        "rigidity_enabled_runs": len(enabled),
        "runs_with_supported_fits": len(supported),
        "runs_missing_measurement": len(missing),
        "total_supported_fits": sum(result.get("rigidity_fits", 0) for result in enabled),
        "total_anchor_correspondences": sum(
            result.get("rigidity_anchor_count", 0) for result in enabled
        ),
        "interpretation": (
            "Fits count supported track-level local maps across frames; anchors count "
            "cumulative correspondences, not unique identities. Execution does not imply benefit."
        ),
        "runs": [
            {
                "sequence": result["sequence"],
                "variant": result["variant"],
                "seed": result["seed"],
                "supported_fits": result.get("rigidity_fits"),
                "anchor_correspondences": result.get("rigidity_anchor_count"),
            }
            for result in enabled
        ],
    }


def variant_config(name: str, seed: int, rigidity_weight: float = 0.35) -> TrackerConfig:
    """Explicit settings protect the control group from changes to defaults."""
    algorithm, local_rigidity, static_scene = VARIANTS[name]
    return TrackerConfig.model_validate(
        {
            "algorithm": algorithm,
            "seed": seed,
            "local_rigidity": local_rigidity,
            "static_scene": static_scene,
            "rigidity_weight": rigidity_weight,
        }
    )


def prepare_cache(
    sequence: MOTSequence, output: Path, mode: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cache boxes once per mode; GT identities never enter association."""
    cache_path = output / "cache" / f"{sequence.name}.txt"
    detector_provenance: dict[str, Any] | None = None
    if mode == "oracle":
        cached = sequence.gt.copy()
        cached[:, 1] = -1
        cached[:, 6] = 1
        source = sequence.root / "gt" / "gt.txt"
    elif mode == "cached":
        source = sequence.root / "det" / "det.txt"
        cached = read_mot(source)
        provenance_path = source.with_suffix(".json")
        if not provenance_path.exists():
            raise ValueError(f"Cached detector provenance is required: {provenance_path}")
        detector_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_hash = detector_provenance.get("cache_sha256")
        if expected_hash and expected_hash != sha256(source):
            raise ValueError(f"Detector cache differs from its recorded checksum: {source}")
        if "ground truth" in str(detector_provenance.get("source", "")).lower():
            # The existing detector provenance explicitly says "not ground truth".
            if "not ground truth" not in str(detector_provenance["source"]).lower():
                raise ValueError("GT-derived detections must use --mode oracle")
    else:
        raise ValueError(f"Unknown cache mode: {mode}")
    if len(cached) and cached[:, 0].max() > sequence.length:
        raise ValueError("Detection frame exceeds sequence coverage")
    write_mot(cache_path, cached)
    return cached, {
        "sequence": sequence.name,
        "mode": mode,
        "frames": sequence.length,
        "detections": len(cached),
        "root": str(sequence.root.resolve()),
        "input_source_sha256": sha256(source),
        "detections_sha256": sha256(cache_path),
        "ground_truth_sha256": sha256(sequence.root / "gt" / "gt.txt"),
        "seqinfo_sha256": sha256(sequence.root / "seqinfo.ini"),
        "manifest_sha256": sha256(sequence.root / "manifest.json"),
        "detector_provenance": detector_provenance,
    }


def run_comparison(
    sequence_paths: list[Path],
    output: Path,
    modes: list[str],
    seeds: list[int],
    variants: list[str],
    rigidity_weight: float = 0.35,
) -> dict[str, Any]:
    if len(set(seeds)) != len(seeds) or len(set(variants)) != len(variants):
        raise ValueError("Seeds and variants must be unique")
    if not seeds or not variants or any(name not in VARIANTS for name in variants):
        raise ValueError("At least one valid seed and variant are required")
    implementation = source_snapshot()
    implementation["experiment_runner_sha256"] = sha256(Path(__file__))
    output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    skipped = []
    for mode in modes:
        destination = output / mode
        destination.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        sources = []
        for root in sequence_paths:
            sequence = MOTSequence.load(root)
            if mode == "cached" and not (root / "det" / "det.txt").exists():
                reason = {
                    "sequence": sequence.name,
                    "mode": mode,
                    "status": "pending",
                    "reason": "No actual detector cache is available; no detections fabricated",
                }
                skipped.append(reason)
                print(f"PENDING {sequence.name} {mode}: {reason['reason']}", flush=True)
                continue
            cached, source = prepare_cache(sequence, destination, mode)
            sources.append(source)
            for seed in seeds:
                for name in variants:
                    config = variant_config(name, seed, rigidity_weight)
                    predictions = destination / "predictions" / f"{sequence.name}-{name}-{seed}.txt"
                    result = run_sequence(sequence, config, cached, predictions)
                    result["variant"] = name
                    result["baseline_settings"] = (
                        "BoT-SORT sparseOptFlow GMC, ReID=False; "
                        "DeepSORT compact HSV/DCT descriptor; no learned encoder in this comparison"
                    )
                    results.append(result)
                    (destination / "progress.json").write_text(
                        json.dumps(
                            {
                                "status": "in_progress",
                                "completed_runs": len(results),
                                "results": results,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(
                        f"{mode} {sequence.name} seed={seed} {name}: "
                        f"HOTA={result['metrics']['HOTA']:.2f} "
                        f"IDF1={result['metrics']['IDF1']:.2f} "
                        f"IDSW={result['metrics']['identity_switches']} "
                        f"count_error={result['metrics']['count_error']}",
                        flush=True,
                    )
        metadata = {
            "executed_at": datetime.now(UTC).isoformat(),
            "experiment": "local_rigidity_static_scene",
            "oracle_detections": mode == "oracle",
            "detection_mode": mode,
            "interpretation": (
                "Controlled local-rigidity and static-scene comparison on two accessible "
                "BUTom-ST21 tomato development subsets. Oracle boxes discard identities "
                "before tracking; actual cached detections are a separate transfer diagnostic. "
                "No blueberry accuracy, official benchmark, independent test or superiority "
                "claim follows from these experiments."
            ),
            "environment": _environment(),
            "implementation": implementation,
            "sources": sources,
            "seeds": seeds,
            "variants": variants,
            "fixed_rigidity_weight": rigidity_weight,
            "aggregate_counts": aggregate_counts(results),
            "rigidity_execution": rigidity_execution_evidence(results),
            "execution_mode": "serial CPU; no GPU allocation",
            "limitations": [
                "Both subsets are from the publisher's evaluation partition, "
                "used here for development.",
                "Sequence205 was used for separate encoder development; "
                "that encoder is not used here.",
                "Hyperparameters are fixed before execution; "
                "this is not a test-set selection sweep.",
                "Static geometry is an assumption to test; "
                "occlusion, moving foliage and parallax can break it.",
                "Identical cache hashes and compact descriptors "
                "are used within each sequence/mode.",
                "Warm association excludes first five frames; full/cold timing remains in results.",
                "Seed deviation is not uncertainty across independently sampled videos.",
            ],
        }
        if results:
            write_reports(results, destination, metadata)
            (destination / "progress.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "completed_runs": len(results),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        summaries[mode] = {"completed_runs": len(results), "sources": sources}
    index = {"status": "complete", "modes": summaries, "pending": skipped}
    (output / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=Path("results/rigidity"))
    parser.add_argument("--mode", choices=["oracle", "cached", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--variants", choices=list(VARIANTS), nargs="+", default=list(VARIANTS))
    parser.add_argument("--rigidity-weight", type=float, default=0.35)
    parser.add_argument(
        "--audit-results", type=Path, help="Audit real result counters without running tracking"
    )
    parser.add_argument(
        "--require-supported-fits",
        action="store_true",
        help="Fail the audit if supported local geometry was never exercised",
    )
    args = parser.parse_args()
    if args.audit_results:
        payload = json.loads(args.audit_results.read_text(encoding="utf-8"))
        evidence = rigidity_execution_evidence(payload["results"])
        print(json.dumps(evidence, indent=2))
        if args.require_supported_fits and evidence["status"] != "exercised":
            raise SystemExit("No measured support: do not claim the geometry method was exercised")
        return
    sequences = args.sequence or [
        Path("data/processed/butom-st21-235"),
        Path("data/processed/butom-st21-205"),
    ]
    modes = ["oracle", "cached"] if args.mode == "both" else [args.mode]
    run_comparison(sequences, args.output, modes, args.seeds, args.variants, args.rigidity_weight)


if __name__ == "__main__":
    main()
