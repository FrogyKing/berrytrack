"""Run identical cached detections through baselines and BerryTrack ablations.

Usage: uv run python -m berrytrack.evaluation.benchmark --sequence PATH --oracle
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from berrytrack.config import TrackerConfig
from berrytrack.data.mot import MOTSequence, read_mot, write_mot
from berrytrack.evaluation.metrics import evaluate_mot
from berrytrack.types import Detections

ABLATIONS: dict[str, dict[str, Any]] = {
    "appearance_only": dict(
        appearance=True,
        motion=False,
        camera_compensation=False,
        context=False,
        adaptive=False,
        uncertainty_trigger=False,
    ),
    "motion_only": dict(
        appearance=False,
        motion=True,
        camera_compensation=False,
        context=False,
        adaptive=False,
        uncertainty_trigger=False,
    ),
    "appearance_motion": dict(
        appearance=True,
        motion=True,
        camera_compensation=False,
        context=False,
        adaptive=False,
        uncertainty_trigger=False,
    ),
    "camera": dict(
        appearance=True,
        motion=True,
        camera_compensation=True,
        context=False,
        adaptive=False,
        uncertainty_trigger=False,
    ),
    "context": dict(
        appearance=True,
        motion=True,
        camera_compensation=True,
        context=True,
        adaptive=False,
        uncertainty_trigger=False,
    ),
    "adaptive": dict(
        appearance=True,
        motion=True,
        camera_compensation=True,
        context=True,
        adaptive=True,
        uncertainty_trigger=False,
    ),
    "uncertainty": dict(
        appearance=True,
        motion=True,
        camera_compensation=True,
        context=True,
        adaptive=True,
        uncertainty_trigger=True,
    ),
}

# Preserve the original ladder's meaning: each historical component was tested
# before the quasi-static/polygon extension existed.
for _settings in ABLATIONS.values():
    _settings.update(local_rigidity=False, static_scene=False)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def mot_detections(rows: np.ndarray) -> Detections:
    if not len(rows):
        return Detections.empty()
    boxes = rows[:, 2:6].astype(np.float32).copy()
    boxes[:, :2] -= 1  # MOT disk coordinates -> zero-based internal pixels
    boxes[:, 2:] += boxes[:, :2]
    return Detections(boxes, np.clip(rows[:, 6], 0, 1).astype(np.float32))


def _environment() -> dict[str, str]:
    packages = (
        "berrytrack",
        "numpy",
        "scipy",
        "torch",
        "ultralytics",
        "trackeval",
        "deep-sort-realtime",
    )
    versions = {name: importlib.metadata.version(name) for name in packages}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "opencv_threads": str(cv2.getNumThreads()),
        **versions,
    }


def source_snapshot() -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {
            str(path.relative_to(package)).replace("\\", "/"): sha256(path)
            for path in sorted(package.rglob("*.py"))
        },
        "uv_lock_sha256": sha256(Path("uv.lock")) if Path("uv.lock").exists() else None,
    }


def run_sequence(
    sequence: MOTSequence,
    config: TrackerConfig,
    cached: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    """Association timing includes features and GMC; excludes disk IO/metrics."""
    from berrytrack.tracking import create_tracker

    np.random.seed(config.seed)
    tracker = create_tracker(config, fps=sequence.fps)
    predictions: list[list[float]] = []
    durations = []
    wall_start = time.perf_counter()
    for frame_number in range(1, sequence.length + 1):
        path = sequence.image_path(frame_number)
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Unreadable sequence frame: {path}")
        det = mot_detections(cached[cached[:, 0] == frame_number])
        start = time.perf_counter()
        tracks = tracker.update(det, np.asarray(frame, dtype=np.uint8))
        durations.append(time.perf_counter() - start)
        for track in tracks:
            if not track.confirmed:
                continue
            x1, y1, x2, y2 = track.bbox
            predictions.append(
                [
                    frame_number,
                    track.track_id,
                    float(x1 + 1),
                    float(y1 + 1),
                    float(x2 - x1),
                    float(y2 - y1),
                    track.score,
                    -1,
                    -1,
                    -1,
                ]
            )
    elapsed = time.perf_counter() - wall_start
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1, 10)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_mot(output, pred)
    metrics = evaluate_mot(sequence.gt, pred, sequence.length)
    association_seconds = sum(durations)
    warmup_frames = min(5, max(0, sequence.length - 1))
    warm_durations = durations[warmup_frames:]
    warm_seconds = sum(warm_durations)
    stats: Any = getattr(tracker, "stats", None)
    appearance_crops = getattr(tracker, "appearance_crops", None)
    if appearance_crops is None and stats is not None:
        appearance_crops = getattr(stats, "appearance_crops", None)
    return {
        "sequence": sequence.name,
        "algorithm": config.algorithm,
        "seed": config.seed,
        "config": config.model_dump(),
        "frames": sequence.length,
        "association_ms": 1000 * warm_seconds / len(warm_durations),
        "association_all_frames_ms": 1000 * association_seconds / sequence.length,
        "cold_first_frame_ms": 1000 * durations[0],
        "latency_warmup_frames_excluded": warmup_frames,
        "per_frame_association_ms": [1000 * duration for duration in durations],
        "association_p95_ms": float(1000 * np.percentile(durations, 95)),
        "tracking_fps": len(warm_durations) / max(warm_seconds, 1e-12),
        "tracking_all_frames_fps": sequence.length / max(association_seconds, 1e-12),
        "pipeline_ms": 1000 * elapsed / sequence.length,
        "pipeline_scope": (
            "cached association + image decode + result collection; detector excluded"
        ),
        "end_to_end_ms": None,
        "peak_gpu_memory_bytes": None,
        "model_parameter_count": 0,
        "approximate_gpu_cost_usd": 0.0,
        "appearance_crops": appearance_crops,
        "rigidity_fits": getattr(tracker, "rigidity_fits", 0),
        "rigidity_anchor_count": getattr(tracker, "rigidity_anchor_count", 0),
        "metrics": metrics,
        "prediction_sha256": sha256(output),
    }


def write_reports(results: list[dict[str, Any]], output: Path, metadata: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2), encoding="utf-8"
    )
    fields = [
        "sequence",
        "variant",
        "algorithm",
        "seed",
        "HOTA",
        "DetA",
        "AssA",
        "IDF1",
        "MOTA",
        "identity_switches",
        "fragmentations",
        "gt_unique_count",
        "pred_unique_count",
        "count_mae",
        "count_rmse",
        "association_ms",
        "tracking_fps",
        "appearance_crops",
    ]
    flat = []
    for result in results:
        row = {**result, **result["metrics"]}
        flat.append({field: row.get(field) for field in fields})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in flat:
        groups.setdefault((row["sequence"], row["variant"]), []).append(row)
    averages = []
    for (sequence_name, variant), group in groups.items():
        summary: dict[str, Any] = {
            "sequence": sequence_name,
            "variant": variant,
            "seeds": len(group),
        }
        for key in fields[4:]:
            values = [row[key] for row in group if row[key] is not None]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values, ddof=1 if len(values) > 1 else 0))
        averages.append(summary)
    if averages:
        with (output / "summary_by_variant.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(averages[0]))
            writer.writeheader()
            writer.writerows(averages)
    table = [
        "# Executed association diagnostic",
        "",
        metadata["interpretation"],
        "",
        "Percent metrics use 0–100; latency excludes detector, disk IO and the first 5 frames. "
        "Single-sequence count MAE/RMSE are absolute final-count errors, "
        "not population estimates.",
        "",
        "| Sequence | Variant | Seed | HOTA | IDF1 | IDSW | Count error | Association ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in flat:
        table.append(
            f"| {row['sequence']} | {row['variant']} | {row['seed']} | {row['HOTA']:.2f} | "
            f"{row['IDF1']:.2f} | {row['identity_switches']} | "
            f"{row['count_mae']:.0f} | {row['association_ms']:.3f} |"
        )
    table.extend(
        [
            "",
            "Executed at "
            + metadata["executed_at"]
            + ". See results.json for environment, hashes, settings, coverage and limitations.",
        ]
    )
    (output / "report.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['variant']}\n{r['sequence']}" for r in averages]
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(averages) * 0.9), 5.4))
    axes[0].bar(
        range(len(averages)),
        [r["HOTA_mean"] for r in averages],
        yerr=[r["HOTA_std"] for r in averages],
        color="#5031a5",
        capsize=2,
    )
    axes[0].set_ylabel("HOTA (%)")
    axes[0].set_ylim(0, 100)
    axes[1].bar(
        range(len(averages)),
        [r["association_ms_mean"] for r in averages],
        yerr=[r["association_ms_std"] for r in averages],
        color="#128c83",
        capsize=2,
    )
    axes[1].set_ylabel("Association latency (ms/frame)")
    for axis in axes:
        axis.set_xticks(range(len(averages)), labels, rotation=55, ha="right", fontsize=8)
    scope = "Oracle-box" if metadata.get("oracle_detections") else "Cached-detection"
    fig.suptitle(f"{scope} diagnostic: means +/- seed SD, not video uncertainty")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def aggregate_counts(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Final-count errors across sequences, grouped by variant and seed."""
    groups: dict[tuple[str, int], list[float]] = {}
    for result in results:
        key = (result["variant"], result["seed"])
        groups.setdefault(key, []).append(float(result["metrics"]["count_error"]))
    return [
        {
            "variant": variant,
            "seed": seed,
            "sequences": len(errors),
            "count_mae": float(np.mean(np.abs(errors))),
            "count_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        }
        for (variant, seed), errors in groups.items()
    ]


def benchmark(
    sequence_paths: list[Path],
    output: Path,
    oracle: bool = False,
    ablations: bool = False,
    seeds: list[int] | None = None,
    algorithms: list[str] | None = None,
    dimensions: list[int] | None = None,
    legacy: bool = False,
) -> dict[str, Any]:
    """Run a serial, CPU-only benchmark; no GPU calls or fabricated detections."""
    seeds = seeds or [42]
    algorithms = algorithms or ["bytetrack", "botsort", "deepsort", "berrytrack"]
    output.mkdir(parents=True, exist_ok=True)
    implementation = source_snapshot()
    results = []
    sources = []
    for root in sequence_paths:
        seq = MOTSequence.load(root)
        if oracle:
            cached = seq.gt.copy()
            cached[:, 1] = -1  # explicitly discard identity before tracker input
            cached[:, 6] = 1
            cache_path = output / "cache" / f"{seq.name}.txt"
            write_mot(cache_path, cached)
        else:
            cache_path = root / "det" / "det.txt"
            cached = read_mot(cache_path)
        sources.append(
            {
                "sequence": seq.name,
                "ground_truth_sha256": sha256(root / "gt" / "gt.txt"),
                "detections_sha256": sha256(cache_path),
                "seqinfo_sha256": sha256(root / "seqinfo.ini"),
                "root": str(root.resolve()),
                "frames": seq.length,
            }
        )
        for seed in seeds:
            variants: list[tuple[str, TrackerConfig]] = [
                (
                    algorithm,
                    TrackerConfig(
                        algorithm=algorithm,
                        seed=seed,
                        local_rigidity=not legacy,
                        static_scene=False,
                    ),
                )
                for algorithm in algorithms
            ]
            if ablations:
                variants += [
                    (name, TrackerConfig(seed=seed, **settings))
                    for name, settings in ABLATIONS.items()
                ]
            if dimensions:
                variants += [
                    (
                        f"embedding_{dim}",
                        TrackerConfig(
                            seed=seed,
                            embedding_dim=dim,
                            local_rigidity=not legacy,
                            static_scene=False,
                        ),
                    )
                    for dim in dimensions
                ]
            for name, config in variants:
                result = run_sequence(
                    seq, config, cached, output / "predictions" / f"{seq.name}-{name}-{seed}.txt"
                )
                result["variant"] = name
                result["baseline_settings"] = (
                    "BoT-SORT GMC=sparseOptFlow, ReID=False; "
                    "DeepSORT embedder=compact HSV/DCT (no learned ReID weights)"
                )
                results.append(result)
                print(
                    f"{seq.name} {name}: HOTA={result['metrics']['HOTA']:.2f} "
                    f"IDF1={result['metrics']['IDF1']:.2f}",
                    flush=True,
                )
    metadata = {
        "executed_at": datetime.now(UTC).isoformat(),
        "oracle_detections": oracle,
        "legacy_association": legacy,
        "interpretation": (
            "Oracle boxes from human-corrected tracking GT; IDs removed from tracker input. "
            "This isolates association on an accessible tomato development subset, and is "
            "not detector performance, blueberry evaluation, or held-out superiority evidence."
            if oracle
            else "Cached detector association experiment. Detector quality must be reported "
            "separately; consult detection cache provenance."
        ),
        "environment": _environment(),
        "implementation": implementation,
        "sources": sources,
        "seeds": seeds,
        "aggregate_counts": aggregate_counts(results),
        "execution_mode": "serial CPU",
        "limitations": [
            "No ignore-region protocol: converter must select foreground objects.",
            "One accessible development sequence is insufficient to accept H1–H4.",
            "No model selection on held-out test data was performed.",
            "CPU timings are machine-specific. Warm association excludes first 5 frames; "
            "full online timing and cold first-frame latency are retained separately. "
            "End-to-end detector latency and GPU memory are not measured here.",
        ],
    }
    write_reports(results, output, metadata)
    return {"metadata": metadata, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark"))
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Use GT boxes with identities discarded; association diagnostic only",
    )
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument(
        "--legacy", action="store_true", help="Reproduce pre-polygon BerryTrack settings"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--algorithms", nargs="+", choices=["bytetrack", "botsort", "deepsort", "berrytrack"]
    )
    parser.add_argument("--dimensions", type=int, nargs="+", choices=[32, 64, 128])
    args = parser.parse_args()
    benchmark(
        args.sequence,
        args.output,
        args.oracle,
        args.ablations,
        args.seeds,
        args.algorithms,
        args.dimensions,
        args.legacy,
    )


if __name__ == "__main__":
    main()
