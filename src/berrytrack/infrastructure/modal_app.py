"""Ephemeral Modal Functions. Launch through infrastructure.launch for budget enforcement."""

import os
from pathlib import Path
from typing import Any

import modal

from berrytrack.infrastructure.budget import RATES

GPU = os.environ.get("BERRYTRACK_GPU", "T4")
TIMEOUT = int(os.environ.get("BERRYTRACK_TIMEOUT", "300"))
if GPU not in RATES or not 30 <= TIMEOUT <= 900:
    raise ValueError("Invalid bounded Modal GPU/timeout configuration")

app = modal.App("berrytrack-research", include_source=False)
datasets = modal.Volume.from_name("berrytrack-datasets", create_if_missing=True)
models = modal.Volume.from_name("berrytrack-models", create_if_missing=True)
artifacts = modal.Volume.from_name("berrytrack-artifacts", create_if_missing=True)
features = modal.Volume.from_name("berrytrack-features", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .uv_sync(frozen=True, extra_options="--no-dev", uv_version="0.12.5")
    .env(
        {
            "YOLO_AUTOINSTALL": "false",
            "YOLO_CONFIG_DIR": "/models/yolo-config",
            "OMP_NUM_THREADS": "2",
            "BERRYTRACK_GPU": GPU,
            "BERRYTRACK_TIMEOUT": str(TIMEOUT),
        }
    )
)
if modal.is_local():
    image = image.add_local_dir(
        os.environ.get("BERRYTRACK_SOURCE", str(Path(__file__).resolve().parents[1])),
        "/root/berrytrack",
    )


@app.function(
    image=image,
    cpu=(1, 2),
    memory=(2048, 8192),
    timeout=180,
    startup_timeout=600,
    max_containers=1,
    scaledown_window=2,
    retries=0,
    volumes={"/datasets": datasets},
)
def preprocess(sequence: str) -> dict:
    """CPU-only data integrity inspection, separate from the GPU allocation."""
    from berrytrack.data.mot import MOTSequence

    root = Path("/datasets") / sequence
    seq = MOTSequence.load(root)
    missing = [i for i in range(1, seq.length + 1) if not seq.image_path(i).exists()]
    if missing:
        raise ValueError(f"Missing frames: {missing[:10]}")
    return {"sequence": seq.name, "frames": seq.length, "annotations": len(seq.gt)}


@app.function(
    image=image,
    gpu=GPU,
    cpu=(1, 2),
    memory=(2048, 8192),
    timeout=TIMEOUT,
    startup_timeout=600,
    max_containers=1,
    scaledown_window=2,
    retries=0,
    volumes={
        "/datasets": datasets,
        "/models": models,
        "/artifacts": artifacts,
        "/features": features,
    },
)
def gpu_job(kind: str, run: dict, options: dict) -> dict:
    import json
    import time

    import torch

    from berrytrack.infrastructure.budget import estimate_cost, reserve

    if run["gpu"] != GPU or run["timeout_seconds"] != TIMEOUT:
        raise ValueError("Reservation does not match allocation")
    artifacts.reload()
    # One worker serializes reservations across this app; no sweeps or async fan-out.
    reserve(Path("/artifacts/budget.json"), GPU, TIMEOUT, kind)
    artifacts.commit()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU job requires real CUDA; refusing CPU fallback")
    torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(2)
    output = Path("/artifacts") / run["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    if kind == "smoke":
        from berrytrack.embeddings.model import CompactEncoder

        model = CompactEncoder(64).cuda()
        with torch.inference_mode():
            encoded = model(torch.rand(8, 3, 32, 32, device="cuda"))
        torch.cuda.synchronize()
        result: dict[str, Any] = {
            "shape": list(encoded.shape),
            "note": "random tensor infrastructure smoke only",
        }
    elif kind == "train-encoder":
        from berrytrack.embeddings.training import train_encoder

        sequence = _relative(options["sequence"])
        target = Path("/models") / _relative(options.get("experiment", "encoder-dev"))
        result = train_encoder(
            [Path("/datasets") / sequence],
            target,
            dimension=int(options.get("dimension", 64)),
            steps=min(int(options.get("steps", 30)), 1000),
            device="cuda",
            seed=int(options.get("seed", 42)),
            max_seconds=max(1, TIMEOUT - 45),
        )
        models.commit()
    elif kind == "infer-encoder":
        from berrytrack.data.mot import MOTSequence
        from berrytrack.embeddings.training import TorchAppearance

        seq = MOTSequence.load(Path("/datasets") / _relative(options["sequence"]))
        encoder = TorchAppearance(Path("/models") / _relative(options["checkpoint"]), "cuda")
        import cv2
        import numpy as np

        embeddings = []
        gt = seq.gt
        for frame_id in range(1, min(seq.length, int(options.get("frames", 16))) + 1):
            if time.perf_counter() - started > TIMEOUT - 30:
                break
            frame = cv2.imread(str(seq.image_path(frame_id)))
            rows = gt[gt[:, 0] == frame_id]
            boxes = rows[:, 2:6].astype(np.float32)
            boxes[:, :2] -= 1
            boxes[:, 2:] += boxes[:, :2]
            embeddings.append(encoder.extract(frame, boxes))
        target = Path("/features") / f"{run['run_id']}.npz"
        np.savez_compressed(target, **{f"frame_{i + 1}": f for i, f in enumerate(embeddings)})
        features.commit()
        result = {
            "frames": len(embeddings),
            "crops": sum(len(f) for f in embeddings),
            "cache": str(target),
        }
    elif kind in {"train-detector", "infer-detector"}:
        from berrytrack.detection.training import cloud_detector_job

        result = cloud_detector_job(kind, options, TIMEOUT, started)
        models.commit()
        features.commit()
    else:
        raise ValueError(f"Unknown job kind: {kind}")
    torch.cuda.synchronize()
    runtime = time.perf_counter() - started
    result.update(
        {
            "run_id": run["run_id"],
            "kind": kind,
            "gpu": torch.cuda.get_device_name(0),
            "runtime_seconds": runtime,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "reserved_usd": run["reserved_usd"],
            "estimated_runtime_usd": estimate_cost(
                GPU, min(TIMEOUT, max(1, int(runtime))), startup_seconds=0
            )
            - 0.05,
            "billing_note": (
                "approximate runtime estimate excludes build/startup; "
                "reservation includes allowance"
            ),
        }
    )
    (output / "result.json").write_text(json.dumps(result, indent=2))
    artifacts.commit()
    return result


def _relative(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ValueError("Volume paths must be relative and cannot traverse directories")
    return str(path)
