"""Reserve worst-case spend before building/running any cloud application."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from berrytrack.infrastructure.budget import reserve


def launch(kind: str, options: dict, gpu: str = "T4", seconds: int = 300) -> dict:
    # Windows pipes commonly default to cp1252; Modal emits Unicode status symbols.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    run = reserve(Path("artifacts/cloud-budget.json"), gpu, seconds, kind)
    print(json.dumps({"reservation": run, "maximum_total_usd": 30}, indent=2), flush=True)
    os.environ["BERRYTRACK_GPU"] = gpu
    os.environ["BERRYTRACK_TIMEOUT"] = str(seconds)
    # Modal hashes uploads while building; freeze source so local edits cannot race the build.
    snapshot = Path("artifacts/builds") / run["run_id"]
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        Path(__file__).resolve().parents[1],
        snapshot / "berrytrack",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    os.environ["BERRYTRACK_SOURCE"] = str((snapshot / "berrytrack").resolve())
    destination = Path("artifacts/modal") / run["run_id"]
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "reservation": run,
        "options": options,
        "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(snapshot.rglob("*.py"))
        },
    }
    (destination / "request.json").write_text(json.dumps(manifest, indent=2))
    import modal

    from berrytrack.infrastructure.modal_app import app, gpu_job

    try:
        with modal.enable_output(), app.run():
            result = gpu_job.remote(kind, run, options)
    except Exception as error:
        (destination / "failure.json").write_text(
            json.dumps({"error": str(error), "run_id": run["run_id"]}, indent=2)
        )
        raise
    (destination / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result
