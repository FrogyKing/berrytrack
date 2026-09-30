"""Conservative, persistent reservations: failed jobs do not release their allowance."""

import json
import time
import uuid
from pathlib import Path
from typing import Any

from filelock import FileLock

RATES = {"T4": 0.59, "L4": 0.80, "A10G": 1.10}
HARD_BUDGET_USD = 30.0
SAFETY_RESERVE_USD = 2.0


def estimate_cost(gpu: str, seconds: int, startup_seconds: int = 600) -> float:
    if gpu not in RATES or not 1 <= seconds <= 900:
        raise ValueError("Supported GPUs: T4, L4, A10G; execution limit: 1--900 seconds")
    # Account at maximum allowed 2 CPU cores and 8 GiB RAM, plus 2 seconds scale-down.
    hourly = RATES[gpu] + 2 * 0.04730 + 8 * 0.008
    return round(hourly * (seconds + startup_seconds + 2) / 3600 + 0.05, 6)


def reserve(path: Path, gpu: str, seconds: int, kind: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        data: dict[str, Any] = (
            json.loads(path.read_text()) if path.exists() else {"budget_usd": 30, "runs": []}
        )
        # A caller cannot raise the hard cap by modifying the ledger's budget field.
        used = sum(float(r["reserved_usd"]) for r in data["runs"])
        amount = estimate_cost(gpu, seconds)
        if used + amount > HARD_BUDGET_USD - SAFETY_RESERVE_USD:
            raise RuntimeError(f"Budget exhausted: reserved ${used:.2f}, requested ${amount:.2f}")
        record = {
            "run_id": uuid.uuid4().hex,
            "kind": kind,
            "gpu": gpu,
            "timeout_seconds": seconds,
            "reserved_usd": amount,
            "created_unix": time.time(),
        }
        data["runs"].append(record)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2))
        temp.replace(path)
    return record
