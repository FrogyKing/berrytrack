"""Collect real run metadata and small aggregates, never model/data payloads."""

import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from berrytrack.detection.training import via_labels


def public_metadata(value):
    """Publish experiment provenance without workstation accounting records."""
    if isinstance(value, dict):
        return {
            ("job" if key == "reservation" else key): public_metadata(item)
            for key, item in value.items()
            if not key.endswith("_usd") and key not in {"billing_note", "cost_scope"}
        }
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    return value


def collect() -> None:
    output = Path("results/cloud")
    output.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(Path("artifacts/cloud-budget.json").read_text())
    records = []
    for record in ledger["runs"]:
        root = Path("artifacts/modal") / record["run_id"]
        result = root / "result.json"
        entry = {
            "reservation": record,
            "status": "completed" if result.exists() else "failed_before_result",
        }
        if result.exists():
            entry["result"] = json.loads(result.read_text())
            if "blueberry-smoke" in str(entry["result"]):
                entry["interpretation"] = (
                    "Excluded initial pilot: publisher TXT collapsed class taxonomy. "
                    "Training time option overrode requested3 epochs;105 CSV epoch records "
                    "within the hard allocation. Corrected VIA training is a separate run."
                )
        request = root / "request.json"
        if request.exists():
            entry["request"] = json.loads(request.read_text())
        snapshot = Path("artifacts/builds") / record["run_id"] / "berrytrack"
        if snapshot.exists():
            entry["source_snapshot_sha256"] = {
                str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(snapshot.rglob("*.py"))
            }
        records.append(entry)
    summary = {
        "collected_utc": datetime.now(UTC).isoformat(),
        "runs": records,
    }
    (output / "runs.json").write_text(json.dumps(public_metadata(summary), indent=2))
    detector = Path("results/detection")
    detector.mkdir(parents=True, exist_ok=True)
    shutil.copy2("artifacts/detector-via-results.csv", detector / "training.csv")
    shutil.copy2("artifacts/detector-results.csv", detector / "excluded-pilot-training.csv")
    shutil.copy2("artifacts/detector-args.yaml", detector / "excluded-pilot-args.yaml")
    native, via = Counter(), Counter()
    for image in Path("data/processed/blueberrydcm").rglob("*.jpg"):
        native.update(
            line.split()[0]
            for line in image.with_suffix(".txt").read_text().splitlines()
            if line.strip()
        )
        via.update(via_labels(image)[1])
    splits = json.loads(Path("data/processed/blueberrydcm-via-yolo/splits.json").read_text())
    data = {
        "native_txt_class_counts": dict(native),
        "via_class_counts": dict(via),
        "annotation_source": "VIA JSON; source TXT collapses four categories",
        "native_total": sum(native.values()),
        "via_total": sum(via.values()),
        "limitation": (
            "Image-level development split, capture groups unknown; source formats also "
            "differ in box count; no temporal identities"
        ),
        "split_seed": 42,
        "split_files": {k: [Path(p).name for p in v] for k, v in splits["splits"].items()},
        "weights_sha256": hashlib.sha256(
            Path("checkpoints/blueberry-via/best.pt").read_bytes()
        ).hexdigest(),
        "test_evaluated": False,
    }
    (detector / "audit.json").write_text(json.dumps(data, indent=2))
    cache = Path("data/processed/butom-st21-235/det/det.json")
    if cache.exists():
        transfer = json.loads(cache.read_text())
        transfer["weights_sha256"] = data["weights_sha256"]
        transfer["scope"] = (
            "Blueberry-trained detector transferred to tomato sequence235; "
            "domain-shift diagnostic only"
        )
        Path("results/detector-transfer/detector.json").write_text(json.dumps(transfer, indent=2))


if __name__ == "__main__":
    collect()
