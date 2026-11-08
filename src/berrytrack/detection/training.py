"""Prepare explicit image splits and run bounded YOLO domain adaptation."""

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, cast

import yaml

CLASS_NAMES = ["Unblue_visible", "Unblue_occluded", "Blue_visible", "Blue_occluded"]


def via_labels(source: Path) -> tuple[str, dict]:
    """Reconstruct the four classes from VIA; publisher TXT collapses them to 0/3."""
    from PIL import Image

    annotation = source.with_suffix(".json")
    document = json.loads(annotation.read_text())
    record = next(
        v for v in document.values() if isinstance(v, dict) and v.get("filename") == source.name
    )
    regions = record["regions"]
    shapes, attributes = regions["shape_attributes"], regions["region_attributes"]
    if len(shapes) != len(attributes):
        raise ValueError(f"VIA shapes/attributes mismatch: {source}")
    with Image.open(source) as image:
        width, height = image.size
    labels = []
    counts = dict.fromkeys(CLASS_NAMES, 0)
    for box, attrs in zip(shapes, attributes, strict=True):
        selected = [k for k, value in attrs["Blueberry"].items() if value is True]
        if len(selected) != 1 or selected[0] not in CLASS_NAMES or box["name"] != "rect":
            raise ValueError(f"Unknown VIA annotation: {source}")
        name = selected[0]
        x, y, w, h = (float(box[k]) for k in ("x", "y", "width", "height"))
        if w <= 0 or h <= 0:
            raise ValueError("Invalid VIA box")
        labels.append(
            f"{CLASS_NAMES.index(name)} {(x + w / 2) / width:.8f} "
            f"{(y + h / 2) / height:.8f} {w / width:.8f} {h / height:.8f}"
        )
        counts[name] += 1
    return "\n".join(labels) + "\n", counts


def prepare_detection_split(root: Path, output: Path, max_train: int | None = None) -> dict:
    """Deterministic image-level development split; capture groups unavailable.

    Images from one plant may appear in different splits; this is a pipeline
    smoke protocol, not a leakage-controlled agricultural generalization study.
    """
    images = sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.with_suffix(".txt").exists()
    )
    if len(images) < 5:
        raise ValueError("Need at least five annotated images")
    import numpy as np

    rng = np.random.default_rng(42)
    order = rng.permutation(len(images))
    ntrain, nval = int(len(images) * 0.7), int(len(images) * 0.15)
    splits = {
        "train": order[:ntrain],
        "val": order[ntrain : ntrain + nval],
        "test": order[ntrain + nval :],
    }
    if len({p.name for p in images}) != len(images):
        raise ValueError("Source image basenames must be unique")
    planned = {
        split: [
            images[i] for i in (indices[:max_train] if max_train and split == "train" else indices)
        ]
        for split, indices in splits.items()
    }
    expected_images = {
        f"images/{split}/{p.name}" for split, paths in planned.items() for p in paths
    }
    existing_images = {
        p.relative_to(output).as_posix() for p in (output / "images").rglob("*") if p.is_file()
    }
    expected_labels = {
        f"labels/{split}/{p.stem}.txt" for split, paths in planned.items() for p in paths
    }
    existing_labels = {p.relative_to(output).as_posix() for p in (output / "labels").rglob("*.txt")}
    if (existing_images and existing_images != expected_images) or (
        existing_labels and existing_labels != expected_labels
    ):
        raise ValueError("Output contains a different split; choose a fresh output directory")
    manifest: dict[str, list[str]] = {}
    for split, indices in splits.items():
        if max_train and split == "train":
            indices = indices[:max_train]
        manifest[split] = []
        for i in indices:
            source = images[i]
            target = output / "images" / split / source.name
            label = output / "labels" / split / source.with_suffix(".txt").name
            target.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            label_text, _ = via_labels(source)
            label.write_text(label_text)
            manifest[split].append(str(source))
    config = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(config))
    (output / "splits.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "splits": manifest,
                "limitation": "image-level smoke split; plant groups unavailable",
                "label_source": "original VIA JSON; publisher TXT class collapse corrected",
                "via_sha256": {
                    str(p): hashlib.sha256(p.with_suffix(".json").read_bytes()).hexdigest()
                    for p in images
                },
            },
            indent=2,
        )
    )
    return {k: len(v) for k, v in manifest.items()}


def cloud_detector_job(kind: str, options: dict, timeout: int, started: float) -> dict:
    from ultralytics import YOLO

    from berrytrack.infrastructure.modal_app import _relative

    root = Path("/datasets") / _relative(options.get("dataset", "blueberrydcm-yolo"))
    data_config = yaml.safe_load((root / "dataset.yaml").read_text())
    data_config["path"] = str(root)
    (root / "cloud.yaml").write_text(yaml.safe_dump(data_config))
    experiment = _relative(options.get("experiment", "blueberry-smoke"))
    output = Path("/models") / experiment
    weights = options.get("weights", "yolo11n.pt")
    if weights.startswith("/"):
        if not weights.startswith("/models/"):
            raise ValueError("Cloud weights must be in /models")
    model = YOLO(weights)
    if kind == "train-detector":

        def stop_at_limit(trainer):
            if time.perf_counter() - started > timeout - 60:
                trainer.stop = True

        last = output / "weights/resume.pt"
        resume = bool(options.get("resume", False))
        if resume:
            if not last.exists():
                raise FileNotFoundError(last)
            model = YOLO(str(last))
        model.add_callback("on_train_batch_end", stop_at_limit)

        def persist_resume(trainer):
            from berrytrack.infrastructure.modal_app import models

            last.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trainer.last, last)
            models.commit()

        model.add_callback("on_model_save", persist_resume)
        model.train(
            data=str(root / "cloud.yaml"),
            epochs=min(int(options.get("epochs", 1)), 20),
            imgsz=int(options.get("image_size", 640)),
            batch=4,
            device=0,
            workers=0,
            project=str(output.parent),
            name=output.name,
            exist_ok=True,
            resume=resume,
            seed=42,
            deterministic=True,
            plots=False,
            amp=False,
        )
        return {
            "checkpoint": str(output / "weights/best.pt"),
            "epochs_requested": options.get("epochs", 1),
            "epochs_completed": int(cast(Any, model).trainer.epoch + 1),
            "parameters": sum(p.numel() for p in cast(Any, model).model.parameters()),
            "metrics": {k: float(v) for k, v in cast(Any, model).trainer.metrics.items()},
            "protocol": "BlueberryDCM image-level smoke adaptation; not final-test evidence",
        }
    metrics = model.val(
        data=str(root / "cloud.yaml"),
        split="val",
        imgsz=int(options.get("image_size", 640)),
        batch=4,
        device=0,
        workers=0,
        plots=False,
        project="/artifacts/detector-validation",
        name=experiment,
        exist_ok=True,
    )
    return {
        "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
        "protocol": "BlueberryDCM validation detection metrics only",
        "weights": weights,
    }
