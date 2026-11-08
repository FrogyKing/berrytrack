"""Verified public releases and economical, provenance-preserving ingestion."""

from __future__ import annotations

import io
import json
import pickle
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from berrytrack.data.download import download, extract_archive, file_digest, safe_archive_path
from berrytrack.data.mot import write_mot

BUTOM_URL = "https://bonndata.uni-bonn.de/api/access/datafile/"
BUTOM_ANNOTATION_MD5 = "d7ef688a68e68ad43bbf69c154a0905c"
DCM_URL = "https://zenodo.org/api/records/14002517/files/BlueberryDCM.7z/content"
DCM_MD5 = "77fc808520dbe7751cb367dbd06ea32f"


class ArrayUnpickler(pickle.Unpickler):
    """Allow only numpy array serialization primitives, never arbitrary globals.

    Only use for checksum-verified publisher archives; pickle remains a poor
    interchange format. Converted MOT files contain no executable serialization.
    """

    def find_class(self, module: str, name: str) -> Any:
        allowed = {
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "scalar"),
            ("numpy.core.multiarray", "scalar"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")
        return super().find_class(module, name)


def stream_butom_rgb(raw: Path, sequences: tuple[str, ...] = ("235", "205")) -> None:
    """Extract the first two official eval sequences without downloading 26 GB.

    Release-v1 RGB archive starts with 235 then 205. Stop at the next sequence.
    The full compressed-part MD5 cannot be verified for a prefix; record individual
    SHA256 checksums, TLS origin and this integrity limitation in the manifest.
    """
    expected = set(sequences)
    if all(len(list((raw / "rgb" / seq).glob("*.png"))) == 83 for seq in sequences):
        return
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl required for bounded streaming RGB subset download")
    url = BUTOM_URL + "54924"
    process = subprocess.Popen(
        [curl, "--silent", "--show-error", "--fail", "--location", "--max-time", "180", url],
        stdout=subprocess.PIPE,
    )
    files = []
    count: dict[str, int] = {seq: 0 for seq in sequences}
    total = 0
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                safe_archive_path(raw, member.name)
                if len(parts) > 1 and parts[1] not in expected:
                    break
                if member.isfile() and len(parts) == 3:
                    total += member.size
                    if total > 400_000_000:
                        raise ValueError("RGB extraction exceeds 400MB limit")
                    target = safe_archive_path(raw, member.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = archive.extractfile(member)
                    if src is None:
                        raise ValueError("RGB archive member missing contents")
                    with src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    count[parts[1]] += 1
                    files.append({"path": member.name, "sha256": file_digest(target)})
    finally:
        process.terminate()
        process.wait()
    if any(value != 83 for value in count.values()):
        raise RuntimeError(f"Incomplete RGB download: {count}")
    (raw / "rgb_subset_manifest.json").write_text(
        json.dumps(
            {
                "source_url": url,
                "release": "1.0",
                "sequences": sequences,
                "whole_archive_checksum_verified": False,
                "integrity_note": (
                    "TLS-verified partial archive; individual SHA256, not publisher digest"
                ),
                "files": files,
            },
            indent=2,
        )
    )


def prepare_butom(
    root: Path = Path("data"), sequences: tuple[str, ...] = ("235", "205")
) -> list[Path]:
    """Create a small real-RGB, human-corrected MOT evaluation subset."""
    raw = root / "raw" / "butom-st21"
    annotation_archive = download(
        BUTOM_URL + "54921",
        raw / "eval_annotations.tar.gz",
        "md5:" + BUTOM_ANNOTATION_MD5,
        51_000_000,
    )
    stream_butom_rgb(raw, sequences)
    annotations: dict[str, dict[str, list[list[float]]]] = {seq: {} for seq in sequences}
    with tarfile.open(annotation_archive, "r|gz") as archive:
        completed: set[str] = set()
        previous: str | None = None
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2:
                continue
            seq = parts[1]
            if previous is not None and seq != previous:
                completed.add(previous)
                if set(sequences) <= completed:
                    break
            previous = seq
            if seq not in annotations or not member.isfile():
                continue
            if member.size > 200_000_000:
                raise ValueError("Annotation pickle exceeds memory bound")
            annotation_source = archive.extractfile(member)
            if annotation_source is None:
                raise ValueError("Annotation archive member missing contents")
            with annotation_source:
                data = ArrayUnpickler(io.BytesIO(annotation_source.read())).load()
            boxes = []
            for identity, instance in data.items():
                x, y, width, height = map(float, instance["bbox"])
                if width <= 0 or height <= 0:
                    continue
                # Original boxes are zero-based COCO xywh; MOT coordinates are one-based.
                boxes.append([int(identity), x + 1, y + 1, width, height, 1, 1, 1, -1])
            annotations[seq][Path(member.name).stem] = boxes
    outputs = []
    for seq, frames in annotations.items():
        if len(frames) != 63:
            raise ValueError(f"Expected 63 human-corrected frames for {seq}, found {len(frames)}")
        destination = root / "processed" / f"butom-st21-{seq}"
        (destination / "img1").mkdir(parents=True, exist_ok=True)
        rows = []
        manifest_frames = []
        for frame, timestamp in enumerate(sorted(frames), 1):
            source = raw / "rgb" / seq / (timestamp + ".png")
            image = Image.open(source).convert("RGB")
            width, height = image.size
            output = destination / "img1" / f"{frame:06d}.jpg"
            image.save(output, quality=95)
            rows.extend([[frame, *box] for box in frames[timestamp]])
            manifest_frames.append(
                {
                    "frame": frame,
                    "source_timestamp_ns": timestamp,
                    "source_sha256": file_digest(source),
                    "converted_sha256": file_digest(output),
                }
            )
        write_mot(destination / "gt" / "gt.txt", rows)
        (destination / "seqinfo.ini").write_text(
            f"[Sequence]\nname=butom-st21-{seq}\nimDir=img1\nframeRate=15\n"
            f"seqLength=63\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n"
        )
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset": "BUTom-ST21",
                    "release": "1.0",
                    "sequence": seq,
                    "source": "https://doi.org/10.60507/FK2/TTPCNV",
                    "license": "CC-BY-4.0",
                    "official_split": "eval",
                    "annotation_kind": "human_corrected",
                    "evaluation_frames": list(range(1, 64)),
                    "omitted_frames": "20 unannotated warm-up frames; never treated as empty GT",
                    "fps_note": "15Hz inferred from timestamp spacing; timestamps preserved below",
                    "annotation_archive_md5_verified": BUTOM_ANNOTATION_MD5,
                    "rgb_whole_archive_checksum_verified": False,
                    "rgb_integrity_note": "Prefix extracted over TLS; per-frame SHA256 recorded",
                    "frames": manifest_frames,
                },
                indent=2,
            )
        )
        outputs.append(destination)
    return outputs


def prepare_blueberrydcm(root: Path = Path("data")) -> Path:
    raw = root / "raw" / "blueberrydcm"
    archive = download(DCM_URL, raw / "BlueberryDCM.7z", "md5:" + DCM_MD5, 108_000_000)
    destination = root / "processed" / "blueberrydcm"
    if not list(destination.rglob("*.jpg")):
        extract_archive(archive, destination, 500_000_000)
    images = sorted(destination.rglob("*.jpg"))
    if len(images) != 140:
        raise ValueError(f"Expected140 BlueberryDCM images, found {len(images)}")
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "BlueberryDCM",
                "release": "v1",
                "record": "14002517",
                "source": "https://zenodo.org/records/14002517",
                "license": "CC-BY-NC-4.0",
                "archive_md5_verified": DCM_MD5,
                "persistent_temporal_ids": False,
                "annotation_kind": "human_detection",
                "image_count": len(images),
                "classes": ["Unblue_visible", "Unblue_occluded", "Blue_visible", "Blue_occluded"],
                "images": [
                    {"path": str(p.relative_to(destination)), "sha256": file_digest(p)}
                    for p in images
                ],
            },
            indent=2,
        )
    )
    return destination
