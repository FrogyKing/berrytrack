"""Prepare 160 consecutive blueberry frames and inferred masks for the long demo."""

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from berrytrack.data.agrivision import prepare_agrivision, reconstruct_video


def verify(sequence: Path) -> dict:
    manifest = json.loads((sequence / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["camera"], manifest["video"], manifest["frame_count"]) != (1, 104, 160):
        raise ValueError("Expected camera1/video104, source frames1-160")
    records = manifest["frames"]
    if [r["source_frame"] for r in records] != list(range(1, 161)):
        raise ValueError("Long demo source frames must be consecutive and ordered")
    for number, record in enumerate(records, 1):
        for relative, digest in (
            (f"img1/{number:06d}.jpg", record["sha256"]),
            (f"pseudo_masks/{number:06d}.png", record["pseudo_mask"]["sha256"]),
        ):
            if hashlib.sha256((sequence / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source hash mismatch: {relative}")
    return manifest


def prepare(output: Path) -> None:
    if (output / "manifest.json").exists():
        verify(output)
        print(f"Verified cached real sequence: {output}")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory or a complete verified sequence")
    chunks = output.parent / ".agrivision-demo-chunks"

    def download(start: int) -> Path:
        destination = chunks / f"camera1-video104-{start:06d}"
        if not (destination / "manifest.json").exists():
            prepare_agrivision(
                destination, camera=1, video=104, start=start, count=40, playback_fps=10, masks=True
            )
        return destination

    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(download, [1, 41, 81, 121]))
    output.mkdir(parents=True)
    (output / "img1").mkdir()
    (output / "pseudo_masks").mkdir()
    records, manifests = [], []
    for part in parts:
        manifest = json.loads((part / "manifest.json").read_text(encoding="utf-8"))
        manifests.append(manifest)
        if (manifest["camera"], manifest["video"]) != (1, 104):
            raise ValueError("Cannot combine different recordings")
        for record in manifest["frames"]:
            local = record["frame"]
            number = len(records) + 1
            for folder, extension in (("img1", "jpg"), ("pseudo_masks", "png")):
                shutil.copy2(
                    part / folder / f"{local:06d}.{extension}",
                    output / folder / f"{number:06d}.{extension}",
                )
            records.append({**record, "frame": number})
    combined = {
        **manifests[0],
        "frame_count": 160,
        "playback_fps": 10,
        "frames": records,
        "http_bytes_downloaded": sum(m["http_bytes_downloaded"] for m in manifests),
        "http_range_requests": sum(m["http_range_requests"] for m in manifests),
        "preparation": "Four contiguous 40-frame source ranges, same camera and recording",
    }
    (output / "manifest.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    shutil.copy2(parts[0] / "source_metadata.json", output / "source_metadata.json")
    width, height = Image.open(output / "img1/000001.jpg").size
    (output / "seqinfo.ini").write_text(
        "[Sequence]\nname=agrivision-c1-v104-long\nimDir=img1\nframeRate=10\n"
        f"seqLength=160\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n"
    )
    verify(output)
    reconstruct_video(output / "img1", output / "sequence.mp4", 10)
    print(f"Prepared real consecutive blueberry sequence: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/agrivision-camera1-video104-long")
    )
    prepare(parser.parse_args().output)
