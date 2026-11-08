"""Bounded HTTP-range ingestion of contiguous AgriVision DB-2 video frames.

The source provides inferred segmentation masks, not verified temporal identities.
Only selected ZIP members and its central directory are downloaded, never 78GB.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from PIL import Image

ARTICLE_API = "https://api.figshare.com/v2/articles/29222462/versions/1"
ARCHIVE_URL = "https://ndownloader.figshare.com/files/55546409"
ARCHIVE_SIZE = 78_036_031_533
ARCHIVE_MD5 = "172c20f85985d52c2e769e78a1dcc4bf"

# Frozen, explicitly assistant-reviewed annotations for eight inspected source frames.
# These hashes prevent applying the visual reference to another recording or frame range.
REFERENCE_FRAME_SHA256 = (
    "a6ab26024f5e74c44ec22601a61335522fa4d170d3ef4464e63465c831021a2e",
    "de77f0ad23c08b17faf34ccc5ce7a2d50b53573d23d89057c2544efe47f083d7",
    "a9e8f07aef3d015e6bd1e019a1363568e10cef01f0d574df12f13102314e3cb8",
    "2b3732f604a946221b04b409a3af929347372b478fb6511df007ed2390ab58ed",
    "bf8cbc0856bb6bc5025fefe350c0f5b2139f5ed312c6045e685c23927dd6758c",
    "3380943db33565bfc02070a465a11d5983706292819921dd8cbb3508a1e1252d",
    "99947c1fb318799d45cd8151635ddd702a55dc55ef4d1149b3140b3459090c4d",
    "2409a2213e8fa6d928d050c8dd0ed683ed5ccea6b8bdf58f1c63f51c7ffde80a",
)
REFERENCE_ROI = (640, 710, 900, 970)
REFERENCE_BOXES_XYXY = (
    (673, 716, 710, 755),
    (648, 740, 695, 795),
    (663, 790, 710, 834),
    (716, 868, 769, 912),
    (742, 883, 801, 938),
    (793, 882, 840, 930),
    (826, 872, 874, 923),
)


def mask_component_boxes(mask, minimum_area: int = 80, maximum_area: int = 30_000):
    """Turn publisher binary masks into class-agnostic box proposals, never IDs.

    Touching segmented fruit remain one connected proposal. These inferred masks
    cannot establish instance counts or independent tracking ground truth.
    """
    import cv2
    import numpy as np

    mask = np.asarray(mask)
    if mask.ndim != 2 or minimum_area < 1 or maximum_area < minimum_area:
        raise ValueError("Expected a 2D mask and positive, ordered area limits")
    _, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    boxes = []
    for x, y, width, height, area in stats[1:]:
        if minimum_area <= area <= maximum_area:
            boxes.append([x, y, x + width, y + height])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


class RangeReader(io.RawIOBase):
    """Seekable ZIP source which refuses servers that ignore HTTP Range.

    A request is checked before reading its body; absent/incorrect 206 responses
    cannot accidentally become a full-archive download. The total body-byte cap
    includes ZIP metadata and all selected compressed member bytes.
    """

    def __init__(
        self,
        url: str,
        size: int,
        max_bytes: int = 120_000_000,
        fetch: Callable[[int, int], bytes] | None = None,
    ) -> None:
        super().__init__()
        self.url, self.size, self.max_bytes = url, size, max_bytes
        self.position = 0
        self.bytes_fetched = 0
        self.requests = 0
        self.fetch = fetch or self._fetch
        self._cache_start = -1
        self._cache = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError("Invalid seek mode")
        if target < 0:
            raise ValueError("Negative seek position")
        self.position = target
        return target

    def read(self, size: int = -1) -> bytes:
        remaining = max(0, self.size - self.position)
        length = remaining if size < 0 else min(size, remaining)
        if not length:
            return b""
        if self._cache_start <= self.position and self.position + length <= self._cache_start + len(
            self._cache
        ):
            begin = self.position - self._cache_start
            self.position += length
            return self._cache[begin : begin + length]
        if self.bytes_fetched + length > self.max_bytes:
            raise ValueError("HTTP-range ingestion exceeds its total byte budget")
        # Coalesce tiny ZIP-header reads into one bounded range request.
        fetched_length = min(max(length, 65536), remaining, self.max_bytes - self.bytes_fetched)
        data = self.fetch(self.position, self.position + fetched_length - 1)
        if len(data) != fetched_length:
            raise ValueError("Truncated HTTP-range response")
        self._cache_start, self._cache = self.position, data
        self.position += length
        self.bytes_fetched += fetched_length
        self.requests += 1
        return data[:length]

    def _fetch(self, start: int, end: int) -> bytes:
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=45) as response:
            expected_range = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError("Source did not honor the exact HTTP byte range")
            length = end - start + 1
            if int(response.headers.get("Content-Length", -1)) != length:
                raise ValueError("Source returned an unexpected range length")
            return response.read(length)


def select_frames(names: list[str], camera: int, video: int, start: int, count: int) -> list[str]:
    if camera not in (1, 2, 3, 4) or min(video, start, count) < 1 or count > 40:
        raise ValueError("Choose camera1-4, positive video/frame, and at most40 frames")
    prefix = f"DB-2/Camera {camera}/Videos_Frames/Camera {camera} Video ({video})_"
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.jpg$")
    found = {int(m[1]): name for name in names if (m := pattern.fullmatch(name))}
    selected = list(range(start, start + count))
    missing = [frame for frame in selected if frame not in found]
    if missing:
        raise ValueError(f"Source sequence is not contiguous over requested frames: {missing}")
    return [found[frame] for frame in selected]


def reconstruct_video(images: Path, output: Path, playback_fps: float) -> None:
    """Playback speed is explicit; the source archive contains no verified FPS."""
    import imageio_ffmpeg

    if not 0 < playback_fps <= 120:
        raise ValueError("Playback FPS must be in (0,120]")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(playback_fps),
            "-i",
            str(images / "%06d.jpg"),
            "-an",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )


def prepare_agrivision(
    output: Path = Path("data/processed/agrivision-camera1-video1"),
    camera: int = 1,
    video: int = 1,
    start: int = 1,
    count: int = 24,
    playback_fps: float = 10,
    masks: bool = True,
    max_bytes: int = 120_000_000,
) -> Path:
    """Save original JPEG frames, optional pseudo masks, H264 and provenance.

    Existing output is rejected to avoid mixing source sequences. Exact source
    frame numbers are selected numerically, not lexical ZIP filename order.
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new/empty output directory to preserve dataset provenance")
    # Check cheap arguments before opening the archive.
    if camera not in (1, 2, 3, 4) or min(video, start, count) < 1 or count > 40:
        raise ValueError("Choose camera1-4, positive video/frame, and at most40 frames")
    if not 0 < playback_fps <= 120:
        raise ValueError("Playback FPS must be in (0,120]")
    with urllib.request.urlopen(ARTICLE_API, timeout=30) as response:
        metadata = json.load(response)
    archive_metadata = next(f for f in metadata["files"] if f["id"] == 55546409)
    if archive_metadata["size"] != ARCHIVE_SIZE or archive_metadata["computed_md5"] != ARCHIVE_MD5:
        raise ValueError("Publisher archive changed from pinned AgriVision v1")
    reader = RangeReader(ARCHIVE_URL, ARCHIVE_SIZE, max_bytes)
    manifest_frames = []
    with zipfile.ZipFile(reader) as archive:
        names = archive.namelist()
        selected = select_frames(names, camera, video, start, count)
        output.mkdir(parents=True, exist_ok=True)
        (output / "img1").mkdir()
        for number, source in enumerate(selected, 1):
            info = archive.getinfo(source)
            if info.file_size > 20_000_000:
                raise ValueError("Source image exceeds extraction bound")
            data = archive.read(info)  # zipfile verifies this member's CRC32.
            target = output / "img1" / f"{number:06d}.jpg"
            target.write_bytes(data)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                image.verify()
            record: dict = {
                "frame": number,
                "source_frame": start + number - 1,
                "source_member": source,
                "sha256": hashlib.sha256(data).hexdigest(),
                "zip_crc32_verified": f"{info.CRC:08x}",
                "width": width,
                "height": height,
            }
            if masks:
                mask = source.replace("Videos_Frames/", "Videos_Frames_Masks/")[:-4] + ".png"
                mask_info = archive.getinfo(mask)
                if mask_info.file_size > 20_000_000:
                    raise ValueError("Source mask exceeds extraction bound")
                mask_data = archive.read(mask_info)
                (output / "pseudo_masks").mkdir(exist_ok=True)
                (output / "pseudo_masks" / f"{number:06d}.png").write_bytes(mask_data)
                record["pseudo_mask"] = {
                    "source_member": mask,
                    "sha256": hashlib.sha256(mask_data).hexdigest(),
                    "zip_crc32_verified": f"{mask_info.CRC:08x}",
                    "label_kind": "publisher_inferred_segmentation_not_tracking_gt",
                }
            manifest_frames.append(record)
    reconstruct_video(output / "img1", output / "sequence.mp4", playback_fps)
    (output / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "AgriVision DB-2",
                "article_id": 29222462,
                "version": 1,
                "doi": metadata["doi"],
                "metadata_api": ARTICLE_API,
                "license": metadata["license"],
                "authors": [a["full_name"] for a in metadata["authors"]],
                "archive_url": ARCHIVE_URL,
                "archive_size": ARCHIVE_SIZE,
                "publisher_archive_md5": ARCHIVE_MD5,
                "whole_archive_md5_verified": False,
                "integrity": "TLS HTTP206 ranges; per-member ZIP CRC32; extracted SHA256",
                "camera": camera,
                "video": video,
                "first_source_frame": start,
                "frame_count": count,
                "contiguous_source_indices": True,
                "source_fps": None,
                "playback_fps": playback_fps,
                "timing_note": "Playback FPS chosen; capture FPS unknown",
                "persistent_temporal_identities": False,
                "tracking_ground_truth": "unavailable",
                "http_bytes_downloaded": reader.bytes_fetched,
                "http_range_requests": reader.requests,
                "frames": manifest_frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def prepare_assistant_reference(source: Path, output: Path) -> Path:
    """Reproduce the visually authored reference, never claim human-verified GT.

    All seven sufficiently visible fruit instances in the fixed ROI were inspected
    in source frames1--8. Their small motion stays within estimated +/-5px boundary
    uncertainty, so the reviewed coordinates are retained across these frames.
    No tracker/detector or inferred mask supplied boxes or temporal identities.
    This is a near-stationary pipeline diagnostic, not an independent benchmark.
    """
    from berrytrack.data.mot import write_mot

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Reference output must be new or empty")
    for frame, expected in enumerate(REFERENCE_FRAME_SHA256, 1):
        path = source / "img1" / f"{frame:06d}.jpg"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Source differs from the exact visually inspected AgriVision frames")
    (output / "img1").mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    for frame, expected in enumerate(REFERENCE_FRAME_SHA256, 1):
        with Image.open(source / "img1" / f"{frame:06d}.jpg") as image:
            image.crop(REFERENCE_ROI).save(output / "img1" / f"{frame:06d}.jpg", quality=95)
        objects = []
        for identity, (x1, y1, x2, y2) in enumerate(REFERENCE_BOXES_XYXY, 1):
            box = [x1 - REFERENCE_ROI[0], y1 - REFERENCE_ROI[1], x2 - x1, y2 - y1]
            visibility = 0.5 if identity == 6 else 0.9  # uncertain visual estimates, not measured
            rows.append([frame, identity, box[0] + 1, box[1] + 1, *box[2:], 1, 1, visibility, -1])
            objects.append(
                {
                    "track_id": identity,
                    "bbox_xywh": box,
                    "label_kind": "assistant_visual_review",
                    "occlusion": "partly_leaf_occluded"
                    if identity == 6
                    else "clear_or_minor_overlap",
                    "localization_uncertainty_pixels": 5,
                }
            )
        records.append(
            {
                "frame": frame,
                "source_frame": frame,
                "source_sha256": expected,
                "visually_reviewed": True,
                "objects": objects,
            }
        )
    write_mot(output / "gt" / "gt.txt", rows)
    (output / "seqinfo.ini").write_text(
        "[Sequence]\nname=agrivision-c1-v1-assistant-roi\nimDir=img1\nframeRate=10\n"
        "seqLength=8\nimWidth=260\nimHeight=260\nimExt=.jpg\n"
    )
    provenance = {
        "dataset": "AgriVision DB-2 tiny assistant-reviewed ROI",
        "version": 1,
        "source_sequence_root": str(source),
        "source_doi": "10.6084/m9.figshare.29222462.v1",
        "label_kind": "assistant_visual_review",
        "annotation_kind": "assistant_visual_review",
        "human_verified": False,
        "independent_benchmark": False,
        "tracking_ground_truth": "assistant_reference_only",
        "evaluation_frames": list(range(1, 9)),
        "roi_xyxy": REFERENCE_ROI,
        "source_fps": None,
        "playback_fps": 10,
        "license": "CC-BY-4.0",
        "annotated_reference_identities": 7,
        "scope": "All seven sufficiently visible fruit instances in this ROI; not full scene",
        "protocol": "Boxes and IDs authored by assistant visual review of eight consecutive "
        "frames; unchanged reviewed boxes retained for near-stationary fruit; no tracker output",
        "limitations": [
            "Not human verified or independent ground truth; estimated +/-5px localization",
            "Near-stationary easy crop, no occlusion transitions/reentry; not H1-H4 evidence",
            "ID6 partly leaf occluded; visibility values are uncertain visual estimates",
            "Capture FPS unknown; seqinfo10FPS is chosen playback only",
        ],
        "frames": records,
    }
    (output / "reference_annotations.json").write_text(json.dumps(provenance, indent=2))
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2))
    reconstruct_video(output / "img1", output / "sequence.mp4", 10)
    return output
