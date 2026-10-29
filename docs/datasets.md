# Dataset audit and reproducible ingestion

Verified on 2026-11-07 against publisher records and machine-readable metadata.
Data and generated labels remain outside Git. Licenses below apply to data,
independently of this repository's software license.

| Dataset | Access and license | Annotation meaning | Use in BerryTrack |
|---|---|---|---|
| [AgriVision](https://www.nature.com/articles/s41597-025-06263-3), [Figshare v1](https://doi.org/10.6084/m9.figshare.29222462.v1) | CC-BY-4.0 per source metadata. DB-2 archive78GB; bounded subsets obtained by HTTP ranges. Article license differs. | DB-1:1,195 manual keyframes. DB-2:141,453 frames/520 recordings with inferred masks, not established persistent IDs. | Real contiguous blueberry frames reconstructed; tiny visually inspected reference separately labeled. Keep recordings together in splits. [Workflow](agrivision.md). |
| [BlueberryDCM v1](https://zenodo.org/records/14002517) | Public, **CC-BY-NC-4.0** (verified API). `BlueberryDCM.7z`, 107,267,412 bytes. | 140 static images; VIA JSON, VOC XML and YOLO TXT boxes. Four classes distinguish ripe/unripe and visible/occluded. No temporal identities. | Fully downloaded, publisher MD5 checked, extracted and loader checked. Detection and occlusion research; never turn image box indices into tracking GT. |
| [BUTom-ST21 v1](https://doi.org/10.60507/FK2/TTPCNV), [author tools](https://github.com/Agricultural-Robotics-Bonn/BUTom21-ST21) | Public, CC-BY-4.0. RGB split across seven large archive parts; annotations separate. | Real mobile-robot RGB; PAg-NeRF supplies weak train/validation labels. Official evaluation annotations are hand corrected and have consistent IDs. 83 RGB frames per sequence, **63 annotated**; omit 10 warm-up frames at either end. | Sequences 235 and 205 ingested with human-corrected annotations. These are official evaluation sequences; any use of 205 for exploratory training changes the protocol and must be disclosed. |
| [NeSTCount v1](https://zenodo.org/records/21636487) | CC-BY-4.0; 10,325,344,888-byte archive. Record explicitly says original images require contacting authors. | Global-ID MOT GT and metadata for eight sequences, four species. | No images fabricated or requested automatically. Not downloaded; image-dependent experiments pending access. |

## Download and inspect

```powershell
uv run scripts/prepare_public_data.py butom
uv run scripts/prepare_public_data.py blueberrydcm
uv run python -c "from berrytrack.data.mot import MOTSequence; s=MOTSequence.load('data/processed/butom-st21-235'); print(s, s.gt.shape)"
```

The BUTom command streams only the first two RGB sequences (~220MB) from part
`aa`, stops at the next sequence, and converts their **annotated timestamps only**
to contiguous MOT frame indices. The 50.5MB annotation archive is fully downloaded
and publisher-MD5 verified. Its large binary masks are decoded one frame at a time
and discarded after extracting boxes and original IDs; no 41GB mask extraction.

The partial RGB archive's publisher MD5 **cannot** be validated without the full
4GB part. Its TLS origin and SHA256 for every extracted frame are recorded, with
`rgb_whole_archive_checksum_verified: false`; these hashes detect subsequent
corruption but do not independently authenticate the source. Re-download the full
part and check publisher MD5 if that stronger guarantee is needed.

Output `data/processed/butom-st21-235/` contains `img1/000001.jpg` through
`000063.jpg`, `gt/gt.txt`, `seqinfo.ini`, and `manifest.json`. The manifest maps each
new frame to the original nanosecond timestamp and checksums. Images are JPEG q95
conversions of source RGB; 15Hz is inferred from timestamp spacing. Coordinates
on disk are MOT **one-based xywh**; loaders return zero-based xyxy tensors.

Verified subset statistics (not tracking scores): sequence 235 has 1,880 boxes and
84 annotated identities; sequence 205 has 1,658 boxes and 53 identities. Annotated
coverage is all 63 exported frames; omitted warm-up frames are unknown, not empty
GT. A missing GT file raises an error.

## Integrity and versioning

`configs/datasets/public.json` pins release IDs, download URLs, publisher sizes and
checksums. Every preparation run writes a local manifest. Download limits, archive
path traversal/symlink rejection, unpacked-size limits, checksum verification and a
numpy-only unpickler constrain ingestion. Legacy publisher pickle annotations are
accepted only from the checksum-verified archive; normal pipeline files use text
or JSON. The RGB streaming subset relies on the documented v1 archive ordering
and fails when expected frame counts differ.

`validate_splits` rejects duplicate sequence membership. `grouped_split` hashes an
explicit capture-group key, so all frames from one recording stay together. For
BlueberryDCM, capture-session/plant grouping is not supplied reliably by the record;
image-level random splits cannot establish independence and must be labeled
exploratory. Do not tune on sequence 235 or call subset results an official full
benchmark. Cache detections once and share the exact cache among all trackers.

### BlueberryDCM source-format discrepancy

Inspection of the downloaded release found that its TXT files contain only
class0 (10,981 boxes) and class3 (6,974 boxes), although the VIA JSON distinguishes
all four categories. VIA totals are 4,026 Unblue_visible, 6,861 Unblue_occluded,
1,682 Blue_visible and 5,285 Blue_occluded (17,854 boxes). TXT totals17,955.
Thus both category mapping and total annotation counts differ by source format.
The project preserves the original files and documents the discrepancy; it does
not silently treat TXT as a faithful four-class export.

`scripts/prepare_detector.py` reconstructs four classes and normalized rectangles
from VIA attributes. Its fixed development subset uses32 train,21 validation and
21 reserved test images, with the remaining66 train candidates omitted to keep
the initial job small. The test images were not evaluated. Capture groups are
unknown, so this is an image-level pipeline experiment with possible plant leakage.
Ultralytics additionally rejects duplicated labels during its own validation.
The raw `YoloDetectionDataset` loader reads publisher TXT as supplied; it does not
claim to repair class semantics. Use the VIA-derived split for corrected training.

Cached source reuse verifies previously recorded SHA256 values for RGB and all
BlueberryDCM image/annotation files. A changed file fails closed. An older
image-only manifest is upgraded by re-extracting the publisher-checksummed archive.
Preparing a different detection split into a populated output is rejected, so
stale training files cannot silently defeat the declared split size.

## Human identity workflow for blueberry recordings

1. Select a short licensed recording; reserve entire recordings as development,
   validation, or test **before** drawing labels.
2. Create an exchange task with
   `berrytrack.data.annotation.create_annotation_task(video, output, frame_count, license)`.
   It contains blank objects and `reviewed: false` for every frame.
3. Annotate persistent tracks in CVAT or an equivalent visual tool; review entry,
   occlusion, reappearance and camera reversals. Map its export to `objects` with
   `track_id` and zero-based `bbox_xywh`. Use one ID per physical fruit, not per box.
4. Have a human review every evaluation frame, including genuinely empty ones;
   set `label_kind: manual_verified`, `reviewer`, and each `reviewed: true` only
   after inspection. Preserve uncertain intervals separately rather than guessing.
5. `export_verified_task(task, mot_file)` rejects pending/pseudo or incomplete
   tasks, repeated per-frame IDs, missing reviewers and discontinuous coverage.
   It writes MOT and provenance sidecar. Archive the original human task.

Tracker suggestions may accelerate annotation but remain **pseudo-labels** until
human review. A later extension adds a tiny visually inspected blueberry
reference, explicitly without human verification; see [its scope](agrivision.md).
A genuinely independent human-verified blueberry tracking benchmark remains pending.
