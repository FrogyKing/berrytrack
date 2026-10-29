# Real blueberry video and explicit reference labels

Source: [AgriVision DB-2 v1](https://doi.org/10.6084/m9.figshare.29222462.v1),
[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12748667/).
The dataset's Figshare metadata declares CC-BY-4.0; the article has separate terms.
Authors: Muhammad Owais, Muhammad Shafay, Muhammad Zubair, Shamal Mohammed,
Lakmal Seneviratne and Irfan Hussain. No endorsement is implied.

## Obtained data

Camera 1, video1, source frames 1â€“24 were extracted as original1920x1080 JPEGs,
along with their24 inferred segmentation masks. A54,330,482-byte HTTP-range
transfer (including the39.5 MB ZIP central directory) obtained the subset without
downloading the78,036,031,533-byte DB-2 archive. The source archive is file55546409
and its published MD5 is172c20f85985d52c2e769e78a1dcc4bf.

Numeric source indices are required to be contiguous and from the same recording.
Frames from different cameras/videos and unrelated detection images cannot be
concatenated into a valid tracking video. The reconstructed `sequence.mp4` uses
H.264 and an explicitly selected playback rate10 FPS. Capture FPS is unknown;
the playback rate is not a measured biological/video-time calibration.

Source masks are publisher-inferred segmentation, not persistent identity GT.
The download records the source DOI/license, member names, frame indices,
dimensions, CRC32, SHA256 and actual transferred bytes. It refuses a200 response
before reading a body when206 was requested and enforces a120 MB transfer bound.
Partial extraction cannot validate the whole archive's publisher MD5.

A second24-frame source range (91â€“114) was also downloaded and reconstructed to
inspect later camera motion. It remains a mild-viewpoint example without new
identity GT. Both full-scene clips were processed with identical detector outputs
for BerryTrack and BoT-SORT. The preliminary detector finds very few confirmed
identities despite visible green berries; no full-scene counting accuracy is
claimed. Better in-domain detection is still needed for a useful field pipeline.

## Reproduce

```powershell
uv run scripts/prepare_agrivision.py --output data/processed/agrivision-blueberry --reference-output data/processed/agrivision-blueberry-reference
uv run berrytrack track data/processed/agrivision-blueberry/sequence.mp4 --weights checkpoints/blueberry-via/best.pt --compare botsort --output artifacts/agrivision-comparison --max-frames 24
uv run python experiments/blueberry.py --sequence data/processed/agrivision-blueberry-reference --output artifacts/agrivision-reference-check
```

Data and weights remain ignored by Git. The inference command uses the previously
adapted blueberry detector and reports confirmed tracker IDs, not an independently
known full-scene fruit count. Recompression can affect appearance/camera estimates.

## Exploratory visual reference labels

The fixed ROI is source xyxy(640,710,900,970),260x260 pixels. Seven sufficiently
visible berries were assigned persistent IDs over source frames 1â€“8. All eight
crops and an enlarged labeled contact sheet were visually inspected. The small
motion remained within approximately5 px boundary uncertainty, so the inspected
coordinates are shared across these frames. ID6 is partly leaf occluded;
visibility values in MOT are uncertain visual estimates, not measured fractions.

`prepare_visual_reference` reproduces these 56 box/ID labels only when all eight
original SHA256 values match. The JSON states `exploratory_visual_review`,
`human_verified=false`, and `independent_benchmark=false`. No tracker output or
publisher-inferred mask generated the reference identities. The existing export
route for human-verified labels remains separate and still requires a human reviewer.

The scope is this crop and its sufficiently visible fruit. Invisible/indeterminate
fruit and the rest of the full image are outside its counting claim. This tiny,
near-stationary sequence has no meaningful occlusion/re-entry challenge, so it
validates pipeline execution rather than research accuracy. A larger, independently
human-verified and sequence-separated blueberry MOT test remains necessary.

See [executed comparisons and limitations](rigidity-results.md).
