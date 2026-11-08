# BerryTrack

**Geometry-aware multi-object tracking for dense agricultural scenes, combining
adaptive embeddings, motion compensation, and uncertainty-guided identity association.**

Blueberries are small, crowded and visually similar. A moving camera changes their
apparent positions; occlusion can break tracks and inflate unique-fruit counts.
BerryTrack is an executable research implementation with detector-independent
baselines, validated MOT metrics, a local video application and bounded GPU jobs.

**Research question:** can adaptive appearance, camera compensation and local
geometry improve identity association and counting at lower computational cost?
The initial evidence is mixed. No novelty, superiority or validated blueberry
tracking accuracy is claimed.

[Methods](docs/research.md) · [Dataset audit](docs/datasets.md) ·
[Evaluation](docs/evaluation.md) · [Modal guide](docs/modal.md) ·
[Executed results](results/diagnostic/report.md)

## What runs today

- Custom BerryTrack: box Kalman filtering, robust affine camera compensation,
  sparse neighborhood descriptors, adaptive costs, gated assignment, lazy
  appearance and persistent identity memory.
- Actual upstream ByteTrack, BoT-SORT and DeepSORT using identical cached detections.
- 32/64/128-dimensional compact descriptors; a trainable PyTorch encoder;
  optional ROI-pooled detector features and sparse graph attention.
- Verified BUTom-ST21 RGB/identity and BlueberryDCM detection ingestion,
  manifests, checksums, MOT conversion, loaders and manual annotation exchange.
- TrackEval HOTA/DetA/AssA, IDF1, CLEAR metrics, counting errors, timings and plots.
- Gradio upload, comparison, trajectories, cumulative IDs, switch-frame inspection
  with GT, H.264 export and MOT/JSON downloads.
- Real T4 training, checkpoint resumption and inference on persistent Modal Volumes.

The default CPU benchmark uses an **untrained** compact appearance descriptor.
Learned encoder execution is verified separately. Sparse graph attention has unit
coverage but no trained tracking result. Scientific limitations remain explicit.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/), from the repository root:

```powershell
uv sync --locked --dev
uv run berrytrack version
uv run ruff check .
uv run ruff format --check .
uv run mypy src/berrytrack
uv run pytest
```

CPU tests require neither Modal credentials nor a GPU. Tests never download model
weights or datasets. Dependencies are locked in `uv.lock`; automatic Ultralytics
dependency installation is disabled and `lap` is explicitly locked.

## Reproduce the association diagnostic

```powershell
uv run scripts/prepare_public_data.py butom
uv run python -m berrytrack.evaluation.benchmark --sequence data/processed/butom-st21-235 --oracle --ablations --dimensions 32 64 128 --seeds 42 43 44 --output artifacts/association
```

The downloader streams two small RGB sequences and validates the complete
annotation archive's publisher MD5. The partial RGB archive cannot receive its
whole-file checksum; per-frame hashes and this limitation are recorded. Cached
source files are checked against saved hashes before reuse.

`--oracle` removes identities from human-corrected GT boxes, then caches the
boxes once. This isolates association. It is **not detector performance**, a
blueberry result or an official full BUTom benchmark. All 63 annotated frames of
sequence 235 are used, with 84 original identities. Unannotated warm-up frames are
excluded, never treated as empty GT.

To evaluate actual detector outputs:

```powershell
uv run berrytrack cache-detections data/processed/butom-st21-235 --weights checkpoints/blueberry-via/best.pt --output data/processed/butom-st21-235/det/det.txt
uv run python -m berrytrack.evaluation.benchmark --sequence data/processed/butom-st21-235 --seeds 42 --output artifacts/detector-transfer
```

This example transfers a blueberry detector to tomatoes: its poor result is a
domain-shift diagnostic. Detector AP and association performance are separate
measurements. Broader experiments need a detector trained on independent,
compatible recordings.

## Video application

![Reference-box video comparison](docs/assets/tracking-demo.gif)

This rendered real-sequence demo uses reference boxes and includes remaining
identity errors. [Source attribution and processing](docs/assets/README.md).

```powershell
uv run berrytrack demo
```

Open [the local app](http://127.0.0.1:7860). Upload an agricultural video, choose a
tracker and comparison, and supply adapted detector weights. It runs locally on
CPU and never starts paid cloud instances. Generic COCO weights are unreliable for
blueberries. Ripeness/visibility labels collapse to one physical-fruit tracking
class so classifier changes cannot force new identities.

[Application controls, exports and verification](docs/demo.md).

Generate a working reference demo from the licensed real sequence:

```powershell
uv run scripts/make_demo_video.py
uv run berrytrack track artifacts/demo/input.mp4 --detections artifacts/association/cache/butom-st21-235.txt --gt data/processed/butom-st21-235/gt/gt.txt --compare botsort --output artifacts/demo/comparison --max-frames 63
```

In the UI, upload that video and the same cache/GT under the advanced section.
Reference detections bypass the detector. Video compression changes pixels;
exact benchmark reproduction uses the original converted JPEG frames. Generated
videos, datasets and weights stay outside Git.

Normal inference and optional appearance alternatives:

```powershell
uv run berrytrack track your-video.mp4 --weights checkpoints/blueberry-via/best.pt --compare botsort --output artifacts/your-video
uv run berrytrack track your-video.mp4 --weights checkpoints/blueberry-via/best.pt --reuse-backbone --output artifacts/backbone
uv run berrytrack track your-video.mp4 --weights checkpoints/blueberry-via/best.pt --encoder checkpoints/encoder-dev/encoder.pt --output artifacts/encoder
```

Backbone feature reuse has been executed and checked for finite 64D features.
Its tracking accuracy still needs controlled evaluation. Counts represent
confirmed tracker identities, not guaranteed physical-fruit totals.

## Architecture and mathematics

```mermaid
flowchart LR
    V[Video / sequence] --> D[Detector or immutable cache]
    V --> C[Validated camera motion]
    D --> E[Lazy crops / cached backbone features]
    D --> G[Sparse local neighborhood]
    M[Identity memory + Kalman covariance] --> A[Adaptive gated costs]
    C --> A
    E --> A
    G --> A
    A --> H[Hungarian assignment + unmatched states]
    H --> M
    H --> O[Tracks · count · visualization]
    O --> Q[TrackEval + reproducibility artifacts]
```

$$C(i,j)=\alpha_{ij}D_{\mathrm{appearance}}+
\beta_{ij}D_{\mathrm{motion}}+
\gamma_{ij}D_{\mathrm{geometry}}+
\delta_{ij}D_{\mathrm{context}}.$$

Terms lie in $[0,1]$. Coefficients form a simplex and adapt to available appearance,
track uncertainty, camera support and neighborhood occupancy. Missing terms are
removed before normalization. High-confidence matching precedes weak-detection
recovery; weak detections cannot initialize tracks.

The camera model rejects unsupported transforms instead of assuming a global
homography fits a non-planar canopy. The neighborhood graph is sparse; current
pair costs and Hungarian assignment are dense. See the
[derivations, assumptions and complexity](docs/research.md).

## Executed results

**Oracle-box diagnostic: BUTom235, one tomato clip, three descriptor seeds
(42/43/44).** Metrics are percentages. Warm timing excludes the first five online
frames; all frames remain in accuracy metrics. Cold, full-online and per-frame
timings are retained. Seed deviations are not uncertainty across videos.

| Tracker | HOTA | IDF1 | ID switches | Count MAE | Warm ms/frame |
|---|---:|---:|---:|---:|---:|
| ByteTrack | 59.12 ±0.00 | 70.49 ±0.00 | 123.0 | 81.0 | 1.69 |
| BoT-SORT, default ReID off | 82.44 ±0.00 | 89.48 ±0.00 | 30.0 | 18.0 | 7.38 |
| DeepSORT, compact appearance | 75.65 ±0.36 | 76.55 ±0.52 | 108.3 | 61.7 | 17.96 |
| BerryTrack | 83.65 ±0.44 | 91.35 ±0.44 | 31.7 | 4.0 | 23.99 |

BerryTrack is slower than BoT-SORT here and has slightly more switches. Its HOTA
and count on one ideal-box clip do not establish general superiority.

| Cumulative ablation | HOTA mean | Count MAE mean | Crops mean |
|---|---:|---:|---:|
| Appearance only | 36.70 | 45.0 | 1880.0 |
| Motion only | 83.28 | 10.0 | 0.0 |
| Appearance + motion | 84.01 | 7.7 | 1880.0 |
| + camera compensation | 82.75 | 2.3 | 1880.0 |
| + local context | 82.74 | 2.0 | 1880.0 |
| + adaptive weights | 83.37 | 2.0 | 1880.0 |
| + uncertainty trigger | 83.65 | 4.0 | 1059.7 |

Context did not improve mean HOTA in this ladder. Camera compensation reduced
switches but hurt HOTA relative to appearance+motion. Lazy appearance reduced
crops by 43.6% and paired warm latency from 27.63 to 23.39ms, while count MAE
worsened from 2 to 4. H1–H4 remain unproven.
[Full analysis and dimensions](docs/evaluation.md).

![Executed comparison](results/diagnostic/comparison.png)

![Measured identity-switch failures](docs/assets/identity-switches.png)

Source-image derivatives are CC-BY-4.0 with [attribution](docs/assets/README.md).
They show measured failures, not guaranteed performance.

**Actual detections, cross-crop transfer:** on the same tomato clip, the
blueberry detector gives BerryTrack HOTA 26.69 versus BoT-SORT 27.13, ByteTrack
20.48 and compact DeepSORT 33.29. This negative transfer result exposes the
detector bottleneck. [Full outputs](results/detector-transfer/report.md).

**Corrected BlueberryDCM detection validation:** after 20 small-data epochs on
32 training images, a separate 21-image development validation gives mAP50
**28.67%** and mAP50–95 **16.82%**. The image split may share plants and is not
independent field or tracking evidence. The 21 reserved test images were not
evaluated. Source TXT/VIA inconsistencies are [audited](results/detection/audit.json).
An initial collapsed-label pilot is preserved for provenance and excluded from
four-class conclusions.

## Data and licensing

| Dataset | License / access | Temporal identities |
|---|---|---|
| [AgriVision blueberries](https://www.nature.com/articles/s41597-025-06263-3) | Figshare displays CC BY; 88GB download deferred | Masks do not establish persistent IDs |
| [BlueberryDCM](https://zenodo.org/records/14002517) | CC-BY-NC-4.0; downloaded and checked | Static images only |
| [BUTom-ST21](https://doi.org/10.60507/FK2/TTPCNV) | CC-BY-4.0; two real RGB/hand-corrected subsets | Consistent original IDs preserved |
| [NeSTCount](https://zenodo.org/records/21636487) | CC-BY-4.0; original images on request | ID annotations available; images pending |

The [audit](docs/datasets.md) records URLs, releases, checksums and label semantics.
Automatically generated masks and pseudo labels are not tracking GT. Both
downloaded BUTom sequences are official evaluation sequences; development
training on 205 is disclosed and does not reproduce the official protocol.

To prepare a genuinely verified blueberry identity set:

```powershell
uv run berrytrack annotation-task your-video.mp4 task.json --frames 100 --license-name CC-BY-4.0
# A human annotates persistent identities and reviews every frame.
uv run berrytrack export-annotations task.json gt.txt
```

The exporter rejects pending/pseudo labels, missing reviewers and incomplete
coverage. No verified blueberry identity set was fabricated.

## GPU training and budget

```powershell
$env:PYTHONUTF8 = "1"
uv run scripts/prepare_public_data.py blueberrydcm
uv run scripts/prepare_detector.py
# Upload prepared data first; full commands are in docs/modal.md.
uv run scripts/modal_job.py train-encoder --seconds 180 --options-file configs/encoder-smoke.json
uv run scripts/modal_job.py train-detector --seconds 180 --options-file configs/detector-via.json
```

Real T4 runs include 30 encoder steps, resumption to 35, 498 inference crops,
and detector training/validation. The 64D encoder has 27,744 parameters. Four named
Volumes persist data, checkpoints, features and artifacts. No GPU endpoint is
deployed. CPU preprocessing is separate from GPU allocation.

The initial hard budget is **US$30**. Conservative cumulative reservations are
**US$2.30131**, including failed build/startup attempts. Successful runtime-only
estimates sum to approximately **US$0.07902**, excluding builds/startup. These are
different accounting scopes, not invoices. Provider metering is delayed:
[run records](results/cloud/runs.json), [billing snapshot](results/cloud/provider-billing.json),
[budget controls](docs/modal.md). The launcher is not a provider-wide billing cap;
executed jobs remain far below the allowance.

## Reproducibility and remaining research

The package uses `src/`, validated YAML configuration, deterministic seeds where
practical, Ruff, mypy, CPU unit/integration tests and Windows/Linux GitHub Actions.
Generated test fixtures validate engineering contracts only; no synthetic tracker
sequence is reported as a research benchmark. Results record source/cache/GT
hashes, environment versions and complete settings.

Pending research includes human-verified blueberry video IDs, independent plant/
recording splits, broader sequence evaluation, calibrated uncertainty, trained
graph context and controlled learned/backbone ReID comparisons. Long occlusions
and re-entry can still overcount. The current detector is preliminary and
occluded-fruit performance is weak. Unexecuted work has no hypothetical scores.

## Inspirations and software

[ByteTrack](https://github.com/FoundationVision/ByteTrack),
[BoT-SORT](https://github.com/NirAharon/BoT-SORT),
[DeepSORT](https://github.com/nwojke/deep_sort),
[TrackEval / HOTA](https://github.com/JonathonLuiten/TrackEval),
[PAg-NeRF / BUTom-ST21](https://github.com/Agricultural-Robotics-Bonn/BUTom21-ST21),
[Ultralytics](https://github.com/ultralytics/ultralytics) and the
[NeSTCount record](https://zenodo.org/records/21636487) inform this work. Existing
neighborhood-structure research matters when assessing novelty; this project
does not claim to originate that idea.

Software is [AGPL-3.0-or-later](LICENSE), consistent with the integrated
Ultralytics dependency. Data, images and models retain independent terms;
BlueberryDCM is noncommercial. [Third-party notices](docs/licenses.md).
