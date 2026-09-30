# Bounded GPU experiments with Modal

The installed Modal skill and matching SDK 1.6.3 were read before implementation.
The code uses current `Image.uv_sync`, named Volumes and ephemeral Functions.
Official references: [images](https://modal.com/docs/guide/images),
[GPU allocation](https://modal.com/docs/guide/gpu),
[Volumes](https://modal.com/docs/guide/volumes),
[resource limits](https://modal.com/docs/guide/resources),
[pricing](https://modal.com/pricing).

## Budget and lifecycle

The initial research cap is **US$30 total**. The launcher reserves a conservative
allowance before creating an application. Reservations are cumulative and failures
do not refund them. The ledger is `artifacts/cloud-budget.json`, protected by a
local file lock. An additional ledger is retained on the artifacts Volume.
Never reset/delete ledgers to obtain more allowance. This launcher is intended
for one development machine, not distributed multi-user budget accounting.

Allowed single GPUs: T4 ($0.59/hour default), L4 ($0.80/hour), A10G ($1.10/hour),
verified using `uv run modal billing rates --json` on 2026-11-07. Each reservation
includes GPU, the configured maximum 2 CPU cores/8GiB, 600s startup allowance,
2s scale-down and $0.05 incidental allowance. $2 remains unavailable as a buffer.
Current prices are estimates; the workspace billing report is authoritative.
The code is not a provider-enforced billing cap. Keep the initial jobs far below
$30 and reconcile provider usage; other workspace applications are independent.

The function has 30–900s hard execution bounds, a 600s startup limit, one container,
no automatic retries, no schedule and no HTTP endpoint. No persistent GPU service
is deployed. CPU integrity inspection is a distinct Function. The launcher freezes
source files into an ignored local snapshot to prevent edits racing a cloud upload.
The image installs the frozen uv lock, with no competing dependency manager.

Named persistent Volumes:

| Volume | Mount | Contents |
|---|---|---|
| berrytrack-datasets | /datasets | Converted real frames, GT, detection splits |
| berrytrack-models | /models | Encoder/detector weights, optimizer/checkpoint metadata |
| berrytrack-features | /features | Extracted features and reusable inference caches |
| berrytrack-artifacts | /artifacts | Run JSON, budget ledger, validation artifacts |

## Launch commands

Authentication was already valid; no setup was repeated. On Windows, UTF-8 avoids
status-symbol failures in piped Modal CLI output:

```powershell
$env:PYTHONUTF8 = "1"
uv run modal billing rates --json
uv run modal billing summary --json
uv run scripts/modal_job.py smoke --seconds 120
```

Upload prepared data (the first smoke creates the named volumes):

```powershell
uv run modal volume put berrytrack-datasets data/processed/butom-st21-205 /butom-st21-205
uv run modal volume put berrytrack-datasets data/processed/butom-st21-235 /butom-st21-235
uv run scripts/prepare_detector.py
uv run modal volume put berrytrack-datasets data/processed/blueberrydcm-via-yolo /blueberrydcm-via-yolo
```

Train, resume the encoder and extract held-out development features:

```powershell
uv run scripts/modal_job.py train-encoder --seconds 180 --options-file configs/encoder-smoke.json
uv run scripts/modal_job.py train-encoder --seconds 120 --options-file configs/encoder-resume.json
uv run scripts/modal_job.py infer-encoder --seconds 120 --options-file configs/encoder-inference.json
uv run scripts/modal_job.py train-detector --seconds 180 --options-file configs/detector-via.json
uv run scripts/modal_job.py infer-detector --seconds 120 --options-file configs/detector-via-validation.json
```

Use `--gpu L4` or `--gpu A10G` for an explicitly priced alternative; allocation
does not silently fall back to a different GPU. The encoder refuses a CPU fallback
inside a GPU job. The training loop has step and elapsed limits, atomic checkpoints
and metadata. Detector training has epoch, callback and Modal hard-timeout limits;
`weights/resume.pt` preserves optimizer state before inference checkpoints are
stripped. Set `resume: true` in a detector options file to continue an interrupted
run. A fully completed training schedule is not an interrupted run.

Retrieve artifacts and inspect remaining applications:

```powershell
uv run modal volume get berrytrack-models blueberry-via/weights/best.pt checkpoints/blueberry-via/best.pt
uv run modal volume get berrytrack-models encoder-dev/encoder.pt checkpoints/encoder-dev/encoder.pt
uv run modal volume ls berrytrack-artifacts
uv run modal volume get berrytrack-artifacts RUN_ID/result.json artifacts/RUN_ID.json
uv run modal app list --json
```

The launcher also saves each successful result to `artifacts/modal/RUN_ID/result.json`.
Every result includes runtime, actual GPU name, peak allocated CUDA memory,
reservation and approximate runtime cost. Approximate runtime estimates exclude
image builds/startup and must not be presented as final invoices.

## Executed scope and limitations

Real T4 execution includes a CUDA infrastructure smoke, 30 encoder training steps
on BUTom205, resumption to step35, and 498 real held-out-development crop embeddings
from the first16 frames of BUTom235. The encoder has27,744 parameters. Its training
loss is not a tracking-quality metric. See committed run records under
`results/cloud/` for exact values and provenance.

The first detector pilot used publisher TXT labels, whose collapsed class mapping
was discovered during audit. It is excluded from four-class detector conclusions.
It was time bounded: an early Ultralytics `time` option overrode its requested
three-epoch count and ran105 partial/completed epochs within the allocation.
The final implementation removes that option, explicitly bounds epochs, and uses
VIA-derived four-class labels for the corrected20-epoch development adaptation.
Original pilot records are retained for transparency; this is not a hyperparameter
sweep. The corrected validation is image-level, with no guaranteed plant-level
independence. It supports execution verification, not field generalization claims.
