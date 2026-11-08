# GPU training with Modal

BerryTrack uses ephemeral Modal Functions for encoder and detector training,
checkpoint resumption and feature extraction. Named Volumes keep data, models,
features and run artifacts available across executions.

Official references: [images](https://modal.com/docs/guide/images),
[GPU allocation](https://modal.com/docs/guide/gpu),
[Volumes](https://modal.com/docs/guide/volumes) and
[resource configuration](https://modal.com/docs/guide/resources).

## Execution model

Jobs run in one container with an explicit execution timeout, a startup timeout,
no automatic retries and no public endpoint. CPU integrity inspection uses a
separate Function. Source files are frozen into a local snapshot before upload,
and the image installs the frozen uv lock.

| Volume | Mount | Contents |
|---|---|---|
| berrytrack-datasets | /datasets | Converted frames, GT and detection splits |
| berrytrack-models | /models | Encoder/detector weights and checkpoint metadata |
| berrytrack-features | /features | Extracted features and reusable inference caches |
| berrytrack-artifacts | /artifacts | Run JSON and validation artifacts |

## Prepare and upload

Install project dependencies and authenticate with Modal. On Windows, enable UTF-8
for piped CLI output:

```powershell
uv sync --locked --dev
uv run modal token new
$env:PYTHONUTF8 = "1"
uv run scripts/modal_job.py smoke --seconds 120
```

Prepare the datasets locally and upload them to the named Volumes:

```powershell
uv run scripts/prepare_public_data.py butom
uv run scripts/prepare_public_data.py blueberrydcm
uv run scripts/prepare_detector.py
uv run modal volume put berrytrack-datasets data/processed/butom-st21-205 /butom-st21-205
uv run modal volume put berrytrack-datasets data/processed/butom-st21-235 /butom-st21-235
uv run modal volume put berrytrack-datasets data/processed/blueberrydcm-via-yolo /blueberrydcm-via-yolo
```

## Train and resume

```powershell
uv run scripts/modal_job.py train-encoder --seconds 180 --options-file configs/encoder-smoke.json
uv run scripts/modal_job.py train-encoder --seconds 120 --options-file configs/encoder-resume.json
uv run scripts/modal_job.py infer-encoder --seconds 120 --options-file configs/encoder-inference.json
uv run scripts/modal_job.py train-detector --seconds 180 --options-file configs/detector-via.json
uv run scripts/modal_job.py infer-detector --seconds 120 --options-file configs/detector-via-validation.json
```

T4 is the default; `--gpu L4` and `--gpu A10G` select alternatives explicitly.
Allocation does not silently switch GPU types. The encoder requires CUDA inside
the GPU job. Training has step/elapsed limits, atomic checkpoints and metadata.
Detector training uses epochs, a callback and a Function timeout;
`weights/resume.pt` preserves optimizer state. Set `resume: true` in an options
file to continue an interrupted run.

## Retrieve artifacts

```powershell
uv run modal volume get berrytrack-models blueberry-via/weights/best.pt checkpoints/blueberry-via/best.pt
uv run modal volume get berrytrack-models encoder-dev/encoder.pt checkpoints/encoder-dev/encoder.pt
uv run modal volume ls berrytrack-artifacts
uv run modal volume get berrytrack-artifacts RUN_ID/result.json artifacts/RUN_ID.json
uv run modal app list --json
```

The launcher also saves results to `artifacts/modal/RUN_ID/result.json` locally.
Public [execution records](../results/cloud/runs.json) retain hardware, runtime,
peak allocated CUDA memory, training settings, checkpoint provenance and failures.
They contain experiment metadata rather than workstation administration records.

## Executed scope

Real T4 execution includes a CUDA smoke, 30 encoder training steps on BUTom205,
resumption to step35 and 498 crop embeddings from the first16 frames of BUTom235.
The encoder has27,744 parameters. Training loss is not a tracking-quality metric;
the default association comparison uses the compact untrained descriptor.

The first detector pilot used publisher TXT labels with a collapsed class mapping
and is excluded from four-class conclusions. An initial training-time option
overrode its requested three-epoch count; the corrected implementation uses
explicit epoch and callback limits. Corrected training uses VIA-derived four-class
labels for20 development epochs. The image-level validation split does not
guarantee plant-level independence. See the [dataset audit](datasets.md) and
[executed evaluation](evaluation.md) for interpretation.
