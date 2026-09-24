# Reproducible experiments

After preparing an official sequence, run the CPU association diagnostic:

```powershell
uv run python -m berrytrack.evaluation.benchmark --sequence data/processed/butom-st21-235 --oracle --ablations --dimensions 32 64 128 --seeds 42 --output artifacts/diagnostic
```

The `--oracle` flag copies human-corrected bounding boxes, discards their IDs,
sets detection confidence to one, and caches the resulting detections once.
Each algorithm sees identical detections and image frames. These results describe
association with ideal boxes on an accessible **tomato development subset**.
They are not blueberry detector performance or held-out benchmark claims.

For detector experiments, prepare a `det/det.txt` file in the sequence directory
and omit `--oracle`. Save detector weights hash, inference settings, training
split and detector metrics alongside that cache. Do not use GT-derived boxes for
detector metrics. The runner produces full configurations, hashes, predictions,
TrackEval metrics, CSV, Markdown and a comparison plot. GPU training and inference
are separate, explicitly budgeted commands documented in the main README.

Seven cumulative ablations are included. The final `uncertainty` variant is
identical to the full BerryTrack configuration; this duplicate makes the ladder
explicit. Embedding dimension comparisons use 32/64/128 compact descriptors.
Multiple seeds are supported, but deterministic classical methods do not gain
statistical independence merely by repeating seeds. Broader sequence sampling,
manual blueberry identities, detector adaptation and trained ReID comparisons
remain necessary for testing H1–H4.
