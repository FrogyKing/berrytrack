# Local video application

Launch with `uv run berrytrack demo` and open http://127.0.0.1:7860.
The interface runs on localhost, with one queued task at a time, without a public
share link or GPU endpoint. CLI inference uses the same pipeline.

1. Upload a real video.
2. Choose BerryTrack, ByteTrack, BoT-SORT or compact-descriptor DeepSORT, and an
   optional comparison. Both see the same per-frame detections.
3. Supply local adapted weights. In the advanced section, an optional MOT cache
   bypasses the detector; a verified MOT GT file enables identity-switch metrics.
4. Run and inspect persistent IDs, short trajectories, confirmed unique-ID counts,
   comparison timing, switch rows and up to six measured switch frames.
5. Download H.264 video, each algorithm's MOT tracks and JSON statistics.

The maximum frame setting and ten-minute CPU wall bound prevent accidental
unbounded processing. The exported record states truncation and timing scope.
Videos may be at most3840x2160 pixels. Result exports live in temporary local
directories; CLI exports use the chosen output path. Rendering/transcoding has
its own bounded subprocess. CPU pipeline wall timing includes frame decode,
shared detector, trackers and rendering, and excludes final H.264 transcoding.

The uploaded GT must cover every processed frame. The system cannot infer unknown
annotation coverage from an empty/missing row. For partial GT, use the evaluation
API's explicit annotated-frame list rather than representing gaps as empty frames.
GT is never required for ordinary inference or counting.

For a licensed reference sample, run the public BUTom preparation, the oracle
association command and `uv run scripts/make_demo_video.py`. Upload
`artifacts/demo/input.mp4`, the generated reference detection cache and sequence235
GT. This demonstrates association with reference boxes. It cannot measure detector
accuracy. Source conversion/compression can change appearance and camera estimates,
so do not compare its scores numerically with the original-frame benchmark.

For a fresh checkout without weights, this reference-cache path works on CPU.
Adapted weights can be trained/retrieved using [the Modal commands](modal.md).
No raw dataset or model artifact is tracked in Git.

Engineering verification includes a generated video fixture only for codec,
comparison, identity continuity and MOT/JSON export contracts. The real reference
video, real detector inference, learned encoder inference and direct Gradio callback
were also executed locally. Synthetic test fixtures never enter research tables.
