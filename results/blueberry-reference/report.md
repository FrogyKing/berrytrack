# Executed association diagnostic

EXPLORATORY EXECUTION CHECK ONLY. Eight near-stationary real blueberry ROI frames with seven identities authored by exploratory visual inspection. Reference boxes are also tracker inputs with IDs removed. No independent human-verified GT, full-scene count, detector accuracy or H1-H4 evidence.

Percent metrics use 0–100; latency excludes detector, disk IO and the first 5 frames. Single-sequence count MAE/RMSE are absolute final-count errors, not population estimates.

| Sequence | Variant | Seed | HOTA | IDF1 | IDSW | Count error | Association ms |
|---|---|---:|---:|---:|---:|---:|---:|
| agrivision-c1-v1-visual-roi | legacy | 42 | 87.50 | 93.33 | 0 | 0 | 4.300 |
| agrivision-c1-v1-visual-roi | polygon | 42 | 87.50 | 93.33 | 0 | 0 | 6.145 |
| agrivision-c1-v1-visual-roi | static_polygon_experimental | 42 | 87.50 | 93.33 | 0 | 0 | 5.858 |
| agrivision-c1-v1-visual-roi | bytetrack | 42 | 100.00 | 100.00 | 0 | 0 | 0.378 |
| agrivision-c1-v1-visual-roi | botsort | 42 | 100.00 | 100.00 | 0 | 0 | 2.013 |
| agrivision-c1-v1-visual-roi | deepsort | 42 | 87.50 | 93.33 | 0 | 0 | 2.204 |

Executed at 2026-11-08T02:18:44.481214+00:00. See results.json for environment, hashes, settings, coverage and limitations.
