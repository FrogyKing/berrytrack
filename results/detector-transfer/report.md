# Executed association diagnostic

Cached detector association experiment. Detector quality must be reported separately; consult detection cache provenance.

Percent metrics use 0–100; latency excludes detector, disk IO and the first 5 frames. Single-sequence count MAE/RMSE are absolute final-count errors, not population estimates.

| Sequence | Variant | Seed | HOTA | IDF1 | IDSW | Count error | Association ms |
|---|---|---:|---:|---:|---:|---:|---:|
| butom-st21-235 | bytetrack | 42 | 20.48 | 14.99 | 3 | 69 | 1.051 |
| butom-st21-235 | botsort | 42 | 27.13 | 24.63 | 1 | 65 | 8.690 |
| butom-st21-235 | deepsort | 42 | 33.29 | 35.75 | 46 | 105 | 32.748 |
| butom-st21-235 | berrytrack | 42 | 26.69 | 25.42 | 7 | 63 | 25.721 |

Executed at 2026-11-08T01:40:37.382842+00:00. See results.json for environment, hashes, settings, coverage and limitations.
