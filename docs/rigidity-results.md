# Local polygon consistency: executed comparison

Fruit are assumed approximately stationary in the world over a short sequence,
while their local image polygon can deform smoothly with camera viewpoint. The
implemented local-affine constraint tests this assumption without demanding
constant pixel positions or distances. See [the formulation](research.md).

The shipped default enables `local_rigidity` with fixed weight 0.35 and leaves
experimental residual-velocity damping (`static_scene`) off. Camera estimation,
appearance, uncertainty and the other association settings are unchanged in
the paired legacy/polygon comparison. No detector/encoder was retrained for this
extension and no additional GPU budget was spent.

## Protocol and executed evidence

The [corrected experiment](../results/rigidity/index.json) has63 executed runs:
seven methods, three descriptor seeds, two real BUTom sequences with reference
boxes (42 runs), plus actual cached detections on 235 (21 runs). Sequence205 has
no actual detector cache and is explicitly pending in that mode. GT identities
are removed from reference detection inputs. This is development evaluation,
not an official held-out benchmark or a blueberry generalization claim.

Mean HOTA across seeds42/43/44, on human-corrected reference boxes:

| Sequence | Legacy BerryTrack | Local polygons (default) | Damping only | Damping + polygons | BoT-SORT |
|---|---:|---:|---:|---:|---:|
| BUTom 235 | 83.65 | 84.17 | 68.63 | 70.38 | 82.44 |
| BUTom 205 | 89.92 | 89.85 | 43.29 | 50.40 | 86.22 |

On 235, local polygons reduce mean switches from 31.67 to 19 and increase IDF1 from
91.35 to 92.13, while final-count MAE stays4. On 205, HOTA decreases slightly and
count MAE remains1. Thus the polygon constraint gives a modest, mixed result;
there is no demonstrated general superiority or counting improvement.

With actual transferred-detector outputs on 235, mean HOTA stays26.69 for legacy
and local polygons. BoT-SORT gives27.13. The detector/visibility bottleneck is
not fixed by neighborhood constraints. Local fits are exercised about 1,622 times
per run on oracle235 and 1,442 on oracle205, but only9 times on the actual cached
235 detections. Sparse support correctly limits when geometry can contribute.

Full [oracle metrics](../results/rigidity/oracle/summary_by_variant.csv) and
[actual-detection metrics](../results/rigidity/cached/summary_by_variant.csv)
include DetA, AssA, IDF1, MOTA, switches, fragments and count errors. JSON records
also include per-frame timings, source hashes, settings and supported-fit counts.
The Windows host's background load was not isolated; observed timings vary widely.
They are retained transparently and must not support a speedup claim. Local
fitting adds computation. Controlled timing on an isolated host remains pending.

## Preserved negative initial implementation

The [initial diagnostic](../results/rigidity-initial/index.json) is preserved,
including its63 runs and source snapshot. Unconditionally damping image velocity
was an invalid use of world stationarity when camera displacement was unexplained.
It also tightened motion covariance before the local step could recover edges.

The single validity correction conditions damping on accepted camera support,
keeps mean/covariance transitions consistent, and permits strongly supported
local maps to recover otherwise eligible active candidates rejected by a global
motion gate. Weight, neighborhood size and other thresholds were not swept.
The corrected damping mode still performs badly, plausibly because a global2D
camera approximation does not explain all depth-dependent image motion. That
causal attribution is an inference, not a measured parallax stratification.
Damping remains an experimental option, disabled by default.

The correction and default decision use these explicitly declared development
subsets. No independent final test was used for model selection. Preserve both
the negative initial records and the corrected records when reproducing the work.

## Blueberry execution, without inventing benchmark GT

AgriVision DB-2 supplied 24 real consecutive blueberry frames and publisher-inferred
masks. The frames were reconstructed as H.264;10 FPS is chosen playback speed,
because original capture FPS is unavailable. Full-scene inference uses the same
detector outputs for BerryTrack and BoT-SORT, with no full-scene accuracy claim.

A separately authored tiny reference covers8 consecutive260x260 crops and 7
visible identities (56 boxes), with estimated5 px box-boundary uncertainty. Its
provenance is `exploratory_visual_review`, `human_verified=false`, and
`independent_benchmark=false`. Boxes/IDs were inspected visually; no tracker
prediction or inferred segmentation mask supplied their identities.

[Exploratory reference results](../results/blueberry-reference/report.md) only
verify execution on real blueberry pixels. Local polygons and legacy both yield
HOTA 87.5 and zero switches/count error; upstream ByteTrack/BoT-SORT yield100 on
this easy reference-box crop because of different initial confirmation policy.
The crop is near-stationary, has no occlusion transitions or re-entry, and does
not test the main hypothesis. These numbers are not independent blueberry MOT
accuracy and must not be used as evidence of superiority.

## Reproduction

```powershell
uv run python experiments/rigidity.py --output artifacts/rigidity --mode both --seeds 42 43 44
uv run scripts/prepare_agrivision.py --output data/processed/agrivision-c1v1-new --reference-output data/processed/agrivision-reference-new
uv run python experiments/blueberry.py --sequence data/processed/agrivision-reference-new --output artifacts/blueberry-reference
```

Output directories for ingestion must be new/empty to prevent mixing sequences.
The downloader pins source archive metadata and refuses servers that ignore
HTTP Range. The full78 GB archive MD5 cannot be verified for a partial extraction;
per-member CRC32 and extracted SHA256 are recorded. Labels are pinned to the exact
visually inspected source frame hashes. Details: [AgriVision workflow](agrivision.md).
