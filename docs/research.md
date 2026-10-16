# Association model and research protocol

The research question is whether adaptive appearance, camera compensation and
local spatial context improve fruit identity association and counting at a useful
computational cost. The current evidence is exploratory and includes negative
results; this is an implemented hypothesis, not a novelty or superiority claim.

## State and observations

Detection boxes use zero-based absolute pixel coordinates internally. MOT exports
use one-based xywh. Identity state is

$$x_i=(c_x,c_y,w,h,v_x,v_y,v_w,v_h)^T,\quad P_i\succeq0.$$

A constant-velocity Kalman filter predicts state and covariance. Box-size-scaled
process/measurement noise is an engineering assumption. The Joseph covariance
update reduces numerical loss of positive semidefiniteness. Motion gating uses
the center covariance, not the full four-dimensional box state.

## Camera model

ORB features outside foreground boxes are ratio matched. Partial affine RANSAC
estimates rotation, isotropic scale and translation. Reject insufficient inliers,
low inlier fraction, localized support, implausible scale/displacement, or excessive
residual. Rejection returns identity with zero reliability; it does not claim the
camera was stationary. Accepted transforms warp centers, velocities and covariance
with a Jacobian; sizes become an axis-aligned transformed envelope.

This is a 2D approximation. A greenhouse contains non-planar objects at different
depths, moving leaves and parallax. Neither a global affine transform nor a
homography fully explains it. No 3D reconstruction is claimed. The conservative
checks can reject useful motion or accept a dominant plane that harms another depth.

## Comparable costs

For admissible identity/detection edges,

$$C_{ij}=\alpha_{ij} A_{ij}+\beta_{ij} M_{ij}
+\gamma_{ij} G_{ij}+\delta_{ij} L_{ij},\quad
\alpha+\beta+\gamma+\delta=1.$$

All terms lie in $[0,1]$:

$$A_{ij}=\frac{1-\cos(f_i,f_j)}{2},\qquad
M_{ij}=\min\left(1,\frac{\sqrt{\Delta c^TS_i^{-1}\Delta c}}{r}\right).$$

$$G_{ij}=\tfrac12\min\left(1,\frac{\|c_i-c_j\|}{4\sqrt{w_i h_i}}\right)
+\tfrac12(1-\operatorname{IoU}(b_i,b_j)).$$

The geometric context descriptor stores the k nearest neighbors' bounded radial
distance, relative unit direction and occupancy, normalized by fruit scale. $L$
is mean absolute descriptor difference, clipped to $[0,1]$. This preserves local
geometry without creating a dense attention graph. Descriptor neighbor ordering
can change during occlusion, and direction channels are not rotation invariant.

Base coefficients are $(.30,.30,.25,.15)$. Adaptive multipliers are

$$\tilde w_i = w_0\odot
\left[r_{a,i}(1+u_i),\;(1+u_i)^{-1},\;.35+.65r_c,\;r_l\right],$$

where $u_i=\min(3,\text{missed}_i/\text{max_age})$, $r_a$ measures available
appearance, $r_c$ is accepted camera support, and $r_l$ is neighborhood occupancy.
Disabled/missing terms receive zero weight and the remaining terms are renormalized
per pair. These heuristics have not been calibrated on a broad independent set.
Hyperparameters remain fixed in the committed diagnostic.

Motion Mahalanobis and class gates eliminate implausible candidates. Matching
first uses high-confidence detections, then permits active confirmed identities
to recover low-confidence detections. Low-confidence observations cannot create
new identities. Hungarian assignment uses an unmatched dummy for each track;
invalid or over-threshold edges cannot displace admissible assignments.

## Appearance and sparse aggregation

The CPU diagnostic uses HSV histograms and low-frequency DCT, with a seeded
projection to 32, 64 or 128 dimensions. It is an **untrained** descriptor with
zero learned parameters. This keeps scientific interpretation distinct from the
separately implemented trainable encoder.

`CompactEncoder` is a three-convolution PyTorch network with 27,744 parameters at
64 dimensions. Supervised contrastive training uses original track identities
and paired deterministic crop augmentations. Self pairs are excluded; anchors
without positive partners do not contribute. Checkpoints store optimizer, step,
seed and RNG states. Resume restarts the data iterator and is not bit-exact
continuation of an uninterrupted dataloader. The executed training on sequence
205 is development training; both 205 and diagnostic 235 belong to the dataset's
official evaluation partition, so this repository does not claim official scores.

The YOLO adapter can ROI-pool its already-computed feature pyramid into normalized
projected embeddings. This path avoids a second network per object. Its execution
has been smoke checked; its tracking quality has not been benchmarked against
crop descriptors. The default diagnostic does not silently use this alternative.

`SparseContextAttention` implements an optional learnable graph alternative:

$$m_{ij}=W_m[f_j,(c_j-c_i)/s_i],\quad
a_{ij}=\operatorname{softmax}_{j\in N_k(i)}\operatorname{MLP}([f_j,\Delta c/s_i]),$$
$$h_i=\operatorname{normalize}(f_i+\sum_{j\in N_k(i)}a_{ij}m_{ij}).$$

No untrained graph weights enter reported tracking results. Training and a
controlled attention comparison remain pending. The deterministic local descriptor
is the actual context term in the current tracker.

## Lazy appearance and lifecycle

Compute crop embeddings for new identities and ambiguous candidates, competing
tracks, occlusion recovery, or unreliable camera estimates. Easy geometry matches
can skip extraction; cached backbone features can also be supplied directly.
This reduces crops, but stale identity features can harm associations or counts.

Tentative identities need `min_hits` matches. Confirmed tracks can become lost,
then remain in memory until `memory_age`. Recovery beyond `max_age` requires
appearance evidence and a conservative spatial gate. A recovered identity retains
its original count. Each confirmed ID contributes once to the cumulative count.
After memory expiry or ambiguous re-entry, physical identity cannot be guaranteed;
duplicate counting is a known failure. This is not an absolute fruit census.

## Complexity and validity

With T remembered tracks, D detections, k neighbors and embedding dimension d,
the kNN graph uses approximately $O(D\log D+Dk)$ work and $O(Dk)$ context storage.
Current cost construction uses $O(TD(d+k))$ work and $O(TD)$ storage. Hungarian
assignment with dummy columns is dense and has cubic worst-case scaling. Candidate
gating reduces admissible matches but does not yet make matrix allocation sparse.
The Python reference implementation prioritizes inspectability over extreme density.

Detector quality, association quality, pipeline timing and cloud billing are
separate measurements. Reference-box diagnostics remove the detector but remain
sensitive to localization, confirmation and occlusion policies. A single clip
cannot establish H1–H4 or agricultural generalization. See [the executed evaluation](evaluation.md)
for negative findings and [the dataset audit](datasets.md) for access/label limitations.

Future tests must reserve whole recordings before annotation, use identical cached
detector outputs, select models on development/validation data, and evaluate the
final untouched test recordings once. Report all seeds, failures, detector AP,
HOTA/DetA/AssA, IDF1, switches/fragments, sequence count MAE/RMSE, timing, memory,
parameters and cost. Human-verified blueberry identity annotations remain required.
