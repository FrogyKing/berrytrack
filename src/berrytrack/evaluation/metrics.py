"""Adapters into TrackEval. No approximations of HOTA or identity metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: NDArray, b: NDArray) -> NDArray:
    """Pairwise IoU for absolute xyxy boxes, including empty sets."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    overlap = np.maximum(
        0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])
    )
    intersection = overlap.prod(axis=2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)


def _rows(rows: NDArray | Sequence, name: str) -> NDArray:
    arr = np.asarray(rows, dtype=np.float64)
    if not arr.size:
        return np.empty((0, 7), dtype=np.float64)
    arr = np.atleast_2d(arr)
    if arr.shape[1] < 7 or not np.isfinite(arr).all():
        raise ValueError(f"{name}: expected finite MOT rows with at least 7 columns")
    if np.any(arr[:, 0] < 1) or np.any(arr[:, 1] < 0) or np.any(arr[:, 4:6] <= 0):
        raise ValueError(f"{name}: frame>=1, ID>=0 and positive box dimensions required")
    if np.any(arr[:, :2] != np.floor(arr[:, :2])):
        raise ValueError(f"{name}: frame numbers and identities must be integers")
    if len(np.unique(arr[:, :2], axis=0)) != len(arr):
        raise ValueError(f"{name}: duplicate identity in a frame")
    return arr


def _xyxy(rows: NDArray) -> NDArray:
    boxes = rows[:, 2:6].copy()
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def trackeval_input(
    ground_truth: NDArray | Sequence,
    predictions: NDArray | Sequence,
    num_frames: int | None = None,
    annotated_frames: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Convert one foreground class of MOT rows to TrackEval's public data contract.

    GT column 7=0 is excluded. Class-specific distractor/ignore-region rules must
    be applied by the dataset converter, not guessed here. Explicit coverage
    prevents missing annotation files from being counted as empty ground truth.
    """
    gt, pred = _rows(ground_truth, "ground_truth"), _rows(predictions, "predictions")
    gt = gt[gt[:, 6] > 0]
    max_observed = int(max(gt[:, 0].max(initial=0), pred[:, 0].max(initial=0)))
    length = num_frames if num_frames is not None else max_observed
    if length < max_observed or length < 0:
        raise ValueError("num_frames cannot truncate observed rows")
    frames = list(annotated_frames) if annotated_frames is not None else list(range(1, length + 1))
    if len(set(frames)) != len(frames) or any(f < 1 or f > length or int(f) != f for f in frames):
        raise ValueError("annotated_frames must contain unique valid frame indices")
    frames.sort()
    gt, pred = gt[np.isin(gt[:, 0], frames)], pred[np.isin(pred[:, 0], frames)]
    gt_unique, pred_unique = np.unique(gt[:, 1]), np.unique(pred[:, 1])
    g_map = {int(value): i for i, value in enumerate(gt_unique)}
    p_map = {int(value): i for i, value in enumerate(pred_unique)}
    data: dict[str, Any] = {
        "num_timesteps": len(frames),
        "num_gt_ids": len(gt_unique),
        "num_tracker_ids": len(pred_unique),
        "num_gt_dets": len(gt),
        "num_tracker_dets": len(pred),
        "gt_ids": [],
        "tracker_ids": [],
        "similarity_scores": [],
        "frame_numbers": frames,
        "gt_original_ids": gt_unique.astype(int),
        "tracker_original_ids": pred_unique.astype(int),
    }
    for frame in frames:
        g, p = gt[gt[:, 0] == frame], pred[pred[:, 0] == frame]
        data["gt_ids"].append(np.asarray([g_map[int(i)] for i in g[:, 1]], dtype=int))
        data["tracker_ids"].append(np.asarray([p_map[int(i)] for i in p[:, 1]], dtype=int))
        data["similarity_scores"].append(iou_matrix(_xyxy(g), _xyxy(p)))
    return data


def identity_switch_events(data: dict[str, Any], threshold: float = 0.5) -> list[dict[str, int]]:
    """Locate switch frames using CLEAR's continuity-prioritized assignment."""
    last = np.full(data["num_gt_ids"], -1, dtype=int)
    previous = last.copy()
    events = []
    for frame, gt, pred, similarity in zip(
        data["frame_numbers"],
        data["gt_ids"],
        data["tracker_ids"],
        data["similarity_scores"],
        strict=True,
    ):
        if not len(gt) or not len(pred):
            continue
        score = similarity + 1000 * (pred[None, :] == previous[gt, None])
        score[similarity < threshold - np.finfo(float).eps] = 0
        rr, cc = linear_sum_assignment(-score)
        matched = score[rr, cc] > np.finfo(float).eps
        previous[:] = -1
        for r, c in zip(rr[matched], cc[matched], strict=True):
            g, p = gt[r], pred[c]
            if last[g] >= 0 and last[g] != p:
                events.append(
                    {
                        "frame": int(frame),
                        "gt_id": int(data["gt_original_ids"][g]),
                        "previous_id": int(data["tracker_original_ids"][last[g]]),
                        "new_id": int(data["tracker_original_ids"][p]),
                    }
                )
            last[g], previous[g] = p, p
    return events


def evaluate_mot(
    ground_truth: NDArray | Sequence,
    predictions: NDArray | Sequence,
    num_frames: int | None = None,
    annotated_frames: Sequence[int] | None = None,
) -> dict[str, Any]:
    from trackeval.metrics import CLEAR, HOTA, Identity

    data = trackeval_input(ground_truth, predictions, num_frames, annotated_frames)
    hota = HOTA().eval_sequence(data)
    clear = CLEAR({"PRINT_CONFIG": False, "THRESHOLD": 0.5}).eval_sequence(data)
    identity = Identity({"PRINT_CONFIG": False, "THRESHOLD": 0.5}).eval_sequence(data)
    gt_seen: set[int] = set()
    pred_seen: set[int] = set()
    errors = []
    for g, p in zip(data["gt_ids"], data["tracker_ids"], strict=True):
        gt_seen.update(g.tolist())
        pred_seen.update(p.tolist())
        errors.append(len(pred_seen) - len(gt_seen))
    final_error = data["num_tracker_ids"] - data["num_gt_ids"]
    return {
        **{key: float(np.mean(hota[key]) * 100) for key in ("HOTA", "DetA", "AssA")},
        "IDF1": float(identity["IDF1"] * 100),
        "MOTA": float(clear["MOTA"] * 100),
        "identity_switches": int(clear["IDSW"]),
        "fragmentations": int(clear["Frag"]),
        "true_positives": int(clear["CLR_TP"]),
        "false_positives": int(clear["CLR_FP"]),
        "false_negatives": int(clear["CLR_FN"]),
        "gt_unique_count": data["num_gt_ids"],
        "pred_unique_count": data["num_tracker_ids"],
        "count_error": final_error,
        "count_mae": float(abs(final_error)),
        "count_rmse": float(abs(final_error)),
        "cumulative_count_mae": float(np.mean(np.abs(errors))) if errors else 0.0,
        "cumulative_count_rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0,
        "evaluated_frames": data["num_timesteps"],
        "identity_switch_events": identity_switch_events(data),
        "protocol": (
            "TrackEval HOTA mean alpha=.05:.05:.95; CLEAR/Identity IoU>=.5; foreground only"
        ),
    }
