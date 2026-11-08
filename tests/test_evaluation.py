import numpy as np
import pytest

from berrytrack.evaluation.metrics import evaluate_mot, iou_matrix, trackeval_input


def rows(ids):
    return np.array(
        [[frame, identity, 11, 11, 20, 20, 1, -1, -1, -1] for frame, identity in enumerate(ids, 1)],
        dtype=float,
    )


def test_trackeval_perfect_track_and_count():
    gt = rows([42, 42, 42, 42])
    result = evaluate_mot(gt, rows([8, 8, 8, 8]), 4)
    for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA"):
        assert result[metric] == pytest.approx(100)
    assert result["identity_switches"] == 0
    assert result["count_mae"] == 0
    assert result["identity_switch_events"] == []


def test_identity_switch_is_not_detection_failure():
    gt = rows([42, 42, 42, 42])
    result = evaluate_mot(gt, rows([8, 8, 9, 9]), 4)
    assert result["DetA"] == pytest.approx(100)
    assert result["IDF1"] == pytest.approx(50)
    assert result["identity_switches"] == 1
    assert result["AssA"] == pytest.approx(50)
    assert result["count_mae"] == 1
    assert result["identity_switch_events"] == [
        {"frame": 3, "gt_id": 42, "previous_id": 8, "new_id": 9}
    ]


def test_fragmentation_and_explicit_annotation_coverage():
    gt = rows([1, 1, 1])
    pred = rows([5, 5, 5])[[0, 2]]
    result = evaluate_mot(gt, pred, 3)
    assert result["false_negatives"] == 1
    covered = evaluate_mot(gt, pred, 3, annotated_frames=[1, 3])
    assert covered["HOTA"] == pytest.approx(100)
    assert covered["evaluated_frames"] == 2


def test_duplicate_id_and_invalid_coverage_rejected():
    gt = rows([1, 1])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_mot(np.vstack([gt, gt[:1]]), gt, 2)
    with pytest.raises(ValueError, match="annotated_frames"):
        trackeval_input(gt, gt, 2, [1, 1])
    with pytest.raises(ValueError, match="truncate"):
        trackeval_input(gt, gt, 1)


def test_empty_predictions_are_false_negatives_not_perfect():
    result = evaluate_mot(rows([1, 1]), [], 2)
    assert result["HOTA"] == 0
    assert result["IDF1"] == 0
    assert result["false_negatives"] == 2
    assert result["count_error"] == -1


def test_pairwise_iou_handles_empty_and_partial_overlap():
    assert iou_matrix([], [[0, 0, 10, 10]]).shape == (0, 1)
    result = iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])
    assert result[0, 0] == pytest.approx(1 / 3)


def test_unique_count_mae_and_rmse_aggregate_signed_sequence_errors():
    from berrytrack.evaluation.benchmark import aggregate_counts

    results = [
        {"variant": "a", "seed": 42, "metrics": {"count_error": -3}},
        {"variant": "a", "seed": 42, "metrics": {"count_error": 4}},
    ]
    aggregate = aggregate_counts(results)[0]
    assert aggregate["count_mae"] == 3.5
    assert aggregate["count_rmse"] == pytest.approx(np.sqrt(12.5))
