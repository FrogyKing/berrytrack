import numpy as np

from berrytrack.geometry.rigidity import refine_polygon_cost
from berrytrack.tracking.kalman import BoxKalman


def boxes(xy):
    xy = np.asarray(xy, dtype=np.float32)
    return np.column_stack((xy - 5, xy + 5))


def test_polygon_prefers_smooth_affine_deformation_over_better_unary_jump():
    xy = np.array(
        [[-20, -20], [0, -20], [20, -20], [-20, 20], [0, 20], [20, 20], [0, 0]], dtype=np.float32
    )
    affine = np.array([[1, 0.12], [-0.04, 1.02]], dtype=np.float32)
    transformed = xy @ affine.T + [5, 3]
    observed = np.vstack((transformed, [50, 50]))
    cost = np.full((7, 8), 0.4, np.float32)
    cost[6, 6], cost[6, 7] = 0.5, 0.1
    result = refine_polygon_cost(
        boxes(xy), boxes(observed), cost, np.ones_like(cost, bool), [(i, i) for i in range(6)]
    )
    assert result.reliability[6] > 0.9
    assert np.linalg.norm(result.predictions[6] - transformed[6]) < 1
    assert result.cost[6, 6] < result.cost[6, 7]
    assert not result.gate[6, 7]


def test_collinear_and_missing_support_fall_back_without_fabricating_geometry():
    previous = boxes([[0, 0], [10, 0], [20, 0], [15, 10]])
    cost = np.full((4, 4), 0.2, np.float32)
    for anchors in ([(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)]):
        result = refine_polygon_cost(previous, previous, cost, np.ones_like(cost, bool), anchors)
        assert np.allclose(result.cost, cost)
        assert not result.reliability.any()


def test_local_mapping_rejects_mirrored_polygon():
    previous = boxes([[-10, -10], [10, -10], [-10, 10], [10, 10], [0, 0]])
    reflected = previous.copy()
    reflected[:, [0, 2]] = -previous[:, [2, 0]]
    cost = np.zeros((5, 5), np.float32)
    result = refine_polygon_cost(
        previous, reflected, cost, np.ones_like(cost, bool), [(i, i) for i in range(4)]
    )
    assert result.reliability[4] == 0


def test_scene_prior_damps_residual_velocity_after_camera_motion():
    box = boxes([[20, 20]])[0]
    generic, stationary = BoxKalman(box), BoxKalman(box)
    generic.mean[4], stationary.mean[4] = 10, 10
    camera = np.array([[1, 0, 5], [0, 1, 0]], np.float32)
    generic.predict(camera)
    stationary.predict(camera, static_scene=True)
    assert generic.mean[0] == 35
    assert stationary.mean[0] == 27
    assert np.linalg.eigvalsh(stationary.covariance).min() >= 0


def test_stationary_world_does_not_imply_stationary_pixels_without_camera_estimate():
    filter = BoxKalman(boxes([[20, 20]])[0])
    filter.mean[4] = 10
    filter.predict(static_scene=True)
    assert filter.mean[0] == 30


def test_supported_polygon_recovers_only_eligible_globally_gated_candidates():
    xy = np.array(
        [[-20, -20], [0, -20], [20, -20], [-20, 20], [0, 20], [20, 20], [0, 0]], np.float32
    )
    previous, observed = boxes(xy), boxes(xy + [5, 3])
    cost = np.full((7, 7), 0.3, np.float32)
    gate = np.ones_like(cost, bool)
    gate[6] = False
    eligible = np.ones_like(gate)
    for allowed in (True, False):
        eligible[6, 6] = allowed
        result = refine_polygon_cost(
            previous, observed, cost, gate, [(i, i) for i in range(6)], recovery_eligible=eligible
        )
        assert bool(result.gate[6, 6]) == allowed
