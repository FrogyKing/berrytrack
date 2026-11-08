"""Engineering checks: fixture counters are not research results."""

import runpy
from pathlib import Path


def evidence(rows):
    script = Path(__file__).resolve().parents[1] / "experiments" / "rigidity.py"
    return runpy.run_path(str(script))["rigidity_execution_evidence"](rows)


def row(fits):
    result = {
        "algorithm": "berrytrack",
        "config": {"local_rigidity": True},
        "sequence": "fixture",
        "variant": "rigidity",
        "seed": 42,
        "rigidity_anchor_count": 10,
    }
    if fits is not None:
        result["rigidity_fits"] = fits
    return result


def test_enabled_geometry_with_no_supported_fit_is_not_verified():
    result = evidence([row(0)])
    assert result["status"] == "not_exercised"
    assert result["runs_with_supported_fits"] == 0


def test_missing_counter_is_not_a_measured_zero():
    assert evidence([row(None)])["status"] == "not_measured"


def test_supported_fit_counts_are_reported_without_claiming_accuracy():
    result = evidence([row(4), row(0)])
    assert result["status"] == "exercised"
    assert result["runs_with_supported_fits"] == 1
    assert result["total_supported_fits"] == 4
    assert result["total_anchor_correspondences"] == 20
