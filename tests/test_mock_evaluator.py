"""test_mock_evaluator.py - Unit tests for the MockEvaluator simulation model."""

import math
from beampass.evaluator import MockEvaluator


def test_mock_determinism():
    """MockEvaluator must yield identical latencies for identical inputs and seeds."""
    eval1 = MockEvaluator(model_name="small_cnn", seed=123)
    eval2 = MockEvaluator(model_name="small_cnn", seed=123)

    lat1 = eval1.evaluate(["canon", "CC", "I"])
    lat2 = eval2.evaluate(["canon", "CC", "I"])
    assert lat1 == lat2


def test_baseline_matches_empty_sequence():
    """baseline_o3() should evaluate to the baseline latency."""
    evaluator = MockEvaluator(model_name="small_cnn")
    base = evaluator.baseline_o3()
    empty_lat = evaluator.evaluate([])
    assert base == empty_lat
    assert base == MockEvaluator.BASE_LATENCIES["small_cnn"]


def test_channels_last_img2col_synergy():
    """Applying CC before I should produce a super-linear speedup compared to I alone."""
    evaluator = MockEvaluator(model_name="small_cnn", seed=42)
    base = evaluator.baseline_o3()

    lat_cc = evaluator.evaluate(["CC"])
    lat_i = evaluator.evaluate(["I"])
    lat_combo = evaluator.evaluate(["CC", "I"])

    # Both individual passes give minor speedup
    assert lat_cc < base
    assert lat_i < base

    # CC -> I combo should be significantly faster than both
    assert lat_combo < lat_cc
    assert lat_combo < lat_i

    # Speedup of combo should exceed the product of individual speedups due to synergy bonus
    speedup_combo = base / lat_combo
    speedup_cc = base / lat_cc
    speedup_i = base / lat_i
    assert speedup_combo > (speedup_cc * speedup_i * 0.95)

