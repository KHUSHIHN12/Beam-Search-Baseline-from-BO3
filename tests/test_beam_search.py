"""test_beam_search.py - Tests for beam search pruning, budget caps, and integration."""

import math
from typing import Dict, List, Sequence, Union

from beampass.beam_search import BeamSearch
from beampass.evaluator import Evaluator, MockEvaluator
from beampass.legality import IRInfo
from beampass.passes import Pass, PassVocabulary


class FailingMockEvaluator:
    """Mock evaluator where specific candidates fail (return inf)."""

    mode = "SIMULATED"
    model_name = "test_model"

    def __init__(self, fail_sequences: Sequence[str]):
        self.fail_sequences = set(fail_sequences)
        self.base_latency = 100.0

    def baseline_o3(self) -> float:
        return self.base_latency

    def evaluate(self, sequence: Sequence[Union[str, Pass]], final: bool = False) -> float:
        shorts = [s if isinstance(s, str) else s.short for s in sequence]
        seq_str = ",".join(shorts)
        if seq_str in self.fail_sequences:
            return math.inf
        # Return lower latency for longer valid sequences
        return round(self.base_latency * (0.9 ** len(shorts)), 2)

    def evaluate_detailed(self, sequence: Sequence[Union[str, Pass]], final: bool = False):
        lat = self.evaluate(sequence, final=final)
        return lat, 0.0


def test_beam_width_pruning():
    """Beam search must retain exactly the top B candidates at each depth."""
    evaluator = MockEvaluator(model_name="small_cnn", seed=42)
    beam_width = 3

    searcher = BeamSearch(
        evaluator=evaluator,
        beam_width=beam_width,
        max_depth=3,
        max_compiles=100,
    )
    res = searcher.run()

    # Check for each depth > 0 that number of kept nodes is at most beam_width
    for depth in range(1, 4):
        kept_at_depth = [n for n in res.nodes if n.depth == depth and n.kept_in_beam]
        assert len(kept_at_depth) <= beam_width
        assert len(kept_at_depth) > 0


def test_failed_candidates_are_dropped():
    """Candidates returning math.inf must never be kept in the beam."""
    evaluator = FailingMockEvaluator(fail_sequences=["canon", "CC,I"])
    searcher = BeamSearch(
        evaluator=evaluator,
        beam_width=2,
        max_depth=2,
        max_compiles=50,
    )
    res = searcher.run()

    # Verify that failed nodes were marked pruned and not kept
    failed_nodes = [n for n in res.nodes if math.isinf(n.latency)]
    for fn in failed_nodes:
        assert not fn.kept_in_beam


def test_budget_cap_enforcement():
    """Search must terminate immediately once max_compiles is reached."""
    evaluator = MockEvaluator(model_name="small_cnn", seed=42)
    budget = 15

    searcher = BeamSearch(
        evaluator=evaluator,
        beam_width=2,
        max_depth=5,
        max_compiles=budget,
    )
    res = searcher.run()
    assert res.total_compiles <= budget


def test_beam_search_never_worse_than_baseline():
    """Empty sequence fallback guarantees optimizer is never worse than -O3."""
    evaluator = FailingMockEvaluator(fail_sequences=["canon", "CC", "I", "T", "gen", "1x1", "pad", "intrinsics", "trans-mm"])
    searcher = BeamSearch(
        evaluator=evaluator,
        beam_width=2,
        max_depth=2,
    )
    res = searcher.run()
    # If all candidate passes fail, best latency should be <= baseline
    assert res.best_latency <= res.baseline_latency
    assert res.speedup >= 1.0


def test_integration_finds_known_optimal_sequence():
    """Beam search should discover the high-synergy sequence canon -> CC -> I."""
    evaluator = MockEvaluator(model_name="small_cnn", seed=42)
    searcher = BeamSearch(
        evaluator=evaluator,
        beam_width=2,
        max_depth=3,
        max_compiles=60,
    )
    res = searcher.run()

    # Known synergy target: contains CC and I
    assert "CC" in res.best_sequence
    assert "I" in res.best_sequence
    assert res.speedup > 1.35  # Significant speedup over -O3

