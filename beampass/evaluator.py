"""evaluator.py - Evaluator interface, IREEEvaluator, and MockEvaluator."""

from __future__ import annotations

import hashlib
import math
import os
import random
import tempfile
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Union

from beampass.benchmark import BenchmarkRunner
from beampass.cache import EvaluationCache
from beampass.compiler import IREECompiler
from beampass.passes import Pass, PassVocabulary, get_default_vocabulary


class Evaluator(Protocol):
    """Protocol defining compilation and benchmarking evaluation interface."""

    mode: str
    model_name: str

    def baseline_o3(self) -> float:
        """Returns baseline -O3 compilation latency in milliseconds."""
        ...

    def evaluate(self, sequence: Sequence[Union[str, Pass]], final: bool = False) -> float:
        """Evaluates a pass sequence and returns latency in milliseconds."""
        ...

    def evaluate_detailed(
        self,
        sequence: Sequence[Union[str, Pass]],
        final: bool = False,
    ) -> Tuple[float, float]:
        """Returns (mean_latency, std_dev) in milliseconds."""
        ...


class MockEvaluator:
    """Deterministic simulated latency model for tests, offline demos, and platforms without IREE.

    Explicitly tagged as 'SIMULATED'. Implements realistic compiler optimization behaviors:
    - Base latencies per model architecture
    - Individual pass impacts
    - Strong pair synergies (e.g. Channels-Last (CC) followed by Img2Col (I) unlocks vectorization)
    - Diminishing returns on repeated passes
    - Reproducible seeded jitter
    """

    mode: str = "SIMULATED"

    # Base latencies in milliseconds for different models
    BASE_LATENCIES: Dict[str, float] = {
        "small_cnn": 24.50,
        "resnet18": 68.20,
        "mobilenet_v2": 45.60,
        "mlp": 12.30,
        "default": 1000.0,
    }

    # Multipliers for individual passes
    PASS_MULTIPLIERS: Dict[str, float] = {
        "canon": 0.96,
        "canonicalize": 0.96,
        "CC": 0.89,
        "convert-conv-to-channels-last": 0.89,
        "I": 0.95,
        "convert-conv2d-to-img2col": 0.95,
        "T": 0.97,
        "transpose-convolution-pipeline": 0.97,
        "gen": 0.98,
        "linalg-generalize-named-ops": 0.98,
        "1x1": 0.96,
        "convert-1x1-filter-conv2d-to-matmul": 0.96,
        "pad": 0.99,
        "pad-linalg-ops": 0.99,
        "intrinsics": 0.97,
        "pad-to-intrinsics": 0.97,
        "trans-mm": 0.98,
        "transpose-matmul": 0.98,
    }

    def __init__(
        self,
        model_name: str = "small_cnn",
        seed: int = 42,
        cache: Optional[EvaluationCache] = None,
        knob_config: str = "default_knobs",
    ):
        self.model_name = model_name
        self.seed = seed
        self.cache = cache
        self.knob_config = knob_config
        self.base_latency = self.BASE_LATENCIES.get(
            model_name.lower().replace("-", "_"),
            self.BASE_LATENCIES["default"],
        )
        self._cached_baseline: Optional[float] = None

    def _normalize_seq(self, sequence: Sequence[Union[str, Pass]]) -> List[str]:
        res = []
        for item in sequence:
            if isinstance(item, Pass):
                res.append(item.short)
            else:
                res.append(str(item))
        return res

    def baseline_o3(self) -> float:
        """Returns baseline -O3 latency for the model."""
        if self._cached_baseline is None:
            self._cached_baseline = self.evaluate([], final=False)
        return self._cached_baseline

    def _compute_mock_latency(self, seq_shorts: List[str]) -> float:
        """Calculates deterministic synthetic latency with realistic synergies."""
        if not seq_shorts:
            return self.base_latency

        current_latency = self.base_latency
        applied_counts: Dict[str, int] = {}

        # 1. Individual pass effects with diminishing returns
        for p in seq_shorts:
            count = applied_counts.get(p, 0)
            mult = self.PASS_MULTIPLIERS.get(p, 1.0)
            if count > 0:
                # Diminishing returns: 50% effectiveness on second pass
                mult = 1.0 - (1.0 - mult) * 0.4
            current_latency *= mult
            applied_counts[p] = count + 1

        # 2. Synergies and interactions
        # CC followed by I (Channels Last then Img2Col is a major synergy in IREE)
        if "CC" in seq_shorts and "I" in seq_shorts:
            cc_idx = seq_shorts.index("CC")
            i_idx = seq_shorts.index("I")
            if cc_idx < i_idx:
                current_latency *= 0.83  # Big win when CC comes before I
            else:
                current_latency *= 0.96  # Mild win if reverse

        # Canonicalize early helps simplify before heavy transformations
        if seq_shorts and seq_shorts[0] in ("canon", "canonicalize"):
            current_latency *= 0.95

        # 3. Deterministic pseudo-random variation based on sequence hash
        seq_hash = hashlib.md5(f"{self.seed}_{','.join(seq_shorts)}".encode("utf-8")).hexdigest()
        jitter_pct = ((int(seq_hash[:6], 16) % 1000) / 1000.0 - 0.5) * 0.02  # +/- 1%
        current_latency *= (1.0 + jitter_pct)

        return round(current_latency, 3)

    def evaluate(self, sequence: Sequence[Union[str, Pass]], final: bool = False) -> float:
        """Evaluates pass sequence latency."""
        seq_shorts = self._normalize_seq(sequence)

        # Check cache if available
        if self.cache is not None:
            cached = self.cache.get(self.model_name, self.knob_config, seq_shorts)
            if cached is not None:
                if final and cached.get("final_mean") is not None:
                    return cached["final_mean"]
                return cached["latency"]

        lat = self._compute_mock_latency(seq_shorts)

        if self.cache is not None:
            std = round(lat * 0.015, 3)
            self.cache.set(
                model=self.model_name,
                knob_config=self.knob_config,
                sequence=seq_shorts,
                latency=lat,
                final_mean=lat if final else None,
                final_std=std if final else None,
                metadata={"mode": "SIMULATED"},
            )

        return lat

    def evaluate_detailed(
        self,
        sequence: Sequence[Union[str, Pass]],
        final: bool = False,
    ) -> Tuple[float, float]:
        lat = self.evaluate(sequence, final=final)
        std = round(lat * 0.015, 3)
        return lat, std


class IREEEvaluator:
    """Real evaluator using iree-compile and iree-benchmark-module with disk caching."""

    mode: str = "REAL"

    def __init__(
        self,
        model_name: str,
        model_mlir: str,
        compiler: Optional[IREECompiler] = None,
        benchmark: Optional[BenchmarkRunner] = None,
        cache: Optional[EvaluationCache] = None,
        knob_config: str = "default_knobs",
        vocabulary: Optional[PassVocabulary] = None,
    ):
        self.model_name = model_name
        self.model_mlir = model_mlir
        self.compiler = compiler or IREECompiler()
        self.benchmark = benchmark or BenchmarkRunner()
        self.cache = cache
        self.knob_config = knob_config
        self.vocabulary = vocabulary or get_default_vocabulary()
        self._cached_baseline: Optional[float] = None

    def _normalize_seq(self, sequence: Sequence[Union[str, Pass]]) -> List[str]:
        res = []
        for item in sequence:
            if isinstance(item, Pass):
                res.append(item.short)
            else:
                p = self.vocabulary.get(str(item))
                res.append(p.short if p is not None else str(item))
        return res

    def baseline_o3(self) -> float:
        """Evaluates default -O3 pipeline."""
        if self._cached_baseline is None:
            self._cached_baseline = self.evaluate([], final=False)
        return self._cached_baseline

    def evaluate_detailed(
        self,
        sequence: Sequence[Union[str, Pass]],
        final: bool = False,
    ) -> Tuple[float, float]:
        """Compiles and benchmarks sequence, returning (mean, std)."""
        seq_shorts = self._normalize_seq(sequence)

        # Check cache
        if self.cache is not None:
            cached = self.cache.get(self.model_name, self.knob_config, seq_shorts)
            if cached is not None:
                if final and cached.get("final_mean") is not None:
                    return cached["final_mean"], cached.get("final_std", 0.0)
                if not final:
                    return cached["latency"], 0.0

        with tempfile.NamedTemporaryFile(suffix=".vmfb", delete=False) as tmp:
            vmfb_path = tmp.name

        try:
            # 1. Compile
            ok = self.compiler.compile(
                model_mlir=self.model_mlir,
                sequence=sequence,
                out_vmfb=vmfb_path,
            )
            if not ok:
                if self.cache is not None:
                    self.cache.set(
                        model=self.model_name,
                        knob_config=self.knob_config,
                        sequence=seq_shorts,
                        latency=math.inf,
                        metadata={"mode": "REAL", "status": "COMPILE_FAIL"},
                    )
                return math.inf, 0.0

            # 2. Benchmark
            if final:
                mean_lat, std_lat = self.benchmark.measure_final(vmfb_path)
            else:
                med_lat = self.benchmark.measure_selection(vmfb_path)
                mean_lat, std_lat = med_lat, 0.0

            if self.cache is not None:
                self.cache.set(
                    model=self.model_name,
                    knob_config=self.knob_config,
                    sequence=seq_shorts,
                    latency=mean_lat,
                    final_mean=mean_lat if final else None,
                    final_std=std_lat if final else None,
                    metadata={"mode": "REAL", "status": "OK"},
                )

            return mean_lat, std_lat
        finally:
            if os.path.exists(vmfb_path):
                try:
                    os.unlink(vmfb_path)
                except OSError:
                    pass

    def evaluate(self, sequence: Sequence[Union[str, Pass]], final: bool = False) -> float:
        """Evaluates sequence latency in milliseconds."""
        mean_lat, _ = self.evaluate_detailed(sequence, final=final)
        return mean_lat

