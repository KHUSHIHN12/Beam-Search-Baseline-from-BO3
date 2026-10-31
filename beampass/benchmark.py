"""benchmark.py - Latency measurement protocol implementing paper Section VII-D."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from scripts.verify_passes import find_iree_tool


def _parse_time_in_ms(val: float, unit: str) -> float:
    """Converts a time value with unit to milliseconds."""
    unit_lower = unit.lower()
    if unit_lower in ("ns", "nanoseconds"):
        return val / 1_000_000.0
    elif unit_lower in ("us", "microseconds", "µs"):
        return val / 1_000.0
    elif unit_lower in ("ms", "milliseconds"):
        return val
    elif unit_lower in ("s", "seconds"):
        return val * 1_000.0
    return val


def parse_benchmark_json(stdout_text: str) -> Optional[float]:
    """Parses Google Benchmark JSON output to extract real_time_median in ms."""
    try:
        data = json.loads(stdout_text)
        benchmarks = data.get("benchmarks", [])
        # First check for an explicit median entry
        for b in benchmarks:
            name = b.get("name", "")
            if "real_time_median" in name or name.endswith("_median"):
                raw_time = b.get("real_time")
                unit = b.get("time_unit", "ms")
                if raw_time is not None:
                    return _parse_time_in_ms(float(raw_time), unit)

        # Otherwise collect all repetition runs and take the median
        real_times = []
        for b in benchmarks:
            name = b.get("name", "")
            if not any(agg in name for agg in ("_mean", "_median", "_stddev", "_cv")):
                raw_time = b.get("real_time")
                unit = b.get("time_unit", "ms")
                if raw_time is not None:
                    real_times.append(_parse_time_in_ms(float(raw_time), unit))

        if real_times:
            return float(statistics.median(real_times))
    except Exception:
        pass
    return None


def parse_benchmark_text(stdout_text: str) -> Optional[float]:
    """Fallback text parser for benchmark output lines."""
    for line in stdout_text.splitlines():
        if "real_time_median" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "ms" and i > 0:
                    try:
                        return float(parts[i - 1])
                    except ValueError:
                        pass
                elif p.endswith("ms"):
                    try:
                        return float(p[:-2])
                    except ValueError:
                        pass
    return None


class BenchmarkRunner:
    """Executes iree-benchmark-module with affinity and repetition controls."""

    def __init__(
        self,
        benchmark_path: Optional[str] = None,
        device: str = "local-sync",
        pinned_core: Optional[int] = 0,
        default_timeout: float = 120.0,
    ):
        if benchmark_path is None:
            benchmark_path = find_iree_tool("iree-benchmark-module")
        self.benchmark_path = benchmark_path
        self.device = device
        self.pinned_core = pinned_core
        self.default_timeout = default_timeout

    def is_available(self) -> bool:
        """Returns True if the benchmark binary is available."""
        return self.benchmark_path is not None and os.path.exists(self.benchmark_path)

    def _build_cmd(self, vmfb_path: str, repetitions: int) -> List[str]:
        cmd = []
        # On Linux, use taskset if a core is specified
        if platform.system() == "Linux" and self.pinned_core is not None:
            cmd.extend(["taskset", "-c", str(self.pinned_core)])

        cmd.extend(
            [
                self.benchmark_path,  # type: ignore
                f"--module={vmfb_path}",
                f"--device={self.device}",
                f"--benchmark_repetitions={repetitions}",
                "--benchmark_format=json",
            ]
        )
        return cmd

    def run_repetition_benchmark(
        self,
        vmfb_path: str,
        repetitions: int = 10,
        timeout: Optional[float] = None,
    ) -> float:
        """Runs a single invocation with N repetitions and returns median latency in ms."""
        if not self.is_available() or not os.path.exists(vmfb_path):
            return math.inf

        timeout = timeout or self.default_timeout
        cmd = self._build_cmd(vmfb_path, repetitions)

        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            if res.returncode != 0:
                return math.inf

            out_str = res.stdout.decode("utf-8", errors="replace")
            parsed = parse_benchmark_json(out_str)
            if parsed is not None:
                return parsed

            parsed_text = parse_benchmark_text(out_str)
            if parsed_text is not None:
                return parsed_text

            return math.inf
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return math.inf

    def measure_selection(
        self,
        vmfb_path: str,
        repetitions: int = 10,
        timeout: Optional[float] = None,
    ) -> float:
        """Selection phase protocol: 10 repetitions per candidate, take the median."""
        return self.run_repetition_benchmark(
            vmfb_path=vmfb_path,
            repetitions=repetitions,
            timeout=timeout,
        )

    def measure_final(
        self,
        vmfb_path: str,
        invocations: int = 10,
        repetitions: int = 5,
        timeout: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Final evaluation protocol (Paper VII-D):

        10 independent invocations, each with 5 repetitions.
        Take the median of each invocation, then report mean ± std of the 10 medians.
        """
        medians: List[float] = []
        for _ in range(invocations):
            med = self.run_repetition_benchmark(
                vmfb_path=vmfb_path,
                repetitions=repetitions,
                timeout=timeout,
            )
            if math.isinf(med):
                return math.inf, 0.0
            medians.append(med)

        if not medians:
            return math.inf, 0.0

        mean_val = float(statistics.mean(medians))
        std_val = float(statistics.stdev(medians)) if len(medians) > 1 else 0.0
        return mean_val, std_val

