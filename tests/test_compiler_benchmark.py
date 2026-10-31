"""test_compiler_benchmark.py - Compiler wrapper, benchmark parser, and real IREE smoke tests."""

import math
import os
import pytest
from beampass.benchmark import BenchmarkRunner, parse_benchmark_json, parse_benchmark_text
from beampass.compiler import IREECompiler
from beampass.passes import get_default_vocabulary


def test_compiler_pipeline_builder():
    """Compiler must format pass pipelines correctly as builtin.module(...)."""
    compiler = IREECompiler()
    p_str = compiler.build_pipeline_string(["canon", "CC", "I"])
    assert p_str == "builtin.module(canonicalize,iree-preprocessing-convert-conv-to-channels-last,iree-global-opt-convert-conv2d-to-img2col)"

    # Empty sequence gives None
    assert compiler.build_pipeline_string([]) is None


def test_parse_benchmark_json_median():
    """Parser must extract real_time_median correctly."""
    sample_json = """
    {
      "benchmarks": [
        {"name": "BM_main/process_time/real_time", "real_time": 12.5, "time_unit": "ms"},
        {"name": "BM_main/process_time/real_time", "real_time": 13.0, "time_unit": "ms"},
        {"name": "BM_main/process_time/real_time", "real_time": 12.0, "time_unit": "ms"},
        {"name": "BM_main/process_time/real_time_median", "real_time": 12.5, "time_unit": "ms"}
      ]
    }
    """
    med = parse_benchmark_json(sample_json)
    assert med == 12.5


def test_parse_benchmark_unit_conversion():
    """Parser must convert ns and us to milliseconds."""
    sample_json_ns = """
    {
      "benchmarks": [
        {"name": "BM_main/real_time_median", "real_time": 5000000.0, "time_unit": "ns"}
      ]
    }
    """
    assert parse_benchmark_json(sample_json_ns) == 5.0

    sample_json_us = """
    {
      "benchmarks": [
        {"name": "BM_main/real_time_median", "real_time": 4000.0, "time_unit": "us"}
      ]
    }
    """
    assert parse_benchmark_json(sample_json_us) == 4.0


def test_iree_real_smoke_test(tmp_path):
    """Smoke test: compiles minimal MLIR module with real IREE if installed."""
    compiler = IREECompiler()
    if not compiler.is_available():
        pytest.skip("iree-compile not installed on host machine")

    minimal_mlir = str(tmp_path / "tiny.mlir")
    with open(minimal_mlir, "w") as f:
        f.write("module { func.func @main() { return } }")

    out_vmfb = str(tmp_path / "out.vmfb")
    ok = compiler.compile(
        model_mlir=minimal_mlir,
        sequence=["canon"],
        out_vmfb=out_vmfb,
        timeout=15,
    )
    assert ok
    assert os.path.exists(out_vmfb)
    assert os.path.getsize(out_vmfb) > 0

