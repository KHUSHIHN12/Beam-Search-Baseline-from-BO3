"""verify_passes.py - Verifies IREE pass names against installed IREE compiler.

Checks each pass flag in iree-opt and test-compiles it in iree-compile
using --iree-preprocessing-pass-pipeline="builtin.module(...)".
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple


def find_iree_tool(tool_name: str) -> Optional[str]:
    """Finds an IREE executable using PATH or python package discovery."""
    # 1. Check standard PATH
    path = shutil.which(tool_name)
    if path and os.path.exists(path):
        return path

    # 2. On Windows, check for .exe
    if sys.platform == "win32" and not tool_name.endswith(".exe"):
        path = shutil.which(f"{tool_name}.exe")
        if path and os.path.exists(path):
            return path

    # 3. Check inside iree.compiler.tools.binaries
    try:
        import iree.compiler.tools.binaries as b  # type: ignore

        found = b.find_tool(tool_name)
        if found and os.path.exists(found):
            return found
        if sys.platform == "win32" and found and os.path.exists(f"{found}.exe"):
            return f"{found}.exe"
    except Exception:
        pass

    # 4. Check inside iree.runtime for benchmark
    if "benchmark" in tool_name:
        try:
            import iree.runtime as rt  # type: ignore

            exe = rt.benchmark_exe()
            if os.path.exists(exe):
                return exe
            if sys.platform == "win32" and os.path.exists(f"{exe}.exe"):
                return f"{exe}.exe"
        except Exception:
            pass

    return None


CANDIDATE_PASSES = [
    {
        "name": "canonicalize",
        "short": "canon",
        "iree_pass": "canonicalize",
        "max_count": 2,
        "requires": None,
        "description": "Standard MLIR canonicalization patterns and constant folding",
    },
    {
        "name": "convert-conv-to-channels-last",
        "short": "CC",
        "iree_pass": "iree-preprocessing-convert-conv-to-channels-last",
        "max_count": 2,
        "requires": "has_conv",
        "description": "Convert convolutions from NCHW to NHWC channels-last layout",
    },
    {
        "name": "convert-conv2d-to-img2col",
        "short": "I",
        "iree_pass": "iree-global-opt-convert-conv2d-to-img2col",
        "max_count": 2,
        "requires": "has_conv",
        "description": "Lower 2D convolutions to Img2Col matrix multiplication",
    },
    {
        "name": "transpose-convolution-pipeline",
        "short": "T",
        "iree_pass": "iree-preprocessing-transpose-convolution-pipeline",
        "max_count": 2,
        "requires": "has_conv",
        "description": "Pipeline for transposing and canonicalizing convolution weights",
    },
    {
        "name": "linalg-generalize-named-ops",
        "short": "gen",
        "iree_pass": "linalg-generalize-named-ops",
        "max_count": 2,
        "requires": None,
        "description": "Generalize named Linalg ops to generic tensor operations",
    },
    {
        "name": "convert-1x1-filter-conv2d-to-matmul",
        "short": "1x1",
        "iree_pass": "iree-preprocessing-convert-1x1-filter-conv2d-to-matmul",
        "max_count": 2,
        "requires": "has_conv",
        "description": "Convert 1x1 filter convolutions directly into matmul ops",
    },
    {
        "name": "pad-linalg-ops",
        "short": "pad",
        "iree_pass": "iree-preprocessing-pad-linalg-ops",
        "max_count": 2,
        "requires": None,
        "description": "Pad linalg operations to hardware-friendly alignment multiples",
    },
    {
        "name": "pad-to-intrinsics",
        "short": "intrinsics",
        "iree_pass": "iree-preprocessing-pad-to-intrinsics",
        "max_count": 2,
        "requires": None,
        "description": "Pad tensors to match target CPU/SIMD intrinsic shapes",
    },
    {
        "name": "transpose-matmul",
        "short": "trans-mm",
        "iree_pass": "iree-preprocessing-transpose-matmul-pass",
        "max_count": 2,
        "requires": None,
        "description": "Transpose matrix multiplication operands for cache efficiency",
    },
]


def test_pass_compilation(compile_bin: str, pass_name: str) -> Tuple[bool, str]:
    """Tests if a pass works inside --iree-preprocessing-pass-pipeline."""
    minimal_mlir = b"module { func.func @main() { return } }"
    with tempfile.NamedTemporaryFile(suffix=".vmfb", delete=False) as tmp:
        out_vmfb = tmp.name

    try:
        pipeline = f"builtin.module({pass_name})"
        cmd = [
            compile_bin,
            "--iree-hal-target-backends=llvm-cpu",
            f"--iree-preprocessing-pass-pipeline={pipeline}",
            "-o",
            out_vmfb,
            "-",
        ]
        res = subprocess.run(
            cmd,
            input=minimal_mlir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        if res.returncode == 0 and os.path.exists(out_vmfb) and os.path.getsize(out_vmfb) > 0:
            return True, "Verified (compiles successfully)"
        err_msg = res.stderr.decode("utf-8", errors="replace").strip().splitlines()
        first_err = err_msg[0] if err_msg else "Exit code != 0"
        return False, first_err[:80]
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(out_vmfb):
            try:
                os.unlink(out_vmfb)
            except OSError:
                pass


def verify_all_passes(verbose: bool = True) -> List[Dict]:
    """Verifies all candidate passes and returns list of verified pass definitions."""
    compile_bin = find_iree_tool("iree-compile")
    opt_bin = find_iree_tool("iree-opt")
    bench_bin = find_iree_tool("iree-benchmark-module")

    print("=" * 70)
    print(" BeamPass: IREE Pass Verification Tool")
    print("=" * 70)
    print(f"iree-compile:          {compile_bin or 'NOT FOUND'}")
    print(f"iree-opt:              {opt_bin or 'NOT FOUND'}")
    print(f"iree-benchmark-module: {bench_bin or 'NOT FOUND'}")
    print("-" * 70)

    if not compile_bin:
        print("[!] Warning: iree-compile binary not found.")
        print("    Using default candidate vocabulary for Mock / Simulation mode.")
        return CANDIDATE_PASSES

    verified = []
    print(f"{'Short':<10} {'Pass Flag':<52} {'Status'}")
    print("-" * 70)

    for p in CANDIDATE_PASSES:
        pass_flag = p["iree_pass"]
        ok, msg = test_pass_compilation(compile_bin, pass_flag)
        status_str = "[OK]" if ok else f"[FAIL] {msg}"
        print(f"{p['short']:<10} {pass_flag:<52} {status_str}")
        if ok:
            verified.append(p)

    print("-" * 70)
    print(f"Verified {len(verified)} of {len(CANDIDATE_PASSES)} passes successfully.")
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify IREE pass names against local installation.")
    parser.add_argument("--save-config", action="store_true", help="Save verified passes to configs/passes.yaml")
    args = parser.parse_args()

    passes = verify_all_passes()
    if args.save_config:
        import yaml

        out_path = os.path.join(os.path.dirname(__file__), "..", "configs", "passes.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump({"passes": passes}, f, sort_keys=False)
        print(f"Saved configuration to: {out_path}")

