"""compiler.py - IREE compiler wrapper and pipeline string builder for BeamPass."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional, Sequence, Union

from beampass.passes import Pass, PassVocabulary, get_default_vocabulary
from scripts.verify_passes import find_iree_tool


class IREECompiler:
    """Wraps iree-compile to execute custom preprocessing pipelines."""

    def __init__(
        self,
        compiler_path: Optional[str] = None,
        vocabulary: Optional[PassVocabulary] = None,
        target_backend: str = "llvm-cpu",
        target_cpu: str = "host",
        default_opt_level: str = "O3",
        default_timeout: float = 300.0,
    ):
        if compiler_path is None:
            compiler_path = find_iree_tool("iree-compile")
        self.compiler_path = compiler_path
        self.vocabulary = vocabulary or get_default_vocabulary()
        self.target_backend = target_backend
        self.target_cpu = target_cpu
        self.default_opt_level = default_opt_level
        self.default_timeout = default_timeout

    def is_available(self) -> bool:
        """Returns True if the iree-compile binary is located on this system."""
        return self.compiler_path is not None and os.path.exists(self.compiler_path)

    def build_pipeline_string(self, sequence: Sequence[Union[str, Pass]]) -> Optional[str]:
        """Builds a pass pipeline string, e.g. builtin.module(canonicalize,CC)."""
        if not sequence:
            return None

        pass_names: List[str] = []
        for item in sequence:
            if isinstance(item, Pass):
                pass_names.append(item.iree_pass)
            else:
                p = self.vocabulary.get(str(item))
                pass_names.append(p.iree_pass if p is not None else str(item))

        inner = ",".join(pass_names)
        return f"builtin.module({inner})"

    def compile(
        self,
        model_mlir: str,
        sequence: Sequence[Union[str, Pass]],
        out_vmfb: str,
        opt_level: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Compiles an MLIR module with the given pass sequence to a VMFB artifact.

        Returns True on success, False if compilation fails or times out.
        """
        if not self.is_available():
            return False

        if not os.path.exists(model_mlir):
            return False

        opt_level = opt_level or self.default_opt_level
        timeout = timeout or self.default_timeout

        cmd = [
            self.compiler_path,  # type: ignore
            f"--iree-hal-target-backends={self.target_backend}",
            f"--iree-llvmcpu-target-cpu={self.target_cpu}",
            f"--iree-opt-level={opt_level}",
            "-o",
            out_vmfb,
        ]

        pipeline_str = self.build_pipeline_string(sequence)
        if pipeline_str:
            cmd.append(f"--iree-preprocessing-pass-pipeline={pipeline_str}")

        cmd.append(model_mlir)

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(out_vmfb)), exist_ok=True)
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            success = (
                res.returncode == 0
                and os.path.exists(out_vmfb)
                and os.path.getsize(out_vmfb) > 0
            )
            return success
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
            return False

    def compile_baseline(
        self,
        model_mlir: str,
        out_vmfb: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """Compiles the default -O3 baseline without custom preprocessing."""
        return self.compile(
            model_mlir=model_mlir,
            sequence=[],
            out_vmfb=out_vmfb,
            opt_level="O3",
            timeout=timeout,
        )

