"""legality.py - Legality masking and IR feature inspection for BeamPass."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

from beampass.passes import Pass, PassVocabulary, get_default_vocabulary


@dataclass
class IRInfo:
    """Structural properties of an MLIR / ONNX model."""

    has_conv: bool = True
    has_matmul: bool = True
    op_counts: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_mlir(cls, mlir_source: str) -> IRInfo:
        """Inspects an MLIR file or string for structural operations."""
        if os.path.exists(mlir_source):
            with open(mlir_source, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        else:
            content = mlir_source

        lower_content = content.lower()
        has_conv = (
            "linalg.conv" in lower_content
            or "tosa.conv" in lower_content
            or "torch.aten.convolution" in lower_content
            or "conv2d" in lower_content
        )
        has_matmul = (
            "linalg.matmul" in lower_content
            or "linalg.batch_matmul" in lower_content
            or "tosa.matmul" in lower_content
            or "matmul" in lower_content
        )

        return cls(has_conv=has_conv, has_matmul=has_matmul)

    @classmethod
    def from_onnx(cls, onnx_path: str) -> IRInfo:
        """Inspects an ONNX file for operators."""
        try:
            import onnx  # type: ignore

            model = onnx.load(onnx_path, load_external_data=False)
            op_types = [node.op_type for node in model.graph.node]
            has_conv = any("conv" in op.lower() for op in op_types)
            has_matmul = any(op in ("MatMul", "Gemm") for op in op_types)
            return cls(has_conv=has_conv, has_matmul=has_matmul)
        except Exception:
            # Fallback if onnx parsing fails
            return cls(has_conv=True, has_matmul=True)


def legal_actions(
    sequence: Sequence[Union[str, Pass]],
    vocabulary: Optional[PassVocabulary] = None,
    ir_info: Optional[IRInfo] = None,
    allow_consecutive_repeats: bool = False,
) -> List[Pass]:
    """Returns the deterministic list of legal passes that can be appended to sequence.

    Constraints enforced:
    1. Per-pass count limit: mask a pass if applied >= max_count times.
    2. Structural preconditions: mask passes requiring 'has_conv' if model lacks convolutions.
    3. Immediate repetition: optionally mask immediately repeating the same pass.
    """
    if vocabulary is None:
        vocabulary = get_default_vocabulary()

    if ir_info is None:
        ir_info = IRInfo(has_conv=True, has_matmul=True)

    # Normalize sequence to list of Pass objects
    seq_passes: List[Pass] = []
    for item in sequence:
        if isinstance(item, Pass):
            seq_passes.append(item)
        else:
            p = vocabulary.get(str(item))
            if p is not None:
                seq_passes.append(p)

    last_pass = seq_passes[-1] if seq_passes else None

    # Count occurrences of each pass in the current sequence
    pass_counts: Dict[str, int] = {}
    for p in seq_passes:
        pass_counts[p.name] = pass_counts.get(p.name, 0) + 1

    legal: List[Pass] = []
    for cand in vocabulary:
        # 1. Per-pass count limit
        if pass_counts.get(cand.name, 0) >= cand.max_count:
            continue

        # 2. Structural preconditions
        if cand.requires == "has_conv" and not ir_info.has_conv:
            continue
        if cand.requires == "has_matmul" and not ir_info.has_matmul:
            continue

        # 3. Consecutive repeat check
        if not allow_consecutive_repeats and last_pass is not None and last_pass.name == cand.name:
            continue

        legal.append(cand)

    return legal

