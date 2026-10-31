"""beam_search.py - Core beam-search compiler pass-order optimizer."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from beampass.evaluator import Evaluator
from beampass.legality import IRInfo, legal_actions
from beampass.passes import Pass, PassVocabulary, get_default_vocabulary


@dataclass
class BeamNode:
    """Represents an expanded node in the beam search tree."""

    depth: int
    sequence: List[str]
    latency: float
    kept_in_beam: bool = False
    parent_sequence: List[str] = field(default_factory=list)
    compile_idx: int = 0
    speedup_vs_baseline: float = 1.0

    @property
    def sequence_str(self) -> str:
        return " -> ".join(self.sequence) if self.sequence else "(empty: -O3)"


@dataclass
class BeamSearchResult:
    """Complete summary of a beam search execution."""

    model_name: str
    mode: str
    beam_width: int
    max_depth: int
    baseline_latency: float
    best_sequence: List[str]
    best_latency: float
    speedup: float
    final_latency: float
    final_std: float
    total_compiles: int
    nodes: List[BeamNode] = field(default_factory=list)
    depth_best: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        res = asdict(self)
        res["nodes"] = [asdict(n) for n in self.nodes]
        return res


class BeamSearch:
    """Executes beam search over preprocessing pass sequences."""

    def __init__(
        self,
        evaluator: Evaluator,
        vocabulary: Optional[PassVocabulary] = None,
        ir_info: Optional[IRInfo] = None,
        beam_width: int = 2,
        max_depth: int = 5,
        patience: int = 2,
        max_compiles: int = 60,
        allow_consecutive_repeats: bool = False,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ):
        self.evaluator = evaluator
        self.vocabulary = vocabulary or get_default_vocabulary()
        self.ir_info = ir_info or IRInfo(has_conv=True, has_matmul=True)
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.patience = patience
        self.max_compiles = max_compiles
        self.allow_consecutive_repeats = allow_consecutive_repeats
        self.progress_callback = progress_callback

    def _notify(self, event_type: str, data: Dict) -> None:
        if self.progress_callback:
            payload = {"event": event_type, **data}
            self.progress_callback(payload)

    def run(self) -> BeamSearchResult:
        """Executes beam search optimization."""
        # 1. Establish baseline (-O3 default pipeline)
        baseline_lat = self.evaluator.baseline_o3()
        total_compiles = 1

        self._notify(
            "start",
            {
                "baseline_latency": baseline_lat,
                "beam_width": self.beam_width,
                "max_depth": self.max_depth,
                "model": self.evaluator.model_name,
                "mode": self.evaluator.mode,
            },
        )

        all_nodes: List[BeamNode] = []
        depth_best: Dict[int, float] = {0: baseline_lat}

        # Root node
        root_node = BeamNode(
            depth=0,
            sequence=[],
            latency=baseline_lat,
            kept_in_beam=True,
            parent_sequence=[],
            compile_idx=1,
            speedup_vs_baseline=1.0,
        )
        all_nodes.append(root_node)

        # Beam state: list of (sequence_of_pass_objects, latency)
        beam: List[Tuple[List[Pass], float]] = [([], baseline_lat)]
        best_seq: List[str] = []
        best_lat: float = baseline_lat

        no_improvement_depths = 0

        for depth in range(1, self.max_depth + 1):
            if total_compiles >= self.max_compiles:
                break

            candidates: List[Tuple[List[Pass], float, BeamNode]] = []

            for seq, _ in beam:
                legal = legal_actions(
                    sequence=seq,
                    vocabulary=self.vocabulary,
                    ir_info=self.ir_info,
                    allow_consecutive_repeats=self.allow_consecutive_repeats,
                )

                for p in legal:
                    if total_compiles >= self.max_compiles:
                        break

                    new_seq = seq + [p]
                    new_seq_shorts = [x.short for x in new_seq]
                    parent_shorts = [x.short for x in seq]

                    total_compiles += 1
                    self._notify(
                        "evaluating",
                        {
                            "depth": depth,
                            "sequence": new_seq_shorts,
                            "parent": parent_shorts,
                            "compile_idx": total_compiles,
                            "max_compiles": self.max_compiles,
                        },
                    )

                    lat = self.evaluator.evaluate(new_seq)

                    node = BeamNode(
                        depth=depth,
                        sequence=new_seq_shorts,
                        latency=lat,
                        kept_in_beam=False,
                        parent_sequence=parent_shorts,
                        compile_idx=total_compiles,
                        speedup_vs_baseline=round(baseline_lat / lat, 3) if lat > 0 and not math.isinf(lat) else 0.0,
                    )
                    all_nodes.append(node)
                    self._notify("node_evaluated", {"node": node})

                    if lat < math.inf:
                        candidates.append((new_seq, lat, node))

            if not candidates:
                break

            # Sort candidates by latency ascending
            candidates.sort(key=lambda x: x[1])

            # Select top B candidates for new beam
            new_beam: List[Tuple[List[Pass], float]] = []
            depth_best_node = candidates[0][2]

            for i in range(min(self.beam_width, len(candidates))):
                top_seq, top_lat, top_node = candidates[i]
                top_node.kept_in_beam = True
                new_beam.append((top_seq, top_lat))

            beam = new_beam
            depth_best[depth] = beam[0][1]

            # Update overall best
            current_depth_best_seq = [x.short for x in beam[0][0]]
            current_depth_best_lat = beam[0][1]

            if current_depth_best_lat < best_lat:
                best_lat = current_depth_best_lat
                best_seq = current_depth_best_seq
                no_improvement_depths = 0
            else:
                no_improvement_depths += 1

            self._notify(
                "depth_finished",
                {
                    "depth": depth,
                    "depth_best_latency": current_depth_best_lat,
                    "depth_best_sequence": current_depth_best_seq,
                    "overall_best_latency": best_lat,
                    "overall_best_sequence": best_seq,
                },
            )

            # Early stopping check
            if no_improvement_depths >= self.patience:
                break

        # Final high-fidelity evaluation (mean ± std)
        final_mean, final_std = self.evaluator.evaluate_detailed(best_seq, final=True)
        if math.isinf(final_mean):
            final_mean = best_lat
            final_std = 0.0

        overall_speedup = round(baseline_lat / final_mean, 3) if final_mean > 0 else 1.0

        result = BeamSearchResult(
            model_name=self.evaluator.model_name,
            mode=self.evaluator.mode,
            beam_width=self.beam_width,
            max_depth=self.max_depth,
            baseline_latency=baseline_lat,
            best_sequence=best_seq,
            best_latency=best_lat,
            speedup=overall_speedup,
            final_latency=final_mean,
            final_std=final_std,
            total_compiles=total_compiles,
            nodes=all_nodes,
            depth_best=depth_best,
        )

        self._notify("finished", {"result": result})
        return result

