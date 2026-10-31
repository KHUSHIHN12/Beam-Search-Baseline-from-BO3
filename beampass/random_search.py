"""random_search.py - Random baseline compiler pass search for fair comparison."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from beampass.evaluator import Evaluator
from beampass.legality import IRInfo, legal_actions
from beampass.passes import Pass, PassVocabulary, get_default_vocabulary


@dataclass
class RandomSearchResult:
    """Summary of random search baseline exploration."""

    model_name: str
    mode: str
    num_samples: int
    baseline_latency: float
    best_sequence: List[str]
    best_latency: float
    speedup: float
    final_latency: float
    final_std: float
    total_compiles: int
    history: List[Tuple[List[str], float]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class RandomSearch:
    """Samples random legal pass sequences under the same compile budget."""

    def __init__(
        self,
        evaluator: Evaluator,
        vocabulary: Optional[PassVocabulary] = None,
        ir_info: Optional[IRInfo] = None,
        max_depth: int = 5,
        num_samples: int = 20,
        seed: int = 42,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ):
        self.evaluator = evaluator
        self.vocabulary = vocabulary or get_default_vocabulary()
        self.ir_info = ir_info or IRInfo(has_conv=True, has_matmul=True)
        self.max_depth = max_depth
        self.num_samples = num_samples
        self.seed = seed
        self.progress_callback = progress_callback
        self._rng = random.Random(seed)

    def _notify(self, event_type: str, data: Dict) -> None:
        if self.progress_callback:
            payload = {"event": event_type, **data}
            self.progress_callback(payload)

    def _sample_one_sequence(self) -> List[Pass]:
        """Constructs one legal random pass sequence of variable length."""
        target_len = self._rng.randint(1, self.max_depth)
        seq: List[Pass] = []

        for _ in range(target_len):
            legal = legal_actions(
                sequence=seq,
                vocabulary=self.vocabulary,
                ir_info=self.ir_info,
            )
            if not legal:
                break
            chosen = self._rng.choice(legal)
            seq.append(chosen)

        return seq

    def run(self) -> RandomSearchResult:
        """Executes random rollout search."""
        baseline_lat = self.evaluator.baseline_o3()
        best_lat = baseline_lat
        best_seq: List[str] = []
        history: List[Tuple[List[str], float]] = []

        self._notify(
            "start",
            {
                "baseline_latency": baseline_lat,
                "num_samples": self.num_samples,
                "model": self.evaluator.model_name,
            },
        )

        for i in range(1, self.num_samples + 1):
            seq = self._sample_one_sequence()
            seq_shorts = [p.short for p in seq]

            self._notify(
                "evaluating",
                {
                    "sample_idx": i,
                    "total_samples": self.num_samples,
                    "sequence": seq_shorts,
                },
            )

            lat = self.evaluator.evaluate(seq)
            history.append((seq_shorts, lat))

            if lat < best_lat:
                best_lat = lat
                best_seq = seq_shorts

            self._notify(
                "sample_evaluated",
                {
                    "sample_idx": i,
                    "sequence": seq_shorts,
                    "latency": lat,
                    "best_latency": best_lat,
                },
            )

        final_mean, final_std = self.evaluator.evaluate_detailed(best_seq, final=True)
        if math.isinf(final_mean):
            final_mean = best_lat
            final_std = 0.0

        speedup = round(baseline_lat / final_mean, 3) if final_mean > 0 else 1.0

        result = RandomSearchResult(
            model_name=self.evaluator.model_name,
            mode=self.evaluator.mode,
            num_samples=self.num_samples,
            baseline_latency=baseline_lat,
            best_sequence=best_seq,
            best_latency=best_lat,
            speedup=speedup,
            final_latency=final_mean,
            final_std=final_std,
            total_compiles=self.num_samples + 1,
            history=history,
        )

        self._notify("finished", {"result": result})
        return result

