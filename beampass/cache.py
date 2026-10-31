"""cache.py - Disk-backed compilation and benchmark cache for BeamPass."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Sequence, Tuple


class EvaluationCache:
    """Caches sequence evaluation latencies keyed by (model, knob_config, sequence).

    Persists to disk in JSON format so re-running evaluations or demo presentations
    is instant.
    """

    def __init__(self, cache_file: Optional[str] = None):
        if cache_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_file = os.path.join(base_dir, "results", "cache.json")
        self.cache_file = cache_file
        self._data: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.load()

    def _make_key(
        self,
        model: str,
        knob_config: str,
        sequence: Sequence[str],
    ) -> str:
        seq_str = ",".join(str(s) for s in sequence)
        return f"{model}::{knob_config}::{seq_str}"

    def get(
        self,
        model: str,
        knob_config: str,
        sequence: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        """Retrieves cached result if present."""
        key = self._make_key(model, knob_config, sequence)
        if key in self._data:
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def set(
        self,
        model: str,
        knob_config: str,
        sequence: Sequence[str],
        latency: float,
        final_mean: Optional[float] = None,
        final_std: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Stores evaluation result in cache and persists to disk."""
        key = self._make_key(model, knob_config, sequence)
        record = {
            "model": model,
            "knob_config": knob_config,
            "sequence": [str(s) for s in sequence],
            "latency": latency,
            "final_mean": final_mean,
            "final_std": final_std,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._data[key] = record
        self.save()

    def contains(
        self,
        model: str,
        knob_config: str,
        sequence: Sequence[str],
    ) -> bool:
        """Checks if a sequence is already in cache without incrementing counters."""
        key = self._make_key(model, knob_config, sequence)
        return key in self._data

    def load(self) -> None:
        """Loads cached entries from disk if file exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def save(self) -> None:
        """Persists cache data to disk atomically."""
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
        temp_file = f"{self.cache_file}.tmp"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            if os.path.exists(self.cache_file):
                os.replace(temp_file, self.cache_file)
            else:
                os.rename(temp_file, self.cache_file)
        except Exception:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

    def clear(self) -> None:
        """Clears cache memory and deletes file."""
        self._data.clear()
        self.hits = 0
        self.misses = 0
        if os.path.exists(self.cache_file):
            try:
                os.unlink(self.cache_file)
            except OSError:
                pass

    def __len__(self) -> int:
        return len(self._data)

