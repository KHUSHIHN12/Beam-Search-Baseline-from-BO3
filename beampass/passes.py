"""passes.py - Pass dataclass and vocabulary loader for BeamPass."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml


@dataclass(frozen=True)
class Pass:
    """Represents a compiler optimization pass."""

    name: str
    short: str
    iree_pass: str
    max_count: int = 2
    requires: Optional[str] = None
    description: str = ""

    def __repr__(self) -> str:
        return f"Pass({self.short}: {self.name})"


# Virtual action denoting sequence termination
EVAL_ACTION = "EVAL"


class PassVocabulary:
    """Manages the available pass pool and index mappings."""

    def __init__(self, passes: List[Pass]):
        self.passes = list(passes)
        self._by_name: Dict[str, Pass] = {p.name: p for p in self.passes}
        self._by_short: Dict[str, Pass] = {p.short: p for p in self.passes}
        self._by_iree: Dict[str, Pass] = {p.iree_pass: p for p in self.passes}

    @classmethod
    def from_yaml(cls, config_path: str) -> PassVocabulary:
        """Loads pass vocabulary from a YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Pass config not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        raw_passes = data.get("passes", [])
        passes = [
            Pass(
                name=p["name"],
                short=p.get("short", p["name"][:4]),
                iree_pass=p.get("iree_pass", p["name"]),
                max_count=p.get("max_count", 2),
                requires=p.get("requires"),
                description=p.get("description", ""),
            )
            for p in raw_passes
        ]
        return cls(passes)

    def get(self, identifier: str) -> Optional[Pass]:
        """Looks up a pass by short code, name, or iree_pass."""
        if identifier in self._by_short:
            return self._by_short[identifier]
        if identifier in self._by_name:
            return self._by_name[identifier]
        if identifier in self._by_iree:
            return self._by_iree[identifier]
        return None

    def __len__(self) -> int:
        return len(self.passes)

    def __iter__(self):
        return iter(self.passes)

    def __getitem__(self, idx: int) -> Pass:
        return self.passes[idx]


def get_default_vocabulary(config_path: Optional[str] = None) -> PassVocabulary:
    """Convenience function to load vocabulary from standard config location."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "configs", "passes.yaml")
    return PassVocabulary.from_yaml(config_path)

