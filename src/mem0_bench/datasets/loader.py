"""Load and save ``BenchmarkDataset`` objects as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from mem0_bench.datasets.schema import BenchmarkDataset


def save_dataset(dataset: BenchmarkDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")


def load_dataset(path: str | Path) -> BenchmarkDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return BenchmarkDataset.model_validate(data)
