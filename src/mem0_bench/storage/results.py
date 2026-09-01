"""On-disk results layout for a single benchmark run.

    results/<run_id>/
        config.yaml      snapshot of the configuration used
        metadata.json    run_id, seed, dataset id, backend(s), timestamps
        traces.jsonl      one Trace per executed query
        metrics.csv       flattened per-query metrics for spreadsheet/pandas use
        summary.json      aggregated metrics (see metrics/aggregate.py)
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from mem0_bench.storage.traces import Trace


class RunDirectory:
    def __init__(self, output_dir: str | Path, run_id: str) -> None:
        self.run_id = run_id
        self.path = Path(output_dir) / run_id
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def traces_path(self) -> Path:
        return self.path / "traces.jsonl"

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    @property
    def metadata_path(self) -> Path:
        return self.path / "metadata.json"

    @property
    def summary_path(self) -> Path:
        return self.path / "summary.json"

    @property
    def metrics_csv_path(self) -> Path:
        return self.path / "metrics.csv"

    def write_config_snapshot(self, config: Dict) -> None:
        self.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def write_metadata(self, metadata: Dict) -> None:
        self.metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    def write_summary(self, summary: Dict) -> None:
        self.summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    def write_metrics_csv(self, traces: List[Trace]) -> None:
        if not traces:
            self.metrics_csv_path.write_text("", encoding="utf-8")
            return
        fieldnames = [
            "run_id", "backend", "user_id", "query_id", "query_type", "success",
            "is_correct", "exact_match", "normalized_match", "structured_match",
            "retrieval_applicable", "retrieval_ms", "generation_ms", "total_ms",
            "input_tokens", "retrieved_tokens", "output_tokens",
        ]
        with self.metrics_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for t in traces:
                writer.writerow({k: getattr(t, k) for k in fieldnames})


def new_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def list_runs(output_dir: str | Path) -> List[str]:
    base = Path(output_dir)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def load_summary(output_dir: str | Path, run_id: str) -> Optional[Dict]:
    path = Path(output_dir) / run_id / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
