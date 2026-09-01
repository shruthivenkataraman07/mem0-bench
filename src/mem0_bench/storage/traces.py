"""Structured per-query execution traces.

Every query the :class:`~mem0_bench.benchmark.runner.BenchmarkRunner`
executes produces one :class:`Trace`, capturing everything needed to
recompute metrics later or debug a single query: retrieval + generation
timing, token accounting, retrieval metrics, the deterministic answer
verdict, and full reproducibility metadata (config snapshot, model,
embedding model, seed, git commit).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from mem0_bench import __version__


def get_git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


class Trace(BaseModel):
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    benchmark_version: str = __version__
    git_commit: Optional[str] = None

    backend: str
    backend_config: Dict = Field(default_factory=dict)
    model: str
    embedding_model: Optional[str] = None
    seed: int

    user_id: str
    query_id: str
    query_type: str
    query: str
    expected_answer: str
    expected_memory_ids: List[str] = Field(default_factory=list)

    prediction: str = ""
    retrieved_memory_ids: List[str] = Field(default_factory=list)

    retrieval_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    retrieval_applicable: bool = True

    exact_match: Optional[bool] = None
    normalized_match: Optional[bool] = None
    structured_match: Optional[bool] = None
    is_correct: Optional[bool] = None
    judge_score: Optional[float] = None
    judge_model: Optional[str] = None

    retrieval_ms: float = 0.0
    prompt_construction_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    input_tokens: int = 0
    retrieved_tokens: int = 0
    output_tokens: int = 0

    history_total_turns: Optional[int] = None
    history_included_turns: Optional[int] = None
    history_dropped_turns: Optional[int] = None
    truncated: bool = False

    success: bool = True
    error: Optional[str] = None


class TraceWriter:
    """Appends traces to a JSONL file as they are produced."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, trace: Trace) -> None:
        self._fh.write(trace.model_dump_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_traces(path: str | Path) -> List[Trace]:
    path = Path(path)
    traces = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                traces.append(Trace.model_validate(json.loads(line)))
    return traces
