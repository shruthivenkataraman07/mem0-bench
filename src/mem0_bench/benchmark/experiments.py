"""Multi-backend and multi-scale experiment orchestration, built on top of
:class:`~mem0_bench.benchmark.runner.BenchmarkRunner`.

``run_compare`` runs several backends against the *same* dataset (same
users, same conversations, same questions, same gold facts, same
answer-generation model where configured identically) so their metrics are
directly comparable (README, "Fairness of comparison").

``run_scale_experiment`` regenerates the dataset at each history length in
`turn_sizes` (same seed, so only length varies) and runs every backend
against it, to find the crossover point between backends (README, "Scale
experiments"). It does not assume any backend wins; it just reports what
happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from mem0_bench.benchmark.runner import BenchmarkRunner
from mem0_bench.config import BenchmarkConfig
from mem0_bench.datasets.generator import generate_dataset
from mem0_bench.datasets.schema import BenchmarkDataset
from mem0_bench.memory.factory import build_backend, resolve_embedding_model_name, resolve_model_name
from mem0_bench.storage.traces import Trace


@dataclass
class BackendRunOutcome:
    backend: str
    traces: List[Trace]
    backend_config: Dict


async def run_single_backend(
    backend_name: str,
    dataset: BenchmarkDataset,
    config: BenchmarkConfig,
    run_id: str,
    backend_config_path: Optional[str] = None,
) -> BackendRunOutcome:
    backend, backend_config = build_backend(backend_name, config, backend_config_path)
    runner = BenchmarkRunner(
        backend=backend,
        run_id=run_id,
        seed=config.seed,
        model=resolve_model_name(config),
        backend_config=backend_config,
        embedding_model=resolve_embedding_model_name(backend_name, config, backend_config_path),
        top_k=backend_config.get("top_k", config.top_k),
        concurrency=config.concurrency,
    )
    traces = await runner.run(dataset)
    return BackendRunOutcome(backend=backend_name, traces=traces, backend_config=backend_config)


async def run_compare(
    backend_names: List[str],
    dataset: BenchmarkDataset,
    config: BenchmarkConfig,
    run_id: str,
) -> Dict[str, BackendRunOutcome]:
    results = {}
    for name in backend_names:
        results[name] = await run_single_backend(name, dataset, config, run_id)
    return results


@dataclass
class ScalePoint:
    num_turns: int
    outcomes: Dict[str, BackendRunOutcome]


async def run_scale_experiment(
    backend_names: List[str],
    turn_sizes: List[int],
    num_users: int,
    config: BenchmarkConfig,
    run_id: str,
) -> List[ScalePoint]:
    points = []
    for num_turns in turn_sizes:
        dataset = generate_dataset(num_users=num_users, num_turns=num_turns, seed=config.seed)
        outcomes = await run_compare(backend_names, dataset, config, run_id=f"{run_id}_t{num_turns}")
        points.append(ScalePoint(num_turns=num_turns, outcomes=outcomes))
    return points
