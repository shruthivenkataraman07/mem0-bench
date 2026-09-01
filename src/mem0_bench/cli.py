"""mem0-bench command-line interface."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from mem0_bench.attacks.evaluator import aggregate_attack_trials, run_attack_trial
from mem0_bench.attacks.poisoning import POISON_ATTACKS
from mem0_bench.benchmark.experiments import run_compare, run_scale_experiment
from mem0_bench.config import BenchmarkConfig, load_env
from mem0_bench.datasets.generator import generate_dataset
from mem0_bench.datasets.loader import load_dataset, save_dataset
from mem0_bench.datasets.validation import validate_dataset
from mem0_bench.metrics.aggregate import summarize_run
from mem0_bench.metrics.tokens import load_pricing_table
from mem0_bench.storage.results import RunDirectory, load_summary, new_run_id
from mem0_bench.storage.traces import Trace

app = typer.Typer(
    name="mem0-bench",
    help="Benchmark long-term agent memory architectures: Mem0 vs conventional RAG vs full-context.",
    add_completion=False,
)
console = Console()

ALL_BACKENDS = ["mem0", "rag", "full-context"]


def _split_csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _load_config(config_path: Optional[str], seed: Optional[int], top_k: Optional[int]) -> BenchmarkConfig:
    load_env()
    config = BenchmarkConfig.from_yaml(config_path) if config_path else BenchmarkConfig.default()
    if seed is not None:
        config.seed = seed
    if top_k is not None:
        config.top_k = top_k
    return config


@app.command()
def generate(
    users: int = typer.Option(10, help="Number of simulated users to generate."),
    turns: int = typer.Option(50, help="Number of user turns per simulated user."),
    seed: int = typer.Option(42, help="Random seed. Same seed + args -> byte-identical dataset."),
    output: Optional[str] = typer.Option(None, help="Output JSON path. Defaults under datasets/conversations/."),
) -> None:
    """Deterministically generate a synthetic multi-session benchmark dataset."""
    dataset = generate_dataset(num_users=users, num_turns=turns, seed=seed)
    result = validate_dataset(dataset)
    if not result.is_valid:
        console.print("[bold red]Generated dataset failed validation:[/bold red]")
        for err in result.errors:
            console.print(f"  - {err}")
        raise typer.Exit(code=1)

    out_path = Path(output) if output else Path("datasets/conversations") / f"{dataset.dataset_id}.json"
    save_dataset(dataset, out_path)
    console.print(f"[green]Generated[/green] {dataset.num_users} users, "
                  f"{dataset.num_messages} messages, {dataset.num_questions} questions")
    console.print(f"Saved to [bold]{out_path}[/bold]")


@app.command()
def run(
    backend: str = typer.Option(..., help="Backend to run: mem0, rag, or full-context."),
    dataset: str = typer.Option(..., help="Path to a dataset JSON file (see `generate`)."),
    top_k: Optional[int] = typer.Option(None, help="Override retrieval top-k."),
    seed: Optional[int] = typer.Option(None, help="Override the benchmark seed."),
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to benchmark.yaml override."),
    output_dir: Optional[str] = typer.Option(None, help="Where to write results/<run_id>/."),
) -> None:
    """Run one backend against a dataset and write a results/<run_id>/ directory."""
    config = _load_config(config_path, seed, top_k)
    if output_dir:
        config.output_dir = output_dir

    ds = load_dataset(dataset)
    validation = validate_dataset(ds)
    if not validation.is_valid:
        console.print("[bold red]Dataset failed validation:[/bold red]")
        for err in validation.errors:
            console.print(f"  - {err}")
        raise typer.Exit(code=1)

    run_id = new_run_id(prefix=backend.replace("-", "_"))
    console.print(f"Running backend=[bold]{backend}[/bold] on dataset=[bold]{dataset}[/bold] (run_id={run_id})")

    try:
        outcomes = asyncio.run(run_compare([backend], ds, config, run_id))
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]Run failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    outcome = outcomes[backend]
    _write_run(run_id, config, {backend: outcome.backend_config}, outcome.traces, dataset)
    _print_summary(run_id, config)


@app.command()
def compare(
    backends: str = typer.Option(",".join(ALL_BACKENDS), help="Comma-separated backend list."),
    dataset: str = typer.Option(..., help="Path to a dataset JSON file (see `generate`)."),
    top_k: Optional[int] = typer.Option(None),
    seed: Optional[int] = typer.Option(None),
    config_path: Optional[str] = typer.Option(None, "--config"),
    output_dir: Optional[str] = typer.Option(None),
) -> None:
    """Run several backends against the SAME dataset for a direct comparison."""
    config = _load_config(config_path, seed, top_k)
    if output_dir:
        config.output_dir = output_dir
    backend_list = _split_csv(backends)

    ds = load_dataset(dataset)
    validation = validate_dataset(ds)
    if not validation.is_valid:
        console.print("[bold red]Dataset failed validation:[/bold red]")
        for err in validation.errors:
            console.print(f"  - {err}")
        raise typer.Exit(code=1)

    run_id = new_run_id(prefix="compare")
    console.print(f"Comparing backends={backend_list} on dataset=[bold]{dataset}[/bold] (run_id={run_id})")

    try:
        outcomes = asyncio.run(run_compare(backend_list, ds, config, run_id))
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]Run failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    all_traces: List[Trace] = []
    backend_configs = {}
    for name, outcome in outcomes.items():
        all_traces.extend(outcome.traces)
        backend_configs[name] = outcome.backend_config

    _write_run(run_id, config, backend_configs, all_traces, dataset)
    _print_summary(run_id, config)


@app.command()
def scale(
    backends: str = typer.Option(",".join(ALL_BACKENDS), help="Comma-separated backend list."),
    turns: str = typer.Option("10,25,50,100", help="Comma-separated history lengths to test."),
    users: int = typer.Option(5, help="Users per history-length point."),
    seed: Optional[int] = typer.Option(None),
    config_path: Optional[str] = typer.Option(None, "--config"),
    output_dir: Optional[str] = typer.Option(None),
) -> None:
    """Sweep history length to find crossover points between backends."""
    config = _load_config(config_path, seed, None)
    if output_dir:
        config.output_dir = output_dir
    backend_list = _split_csv(backends)
    turn_sizes = [int(t.strip()) for t in turns.split(",") if t.strip()]

    run_id = new_run_id(prefix="scale")
    console.print(f"Scale experiment: backends={backend_list} turns={turn_sizes} users={users} (run_id={run_id})")

    points = asyncio.run(run_scale_experiment(backend_list, turn_sizes, users, config, run_id))

    all_traces: List[Trace] = []
    backend_configs = {}
    table = Table(title="Scale experiment")
    table.add_column("Turns")
    table.add_column("Backend")
    table.add_column("Accuracy")
    table.add_column("p95 total (ms)")
    table.add_column("Avg total tokens")

    pricing_table = load_pricing_table(config.pricing) if config.pricing else {}

    for point in points:
        for name, outcome in point.outcomes.items():
            all_traces.extend(outcome.traces)
            backend_configs[name] = outcome.backend_config
            summary = summarize_run(outcome.traces, pricing_table)[name]
            acc = summary["accuracy"]["accuracy"]
            p95 = summary["latency"]["total_ms"]["p95_ms"]
            avg_tokens = summary["tokens"]["per_query_average"]["total_tokens"]
            table.add_row(str(point.num_turns), name, f"{acc:.1%}", f"{p95:.1f}", str(avg_tokens))

    console.print(table)
    _write_run(
        run_id, config, backend_configs, all_traces, dataset_path=None,
        extra_metadata={"turn_sizes": turn_sizes, "users": users},
    )


@app.command()
def attack(
    backend: str = typer.Option(..., help="Backend to red-team: mem0, rag, or full-context."),
    dataset: str = typer.Option(..., help="Path to a dataset JSON file."),
    attack_name: str = typer.Option("all", "--attack", help="Attack id, or 'all'."),
    top_k: Optional[int] = typer.Option(None),
    config_path: Optional[str] = typer.Option(None, "--config"),
    output_dir: Optional[str] = typer.Option(None),
    max_users: int = typer.Option(5, help="Number of users to run each attack against."),
) -> None:
    """Optional adversarial red-team benchmark: memory poisoning / prompt injection.

    Defensive, research-oriented. See mem0_bench/attacks/poisoning.py.
    """
    from mem0_bench.memory.factory import build_backend

    config = _load_config(config_path, None, top_k)
    ds = load_dataset(dataset)
    backend_instance, backend_config = build_backend(backend, config)

    attacks_to_run = POISON_ATTACKS if attack_name == "all" else [
        a for a in POISON_ATTACKS if a.attack_id == attack_name
    ]
    if not attacks_to_run:
        console.print(f"[bold red]Unknown attack: {attack_name}[/bold red]")
        console.print("Available: " + ", ".join(a.attack_id for a in POISON_ATTACKS))
        raise typer.Exit(code=1)

    users = ds.users[:max_users]
    trials = []

    async def _run_all():
        for a in attacks_to_run:
            for user in users:
                trials.append(await run_attack_trial(backend_instance, user, a, top_k=config.top_k))

    console.print(f"Running {len(attacks_to_run)} attack(s) against backend=[bold]{backend}[/bold] "
                  f"over {len(users)} user(s)...")
    asyncio.run(_run_all())

    summary = aggregate_attack_trials(trials)
    table = Table(title=f"Memory poisoning red-team: {backend}")
    table.add_column("Attack")
    table.add_column("Poison retrieval rate")
    table.add_column("Attack success rate")
    table.add_column("Avg clean degradation")
    for attack_id, stats in summary.items():
        table.add_row(
            attack_id, f"{stats['poison_retrieval_rate']:.1%}",
            f"{stats['attack_success_rate']:.1%}", f"{stats['avg_clean_degradation']:+.1%}",
        )
    console.print(table)

    run_id = new_run_id(prefix="attack")
    run_dir = RunDirectory(config.output_dir, run_id)
    run_dir.write_config_snapshot(config.model_dump())
    run_dir.write_metadata({"run_id": run_id, "backend": backend, "backend_config": backend_config,
                             "dataset": dataset, "n_users": len(users)})
    run_dir.write_summary(summary)
    console.print(f"Results written to [bold]{run_dir.path}[/bold]")


@app.command()
def report(
    run_id: str = typer.Option(..., "--run", help="Run id under the output directory."),
    output_dir: str = typer.Option("results", help="Base results directory."),
) -> None:
    """Print a terminal summary for a previously executed run."""
    summary = load_summary(output_dir, run_id)
    if summary is None:
        console.print(f"[bold red]No summary found for run '{run_id}' under {output_dir}[/bold red]")
        raise typer.Exit(code=1)
    _render_summary_tables(summary)


@app.command()
def dashboard(
    port: int = typer.Option(8501, help="Port to serve the Streamlit dashboard on."),
) -> None:
    """Launch the results dashboard (requires the `dashboard` extra)."""
    app_path = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
    console.print(f"Launching dashboard from {app_path} on port {port} ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)], check=False)


# ---- shared helpers ----

def _write_run(run_id, config, backend_configs, traces, dataset_path, extra_metadata=None) -> None:
    run_dir = RunDirectory(config.output_dir, run_id)
    run_dir.write_config_snapshot(config.model_dump())
    metadata = {
        "run_id": run_id, "seed": config.seed, "dataset": dataset_path,
        "backends": list(backend_configs.keys()), "backend_configs": backend_configs,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    run_dir.write_metadata(metadata)

    from mem0_bench.storage.traces import TraceWriter
    writer = TraceWriter(run_dir.traces_path)
    for t in traces:
        writer.write(t)
    writer.close()

    run_dir.write_metrics_csv(traces)
    pricing_table = load_pricing_table(config.pricing) if config.pricing else {}
    summary = summarize_run(traces, pricing_table)
    run_dir.write_summary(summary)
    console.print(f"Results written to [bold]{run_dir.path}[/bold]")


def _print_summary(run_id: str, config: BenchmarkConfig) -> None:
    summary = load_summary(config.output_dir, run_id)
    if summary:
        _render_summary_tables(summary)


def _render_summary_tables(summary: dict) -> None:
    console.print("\n[bold]MEMORY BENCHMARK[/bold]")
    console.print("-" * 60)

    table = Table()
    table.add_column("Backend")
    table.add_column("R@5")
    table.add_column("p50 (ms)")
    table.add_column("p95 (ms)")
    table.add_column("Avg tokens")
    table.add_column("Accuracy")

    for backend, stats in summary.items():
        retrieval = stats.get("retrieval")
        r5 = f"{retrieval['k5']['recall']:.2f}" if retrieval and "k5" in retrieval else "N/A"
        p50 = stats["latency"]["total_ms"]["p50_ms"]
        p95 = stats["latency"]["total_ms"]["p95_ms"]
        tokens = stats["tokens"]["per_query_average"]["total_tokens"]
        accuracy = stats["accuracy"]["accuracy"]
        table.add_row(backend, r5, f"{p50:.1f}", f"{p95:.1f}", str(tokens), f"{accuracy:.1%}")

    console.print(table)

    for backend, stats in summary.items():
        console.print(f"\n[bold]{backend}[/bold] by query type:")
        by_type_table = Table()
        by_type_table.add_column("Query type")
        by_type_table.add_column("Accuracy")
        by_type_table.add_column("Recall@5")
        for qtype, qstats in stats.get("by_query_type", {}).items():
            acc = qstats["accuracy"]["accuracy"]
            retrieval = qstats.get("retrieval")
            r5 = f"{retrieval['k5']['recall']:.2f}" if retrieval and "k5" in retrieval else "N/A"
            by_type_table.add_row(qtype, f"{acc:.1%}", r5)
        console.print(by_type_table)

    failure_notes = [
        f"{backend}: {stats['accuracy']['failed']}/{stats['accuracy']['total']} failed "
        f"({stats['accuracy']['failure_rate']:.1%})"
        for backend, stats in summary.items() if stats["accuracy"]["failed"] > 0
    ]
    if failure_notes:
        console.print("\n[yellow]Failures (counted as incorrect, not excluded):[/yellow]")
        for note in failure_notes:
            console.print(f"  - {note}")


if __name__ == "__main__":
    app()
