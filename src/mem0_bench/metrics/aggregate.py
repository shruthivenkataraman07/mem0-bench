"""Aggregate a list of per-query :class:`Trace` objects into a run summary.

Guards against the common benchmark-aggregation mistakes called out in the
README ("Benchmark validity"):

- Failed queries are **kept in the denominator** as incorrect, not dropped
  (so a flaky backend can't inflate its accuracy by failing on hard
  queries). The failure rate is reported alongside accuracy so the two are
  never conflated.
- Latency percentiles are always computed from the raw sample list, never
  by averaging percentiles from sub-groups.
- Retrieval metrics are only computed for traces where
  ``retrieval_applicable`` is True (excludes full-context).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from mem0_bench.metrics.latency import summarize_latencies
from mem0_bench.metrics.tokens import ModelPricing, TokenAggregator, TokenUsage
from mem0_bench.storage.traces import Trace

RETRIEVAL_KS = (1, 3, 5)


def _accuracy(traces: List[Trace]) -> Dict[str, Any]:
    total = len(traces)
    if total == 0:
        return {"total": 0, "successful": 0, "failed": 0, "failure_rate": 0.0, "accuracy": 0.0}
    failed = [t for t in traces if not t.success]
    correct = sum(1 for t in traces if t.success and t.is_correct)
    return {
        "total": total,
        "successful": total - len(failed),
        "failed": len(failed),
        "failure_rate": len(failed) / total,
        # Denominator is ALL attempted queries, including failures (counted as incorrect).
        "accuracy": correct / total,
    }


def _retrieval_summary(traces: List[Trace]) -> Optional[Dict[str, Any]]:
    applicable = [t for t in traces if t.retrieval_applicable and t.success]
    if not applicable:
        return None
    out: Dict[str, Any] = {}
    for k in RETRIEVAL_KS:
        key = f"k{k}"
        metrics_present: List[Dict[str, float]] = []
        for t in applicable:
            m = t.retrieval_metrics.get(key)
            if m:
                metrics_present.append(m)
        if not metrics_present:
            continue
        n = len(metrics_present)
        out[key] = {
            metric: sum(m[metric] for m in metrics_present) / n
            for metric in ("precision", "recall", "f1", "mrr", "ndcg", "hit_rate")
        }
    out["_n_queries"] = len(applicable)
    return out


def _latency_summary(traces: List[Trace]) -> Dict[str, Any]:
    successful = [t for t in traces if t.success]
    return {
        "retrieval_ms": summarize_latencies([t.retrieval_ms for t in successful]).to_dict(),
        "generation_ms": summarize_latencies([t.generation_ms for t in successful]).to_dict(),
        "total_ms": summarize_latencies([t.total_ms for t in successful]).to_dict(),
    }


def _token_summary(traces: List[Trace], pricing: Optional[ModelPricing]) -> Dict[str, Any]:
    agg = TokenAggregator()
    for t in traces:
        if t.success:
            agg.record(TokenUsage(t.input_tokens, t.retrieved_tokens, t.output_tokens))
    total = agg.total()
    per_query = agg.per_query_average()
    out: Dict[str, Any] = {
        "total": vars(total) | {"total_tokens": total.total_tokens},
        "per_query_average": vars(per_query) | {"total_tokens": per_query.total_tokens},
    }
    if pricing is not None and agg.usages:
        out["estimated_cost_usd"] = agg.estimated_cost(pricing)
        out["estimated_cost_usd_per_1k_queries"] = agg.estimated_cost_per_1k_queries(pricing)
    return out


def summarize_backend(traces: List[Trace], pricing: Optional[ModelPricing] = None) -> Dict[str, Any]:
    by_type: Dict[str, List[Trace]] = defaultdict(list)
    for t in traces:
        by_type[t.query_type].append(t)

    return {
        "accuracy": _accuracy(traces),
        "retrieval": _retrieval_summary(traces),
        "latency": _latency_summary(traces),
        "tokens": _token_summary(traces, pricing),
        "by_query_type": {
            qtype: {
                "accuracy": _accuracy(qtraces),
                "retrieval": _retrieval_summary(qtraces),
            }
            for qtype, qtraces in sorted(by_type.items())
        },
    }


def compute_efficiency_score(
    accuracy: float, p95_latency_ms: float, total_tokens_per_query: float,
    weight_accuracy: float, weight_latency: float, weight_tokens: float,
    latency_reference_ms: float, tokens_reference: float,
) -> float:
    """EXPERIMENTAL, not a validated metric; see README, "Engineering
    composite metric". A weighted blend of accuracy (higher is better) and
    normalized latency/token cost (lower is better), each clipped to
    [0, 1] against a configurable reference point. Always report the raw
    underlying accuracy/latency/token numbers alongside this.
    """
    latency_score = max(0.0, 1.0 - (p95_latency_ms / latency_reference_ms)) if latency_reference_ms > 0 else 0.0
    token_score = max(0.0, 1.0 - (total_tokens_per_query / tokens_reference)) if tokens_reference > 0 else 0.0
    return (
        weight_accuracy * accuracy
        + weight_latency * min(1.0, latency_score)
        + weight_tokens * min(1.0, token_score)
    )


def summarize_run(traces: List[Trace], pricing_table: Optional[Dict[str, ModelPricing]] = None) -> Dict[str, Any]:
    """Group traces by backend and summarize each independently."""
    by_backend: Dict[str, List[Trace]] = defaultdict(list)
    for t in traces:
        by_backend[t.backend].append(t)

    summary: Dict[str, Any] = {}
    for backend, backend_traces in sorted(by_backend.items()):
        model = backend_traces[0].model if backend_traces else None
        pricing = pricing_table.get(model) if pricing_table and model else None
        summary[backend] = summarize_backend(backend_traces, pricing)
    return summary
