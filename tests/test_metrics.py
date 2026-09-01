import pytest

from mem0_bench.metrics.aggregate import compute_efficiency_score, summarize_backend, summarize_run
from mem0_bench.metrics.answer_quality import (
    exact_match,
    normalize,
    normalized_match,
    structured_fact_match,
)
from mem0_bench.metrics.latency import LatencyTracker, percentile, summarize_latencies
from mem0_bench.metrics.tokens import ModelPricing, TokenAggregator, TokenUsage, load_pricing_table
from mem0_bench.storage.traces import Trace

# ---- latency ----

def test_percentile_matches_known_values():
    samples = [10, 20, 30, 40, 50]
    assert percentile(samples, 0) == 10
    assert percentile(samples, 100) == 50
    assert percentile(samples, 50) == 30


def test_latency_tracker_summary_basic_stats():
    tracker = LatencyTracker()
    for v in [100, 200, 300, 400, 500]:
        tracker.record(v)
    summary = tracker.summarize()
    assert summary.count == 5
    assert summary.mean == pytest.approx(300)
    assert summary.min == 100
    assert summary.max == 500


def test_latency_summary_flags_low_confidence_for_small_samples():
    summary = summarize_latencies([1.0, 2.0, 3.0])
    assert summary.low_confidence is True


def test_latency_summary_not_low_confidence_with_enough_samples():
    summary = summarize_latencies([float(i) for i in range(25)])
    assert summary.low_confidence is False


def test_latency_summary_empty_samples_does_not_crash():
    summary = summarize_latencies([])
    assert summary.count == 0
    assert summary.low_confidence is True


# ---- tokens ----

def test_token_usage_total():
    usage = TokenUsage(input_tokens=10, retrieved_tokens=5, output_tokens=3)
    assert usage.total_tokens == 18


def test_model_pricing_cost():
    pricing = ModelPricing(input_per_1k=1.0, output_per_1k=2.0)
    cost = pricing.cost(input_tokens=1000, output_tokens=500)
    assert cost == pytest.approx(1.0 + 1.0)


def test_token_aggregator_total_and_average():
    agg = TokenAggregator()
    agg.record(TokenUsage(10, 0, 5))
    agg.record(TokenUsage(20, 0, 5))
    total = agg.total()
    assert total.input_tokens == 30
    avg = agg.per_query_average()
    assert avg.input_tokens == 15


def test_token_aggregator_cost_per_1k_queries():
    agg = TokenAggregator()
    for _ in range(10):
        agg.record(TokenUsage(1000, 0, 0))
    pricing = ModelPricing(input_per_1k=1.0, output_per_1k=0.0)
    # 10 queries, 1000 input tokens each -> $1 per query -> $1000 per 1k queries
    assert agg.estimated_cost_per_1k_queries(pricing) == pytest.approx(1000.0)


def test_load_pricing_table():
    table = load_pricing_table({"gpt-x": {"input_per_1k": 0.1, "output_per_1k": 0.2}})
    assert table["gpt-x"].input_per_1k == 0.1


# ---- answer quality ----

def test_exact_match():
    assert exact_match("Stripe", "Stripe")
    assert not exact_match("stripe", "Stripe")


def test_normalize_strips_punctuation_and_case():
    assert normalize("Stripe.") == "stripe"
    assert normalize("  Stripe,  Inc.  ") == "stripe inc"


def test_normalized_match():
    assert normalized_match("Stripe.", "stripe")
    assert not normalized_match("Google", "Stripe")


def test_structured_fact_match_substring():
    assert structured_fact_match("You currently work at Stripe.", "Stripe")


def test_structured_fact_match_token_overlap():
    assert structured_fact_match("engineering manager at Stripe", "Stripe at engineering manager")


def test_structured_fact_match_false_for_unrelated():
    assert not structured_fact_match("You work at Google.", "Stripe")


def test_structured_fact_match_empty_gold_is_false():
    assert not structured_fact_match("anything", "")


# ---- aggregate ----

def _make_trace(**overrides) -> Trace:
    base = dict(
        run_id="r1", backend="rag", model="local-extractive-v1", seed=1,
        user_id="u1", query_id="q1", query_type="preference", query="q",
        expected_answer="Thai", retrieval_applicable=True,
        success=True, is_correct=True,
        retrieval_ms=10.0, generation_ms=20.0, total_ms=30.0,
        input_tokens=100, retrieved_tokens=50, output_tokens=10,
        retrieval_metrics={
            "k5": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "mrr": 1.0, "ndcg": 1.0, "hit_rate": 1.0},
        },
    )
    base.update(overrides)
    return Trace(**base)


def test_summarize_backend_accuracy_counts_failures_in_denominator():
    traces = [_make_trace(success=True, is_correct=True), _make_trace(success=False, is_correct=None, query_id="q2")]
    summary = summarize_backend(traces)
    assert summary["accuracy"]["total"] == 2
    assert summary["accuracy"]["failed"] == 1
    # 1 correct out of 2 total (failure counted as incorrect, not dropped)
    assert summary["accuracy"]["accuracy"] == pytest.approx(0.5)
    assert summary["accuracy"]["failure_rate"] == pytest.approx(0.5)


def test_summarize_backend_excludes_full_context_style_traces_from_retrieval():
    traces = [_make_trace(retrieval_applicable=False, retrieval_metrics={})]
    summary = summarize_backend(traces)
    assert summary["retrieval"] is None


def test_summarize_run_groups_by_backend():
    traces = [_make_trace(backend="rag", query_id="q1"), _make_trace(backend="mem0", query_id="q2")]
    summary = summarize_run(traces)
    assert set(summary) == {"rag", "mem0"}


def test_summarize_by_query_type_breakdown():
    traces = [
        _make_trace(query_type="preference", query_id="q1"),
        _make_trace(query_type="temporal", query_id="q2", is_correct=False),
    ]
    summary = summarize_backend(traces)
    assert set(summary["by_query_type"]) == {"preference", "temporal"}
    assert summary["by_query_type"]["temporal"]["accuracy"]["accuracy"] == 0.0


def test_compute_efficiency_score_bounds():
    score = compute_efficiency_score(
        accuracy=1.0, p95_latency_ms=0.0, total_tokens_per_query=0.0,
        weight_accuracy=0.5, weight_latency=0.3, weight_tokens=0.2,
        latency_reference_ms=1000.0, tokens_reference=1000.0,
    )
    assert score == pytest.approx(1.0)


def test_compute_efficiency_score_penalizes_high_latency():
    good = compute_efficiency_score(1.0, 100.0, 100.0, 0.5, 0.3, 0.2, 1000.0, 1000.0)
    bad = compute_efficiency_score(1.0, 5000.0, 100.0, 0.5, 0.3, 0.2, 1000.0, 1000.0)
    assert bad < good
