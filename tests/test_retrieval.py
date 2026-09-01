import pytest

from mem0_bench.metrics.retrieval import (
    f1_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    retrieval_metrics_at_k,
)


def test_precision_at_k_basic():
    relevance = [True, False, True, False, False]
    assert precision_at_k(relevance, 1) == 1.0
    assert precision_at_k(relevance, 3) == pytest.approx(2 / 3)
    assert precision_at_k(relevance, 5) == pytest.approx(2 / 5)


def test_precision_at_k_fewer_retrieved_than_k():
    relevance = [True, True]
    assert precision_at_k(relevance, 5) == 1.0  # divide by 2 retrieved, not 5


def test_precision_at_k_zero_k():
    assert precision_at_k([True], 0) == 0.0


def test_recall_at_k_multiple_gold():
    relevance = [False, True, False, True, False]
    assert recall_at_k(relevance, 5, num_relevant_total=2) == 1.0
    assert recall_at_k(relevance, 1, num_relevant_total=2) == 0.0
    assert recall_at_k(relevance, 2, num_relevant_total=2) == 0.5


def test_recall_at_k_no_gold():
    assert recall_at_k([True, True], 5, num_relevant_total=0) == 0.0


def test_f1_at_k_zero_when_no_overlap():
    assert f1_at_k([False, False], 2, num_relevant_total=2) == 0.0


def test_f1_at_k_perfect():
    relevance = [True, True]
    assert f1_at_k(relevance, 2, num_relevant_total=2) == pytest.approx(1.0)


def test_mrr_first_hit():
    assert mrr([True, False, False]) == 1.0
    assert mrr([False, True, False]) == pytest.approx(0.5)
    assert mrr([False, False, False]) == 0.0


def test_ndcg_at_k_perfect_ranking():
    relevance = [True, True, False]
    assert ndcg_at_k(relevance, 3, num_relevant_total=2) == pytest.approx(1.0)


def test_ndcg_at_k_worst_ranking_still_positive_if_hit_present():
    relevance = [False, False, True]
    score = ndcg_at_k(relevance, 3, num_relevant_total=1)
    assert 0 < score < 1.0


def test_ndcg_at_k_imperfect_ranking_scores_below_perfect():
    perfect = ndcg_at_k([True, True, False], 3, num_relevant_total=2)
    imperfect = ndcg_at_k([True, False, True], 3, num_relevant_total=2)
    assert imperfect < perfect


def test_ndcg_at_k_no_gold_is_zero():
    assert ndcg_at_k([True], 1, num_relevant_total=0) == 0.0


def test_hit_rate_at_k():
    assert hit_rate_at_k([False, False, True], 3) == 1.0
    assert hit_rate_at_k([False, False, True], 2) == 0.0
    assert hit_rate_at_k([], 3) == 0.0


def test_retrieval_metrics_bundle_keys():
    bundle = retrieval_metrics_at_k([True, False], 2, num_relevant_total=1)
    assert set(bundle) == {"precision", "recall", "f1", "mrr", "ndcg", "hit_rate"}
