"""Retrieval-quality metrics.

These operate on a *relevance judgment list*: for a ranked list of retrieved
items, ``relevance[i]`` is ``True`` iff the item at rank ``i`` (0-indexed) is
relevant to the query. This keeps the metrics themselves independent of how
relevance was decided; see :mod:`mem0_bench.benchmark.evaluator` for how
retrieved memories (which live in backend-specific id spaces) are matched
against a question's gold memories to produce these judgments.

All functions handle multiple gold/relevant items per query correctly.
"""

from __future__ import annotations

import math
from typing import List


def _truncate(relevance: List[bool], k: int) -> List[bool]:
    if k < 0:
        raise ValueError("k must be >= 0")
    return relevance[:k]


def precision_at_k(relevance: List[bool], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant.

    Divides by k (not by the number actually retrieved) unless fewer than k
    items were retrieved at all, in which case it divides by what was
    retrieved; this avoids rewarding a backend for retrieving fewer items.
    """
    if k == 0:
        return 0.0
    top_k = _truncate(relevance, k)
    denom = len(top_k) if len(top_k) < k else k
    if denom == 0:
        return 0.0
    return sum(top_k) / denom


def recall_at_k(relevance: List[bool], k: int, num_relevant_total: int) -> float:
    if num_relevant_total <= 0:
        return 0.0
    top_k = _truncate(relevance, k)
    return sum(top_k) / num_relevant_total


def f1_at_k(relevance: List[bool], k: int, num_relevant_total: int) -> float:
    p = precision_at_k(relevance, k)
    r = recall_at_k(relevance, k, num_relevant_total)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(relevance: List[bool]) -> float:
    """Reciprocal rank of the first relevant item, 0.0 if none is relevant."""
    for i, rel in enumerate(relevance):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevance: List[bool], k: int, num_relevant_total: int) -> float:
    """Binary-relevance nDCG@k."""
    top_k = _truncate(relevance, k)
    dcg = sum((1.0 if rel else 0.0) / math.log2(i + 2) for i, rel in enumerate(top_k))
    ideal_hits = min(num_relevant_total, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(relevance: List[bool], k: int) -> float:
    return 1.0 if any(_truncate(relevance, k)) else 0.0


def retrieval_metrics_at_k(
    relevance: List[bool], k: int, num_relevant_total: int
) -> dict:
    """Convenience bundle of all retrieval metrics at a single k."""
    return {
        "precision": precision_at_k(relevance, k),
        "recall": recall_at_k(relevance, k, num_relevant_total),
        "f1": f1_at_k(relevance, k, num_relevant_total),
        "mrr": mrr(relevance),
        "ndcg": ndcg_at_k(relevance, k, num_relevant_total),
        "hit_rate": hit_rate_at_k(relevance, k),
    }
