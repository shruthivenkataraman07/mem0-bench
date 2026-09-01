"""Per-query evaluation: bridges backend-specific retrieval results to the
dataset's generator-derived gold labels, and scores the final answer.

Backends live in different id spaces (RAG chunks are keyed by message_id,
Mem0 assigns its own internal memory ids after LLM fact-extraction, and a
hypothetical custom backend could use anything). To compare them fairly
without forcing every backend to know about the dataset's gold memory ids,
retrieval relevance is judged by normalized content overlap between a
retrieved item and the question's *expected* gold memories, scoped only
to that question's small expected set, using a greedy one-to-one match so
one retrieved item can't double-count against two expected memories.
"""

from __future__ import annotations

from typing import Dict, List

from mem0_bench.datasets.schema import EvalQuestion, GoldMemory
from mem0_bench.memory.base import Memory
from mem0_bench.metrics.answer_quality import AnswerQualityResult, evaluate_answer, normalize
from mem0_bench.metrics.retrieval import retrieval_metrics_at_k

RETRIEVAL_KS = (1, 3, 5)
DEFAULT_CONTENT_MATCH_THRESHOLD = 0.5


def _content_match(retrieved_norm: str, gold_norm: str, threshold: float) -> bool:
    if not retrieved_norm or not gold_norm:
        return False
    if gold_norm in retrieved_norm or retrieved_norm in gold_norm:
        return True
    gold_tokens = set(gold_norm.split())
    retrieved_tokens = set(retrieved_norm.split())
    if not gold_tokens:
        return False
    overlap = len(gold_tokens & retrieved_tokens) / len(gold_tokens)
    return overlap >= threshold


def match_retrieved_to_gold(
    retrieved: List[Memory],
    expected_memory_ids: List[str],
    gold_by_id: Dict[str, GoldMemory],
    threshold: float = DEFAULT_CONTENT_MATCH_THRESHOLD,
) -> List[bool]:
    """Rank-ordered relevance judgments for `retrieved` against this question's
    expected gold memories, via a greedy one-to-one content match."""
    expected_contents = {
        mid: normalize(gold_by_id[mid].content) for mid in expected_memory_ids if mid in gold_by_id
    }
    consumed: set = set()
    relevance: List[bool] = []
    for m in retrieved:
        norm_r = normalize(m.content)
        matched_id = None
        for mid, gold_norm in expected_contents.items():
            if mid in consumed:
                continue
            if _content_match(norm_r, gold_norm, threshold):
                matched_id = mid
                break
        if matched_id is not None:
            consumed.add(matched_id)
            relevance.append(True)
        else:
            relevance.append(False)
    return relevance


def evaluate_retrieval(
    retrieved: List[Memory],
    question: EvalQuestion,
    gold_by_id: Dict[str, GoldMemory],
) -> Dict[str, Dict[str, float]]:
    relevance = match_retrieved_to_gold(retrieved, question.expected_memory_ids, gold_by_id)
    num_relevant_total = len(question.expected_memory_ids)
    return {
        f"k{k}": retrieval_metrics_at_k(relevance, k, num_relevant_total)
        for k in RETRIEVAL_KS
    }


def score_answer(prediction: str, question: EvalQuestion) -> AnswerQualityResult:
    return evaluate_answer(prediction, question.expected_answer)
