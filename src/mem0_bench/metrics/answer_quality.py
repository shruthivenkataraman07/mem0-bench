"""Final-answer-quality metrics, kept deliberately separate from retrieval
quality: a backend can retrieve the right memory and still answer wrong (or
vice versa via a lucky guess), so the two must never be conflated.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Optional

from mem0_bench.llm.client import LLMClient

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, gold: str) -> bool:
    return prediction.strip() == gold.strip()


def normalized_match(prediction: str, gold: str) -> bool:
    return normalize(prediction) == normalize(gold)


def structured_fact_match(prediction: str, gold: str, token_overlap_threshold: float = 0.6) -> bool:
    """Looser check for free-form generated answers: does the prediction contain
    the gold answer's key content, or share most of its tokens?

    Used because a real LLM will rarely echo the gold string verbatim
    (e.g. gold="Stripe", prediction="You currently work at Stripe.").
    """
    norm_pred = normalize(prediction)
    norm_gold = normalize(gold)
    if not norm_gold:
        return False
    if norm_gold in norm_pred:
        return True
    gold_tokens = set(norm_gold.split())
    pred_tokens = set(norm_pred.split())
    if not gold_tokens:
        return False
    overlap = len(gold_tokens & pred_tokens) / len(gold_tokens)
    return overlap >= token_overlap_threshold


@dataclass
class JudgeVerdict:
    score: float  # 0.0-1.0
    raw_response: str
    judge_model: str
    prompt: str


JUDGE_PROMPT_TEMPLATE = """You are grading whether a candidate answer correctly \
conveys the same fact as a reference answer, for the question below.

Question: {query}
Reference answer: {gold}
Candidate answer: {prediction}

Reply with exactly one line: "SCORE: 1" if the candidate answer conveys the \
same fact as the reference answer, or "SCORE: 0" if it does not or is \
unsupported/contradictory. Do not explain."""


async def llm_judge(
    query: str, prediction: str, gold: str, judge: LLMClient
) -> JudgeVerdict:
    """Optional LLM-as-judge scoring.

    This is explicitly non-deterministic when `judge` is a real LLM provider
    (e.g. OpenAI); callers must record `judge.model`/`judge.provider`
    alongside the score and never treat it as ground truth when a
    deterministic label (exact/normalized/structured match) is available.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, gold=gold, prediction=prediction)
    response = await judge.generate(prompt)
    match = re.search(r"SCORE:\s*(0|1)", response.text)
    score = float(match.group(1)) if match else 0.0
    return JudgeVerdict(
        score=score,
        raw_response=response.text,
        judge_model=f"{judge.provider}:{judge.model}",
        prompt=prompt,
    )


@dataclass
class AnswerQualityResult:
    exact_match: bool
    normalized_match: bool
    structured_match: bool
    judge_verdict: Optional[JudgeVerdict] = None

    @property
    def is_correct(self) -> bool:
        """Deterministic correctness signal used for aggregate accuracy.

        Uses the structured match (a superset of exact/normalized) since it
        is the most realistic deterministic signal for free-form answers.
        The LLM judge, when present, is reported separately and never
        silently substituted for this deterministic label.
        """
        return self.structured_match

    def to_dict(self) -> dict:
        d: dict = {
            "exact_match": self.exact_match,
            "normalized_match": self.normalized_match,
            "structured_match": self.structured_match,
        }
        if self.judge_verdict is not None:
            d["judge_score"] = self.judge_verdict.score
            d["judge_model"] = self.judge_verdict.judge_model
        return d


def evaluate_answer(prediction: str, gold: str) -> AnswerQualityResult:
    return AnswerQualityResult(
        exact_match=exact_match(prediction, gold),
        normalized_match=normalized_match(prediction, gold),
        structured_match=structured_fact_match(prediction, gold),
    )
