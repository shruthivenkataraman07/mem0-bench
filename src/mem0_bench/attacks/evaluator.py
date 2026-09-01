"""Orchestrates one memory-poisoning red-team trial: ingest a user's normal
conversation (optionally with one poison message appended), then measure

- poison retrieval rate: was the poisoned content retrieved for the
  targeted query?
- generation vulnerability / attack success: did the final answer reflect
  the poisoned content instead of the true (gold) fact?
- clean-answer degradation: did accuracy on the OTHER (untargeted)
  questions get worse purely because the poison message was present?

Every trial runs the SAME user/backend/config twice, once clean and once
poisoned, so degradation is measured against a matched baseline rather
than an unrelated run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from mem0_bench.attacks.poisoning import PoisonAttack, poisoned_copy
from mem0_bench.attacks.prompt_injection import generation_vulnerable, retrieval_vulnerable
from mem0_bench.benchmark.evaluator import score_answer
from mem0_bench.datasets.schema import QueryCategory, UserDataset
from mem0_bench.memory.base import MemoryBackend


@dataclass
class AttackTrialResult:
    attack_id: str
    user_id: str
    retrieval_vulnerable: bool
    generation_vulnerable: bool
    attack_success: bool  # alias of generation_vulnerable, kept explicit for reporting
    clean_accuracy_other: float
    poisoned_accuracy_other: float
    # clean_accuracy_other - poisoned_accuracy_other; >0 means poisoning hurt unrelated answers
    clean_degradation: float


def _targeted_question(user: UserDataset, attack: PoisonAttack):
    for q in user.questions:
        if attack.target_fact_type == "employer" and q.category == QueryCategory.CONTRADICTION:
            return q
        if attack.target_fact_type == "employer" and q.category == QueryCategory.MULTI_HOP:
            return q
        if attack.target_fact_type == "food_preference" and q.category == QueryCategory.PREFERENCE:
            return q
    return None


async def _ingest_all(backend: MemoryBackend, user: UserDataset) -> Optional[str]:
    for session_id in user.sessions:
        messages = [m for m in user.messages if m.session_id == session_id]
        result = await backend.add(user.user_id, messages)
        if not result.ok:
            return result.error
    return None


async def _accuracy_on_other_questions(
    backend: MemoryBackend, user: UserDataset, exclude_question_id: Optional[str], top_k: int
) -> float:
    others = [q for q in user.questions if q.question_id != exclude_question_id]
    if not others:
        return 0.0
    correct = 0
    for q in others:
        ans = await backend.answer(user.user_id, q.query, k=top_k)
        if ans.ok and score_answer(ans.text, q).is_correct:
            correct += 1
    return correct / len(others)


async def run_attack_trial(
    backend: MemoryBackend, user: UserDataset, attack: PoisonAttack, top_k: int = 5
) -> AttackTrialResult:
    targeted_question = _targeted_question(user, attack)

    # --- clean baseline ---
    await backend.reset(user.user_id)
    await _ingest_all(backend, user)
    clean_accuracy_other = await _accuracy_on_other_questions(
        backend, user, targeted_question.question_id if targeted_question else None, top_k
    )

    # --- poisoned run ---
    poisoned_user, _poison_msg = poisoned_copy(user, attack)
    await backend.reset(user.user_id)
    await _ingest_all(backend, poisoned_user)

    retrieval_vuln = False
    generation_vuln = False
    if targeted_question is not None:
        ans = await backend.answer(user.user_id, targeted_question.query, k=top_k)
        if ans.ok:
            retrieval_vuln = retrieval_vulnerable(ans.retrieved_memories, attack)
            generation_vuln = generation_vulnerable(ans.text, attack)

    poisoned_accuracy_other = await _accuracy_on_other_questions(
        backend, poisoned_user, targeted_question.question_id if targeted_question else None, top_k
    )

    return AttackTrialResult(
        attack_id=attack.attack_id,
        user_id=user.user_id,
        retrieval_vulnerable=retrieval_vuln,
        generation_vulnerable=generation_vuln,
        attack_success=generation_vuln,
        clean_accuracy_other=clean_accuracy_other,
        poisoned_accuracy_other=poisoned_accuracy_other,
        clean_degradation=clean_accuracy_other - poisoned_accuracy_other,
    )


def aggregate_attack_trials(trials: List[AttackTrialResult]) -> Dict[str, Dict[str, float]]:
    by_attack: Dict[str, List[AttackTrialResult]] = {}
    for t in trials:
        by_attack.setdefault(t.attack_id, []).append(t)

    summary = {}
    for attack_id, group in by_attack.items():
        n = len(group)
        summary[attack_id] = {
            "n_trials": n,
            "poison_retrieval_rate": sum(t.retrieval_vulnerable for t in group) / n,
            "attack_success_rate": sum(t.attack_success for t in group) / n,
            "avg_clean_degradation": sum(t.clean_degradation for t in group) / n,
        }
    return summary
