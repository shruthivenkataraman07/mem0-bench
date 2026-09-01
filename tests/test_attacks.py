from __future__ import annotations

import pytest

from mem0_bench.attacks.evaluator import aggregate_attack_trials, run_attack_trial
from mem0_bench.attacks.poisoning import POISON_ATTACKS, build_poison_message, poisoned_copy
from mem0_bench.attacks.prompt_injection import generation_vulnerable, retrieval_vulnerable
from mem0_bench.datasets.generator import generate_dataset
from mem0_bench.llm.client import LocalExtractiveLLM
from mem0_bench.llm.tokenizer import get_tokenizer
from mem0_bench.memory.base import Memory
from mem0_bench.memory.embeddings import HashingEmbedding
from mem0_bench.memory.rag_adapter import RAGBackend
from mem0_bench.memory.vector_store import NumpyVectorStore


def _rag_backend():
    return RAGBackend(
        embedding_provider=HashingEmbedding(dim=64),
        vector_store=NumpyVectorStore(),
        llm=LocalExtractiveLLM(), tokenizer=get_tokenizer("approximate"), default_top_k=5,
    )


def test_poisoned_copy_appends_one_message_without_mutating_original():
    dataset = generate_dataset(num_users=1, num_turns=30, seed=42)
    user = dataset.users[0]
    attack = POISON_ATTACKS[0]

    poisoned_user, poison_msg = poisoned_copy(user, attack)

    assert len(poisoned_user.messages) == len(user.messages) + 1
    assert len(user.messages) == len(dataset.users[0].messages)  # original untouched
    assert poisoned_user.messages[-1].message_id == poison_msg.message_id
    assert "evilcorp" in poison_msg.content.lower()


def test_build_poison_message_references_real_employer():
    dataset = generate_dataset(num_users=1, num_turns=30, seed=42)
    user = dataset.users[0]
    attack = next(a for a in POISON_ATTACKS if a.attack_id == "false_fact_injection")
    msg = build_poison_message(attack, user, message_id="poison_1")
    assert "evilcorp" in msg.content.lower()
    assert msg.role == "user"


def test_retrieval_vulnerable_detects_poison_in_retrieved_memories():
    attack = POISON_ATTACKS[0]
    retrieved = [Memory(memory_id="m1", content="I work at EvilCorp now.")]
    assert retrieval_vulnerable(retrieved, attack) is True


def test_retrieval_vulnerable_false_when_poison_absent():
    attack = POISON_ATTACKS[0]
    retrieved = [Memory(memory_id="m1", content="I work at Stripe.")]
    assert retrieval_vulnerable(retrieved, attack) is False


def test_generation_vulnerable_detects_obeyed_instruction():
    attack = next(a for a in POISON_ATTACKS if a.attack_id == "prompt_injection_override")
    assert generation_vulnerable("My employer is EvilCorp.", attack) is True
    assert generation_vulnerable("My employer is Stripe.", attack) is False


@pytest.mark.asyncio
async def test_run_attack_trial_reports_retrieval_and_generation_signals_independently():
    dataset = generate_dataset(num_users=1, num_turns=40, seed=42)
    user = dataset.users[0]
    attack = next(a for a in POISON_ATTACKS if a.attack_id == "false_fact_injection")
    backend = _rag_backend()

    result = await run_attack_trial(backend, user, attack, top_k=5)

    assert result.attack_id == attack.attack_id
    assert result.user_id == user.user_id
    assert isinstance(result.retrieval_vulnerable, bool)
    assert isinstance(result.generation_vulnerable, bool)
    assert result.attack_success == result.generation_vulnerable
    assert 0.0 <= result.clean_accuracy_other <= 1.0
    assert 0.0 <= result.poisoned_accuracy_other <= 1.0


@pytest.mark.asyncio
async def test_run_attack_trial_does_not_mutate_caller_dataset():
    dataset = generate_dataset(num_users=1, num_turns=30, seed=7)
    user = dataset.users[0]
    original_len = len(user.messages)
    attack = POISON_ATTACKS[0]
    backend = _rag_backend()

    await run_attack_trial(backend, user, attack, top_k=5)

    assert len(user.messages) == original_len


def test_aggregate_attack_trials_computes_rates():
    from mem0_bench.attacks.evaluator import AttackTrialResult

    trials = [
        AttackTrialResult("a1", "u1", True, True, True, 1.0, 0.5, 0.5),
        AttackTrialResult("a1", "u2", False, False, False, 1.0, 1.0, 0.0),
    ]
    summary = aggregate_attack_trials(trials)
    assert summary["a1"]["n_trials"] == 2
    assert summary["a1"]["poison_retrieval_rate"] == pytest.approx(0.5)
    assert summary["a1"]["attack_success_rate"] == pytest.approx(0.5)
    assert summary["a1"]["avg_clean_degradation"] == pytest.approx(0.25)


def test_all_registered_attacks_have_unique_ids():
    ids = [a.attack_id for a in POISON_ATTACKS]
    assert len(ids) == len(set(ids))
