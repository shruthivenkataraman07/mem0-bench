from __future__ import annotations

import pytest

from mem0_bench.benchmark.runner import BenchmarkRunner
from mem0_bench.datasets.generator import generate_dataset
from mem0_bench.llm.client import LocalExtractiveLLM
from mem0_bench.llm.tokenizer import get_tokenizer
from mem0_bench.memory.base import Answer, IngestResult, MemoryBackend, RetrievalResult
from mem0_bench.memory.embeddings import HashingEmbedding
from mem0_bench.memory.full_context import FullContextBackend
from mem0_bench.memory.rag_adapter import RAGBackend
from mem0_bench.memory.vector_store import NumpyVectorStore
from mem0_bench.storage.traces import Trace


def _rag_backend():
    llm = LocalExtractiveLLM()
    tokenizer = get_tokenizer("approximate")
    return RAGBackend(
        embedding_provider=HashingEmbedding(dim=64),
        vector_store=NumpyVectorStore(),
        llm=llm, tokenizer=tokenizer, default_top_k=5,
    )


def _full_context_backend():
    llm = LocalExtractiveLLM()
    tokenizer = get_tokenizer("approximate")
    return FullContextBackend(llm=llm, tokenizer=tokenizer, max_context_tokens=4000)


@pytest.mark.asyncio
async def test_runner_produces_one_trace_per_question():
    dataset = generate_dataset(num_users=2, num_turns=30, seed=42)
    runner = BenchmarkRunner(backend=_rag_backend(), run_id="test_run", seed=42, model="local-extractive-v1", top_k=5)
    traces = await runner.run(dataset)
    expected = sum(len(u.questions) for u in dataset.users)
    assert len(traces) == expected
    assert all(isinstance(t, Trace) for t in traces)


@pytest.mark.asyncio
async def test_runner_preserves_ingestion_order_per_user():
    dataset = generate_dataset(num_users=1, num_turns=30, seed=1)
    user = dataset.users[0]

    seen_order = []

    class OrderTrackingBackend(MemoryBackend):
        name = "order-tracker"

        async def add(self, user_id, messages):
            seen_order.extend(m.message_id for m in messages)
            return IngestResult(user_id=user_id, num_messages=len(messages), latency_ms=0.0)

        async def retrieve(self, user_id, query, k=5):
            return RetrievalResult(memories=[], retrieval_ms=0.0)

        async def answer(self, user_id, query, k=5):
            return Answer(text="n/a")

        async def reset(self, user_id):
            pass

    runner = BenchmarkRunner(backend=OrderTrackingBackend(), run_id="r", seed=1, model="m", top_k=5)
    await runner.run(dataset)

    expected_order = [m.message_id for m in user.messages]
    assert seen_order == expected_order


@pytest.mark.asyncio
async def test_runner_marks_retrieval_not_applicable_for_full_context():
    dataset = generate_dataset(num_users=1, num_turns=20, seed=3)
    runner = BenchmarkRunner(backend=_full_context_backend(), run_id="r", seed=3, model="m", top_k=5)
    traces = await runner.run(dataset)
    assert all(t.retrieval_applicable is False for t in traces)
    assert all(t.retrieval_metrics == {} for t in traces)


@pytest.mark.asyncio
async def test_runner_marks_retrieval_applicable_for_rag():
    dataset = generate_dataset(num_users=1, num_turns=20, seed=3)
    runner = BenchmarkRunner(backend=_rag_backend(), run_id="r", seed=3, model="m", top_k=5)
    traces = await runner.run(dataset)
    assert all(t.retrieval_applicable is True for t in traces)
    assert any(t.retrieval_metrics for t in traces)


@pytest.mark.asyncio
async def test_runner_handles_ingestion_failure_without_crashing():
    class FailingIngestBackend(MemoryBackend):
        name = "failing-ingest"

        async def add(self, user_id, messages):
            return IngestResult(user_id=user_id, num_messages=0, latency_ms=0.0, error="disk full")

        async def retrieve(self, user_id, query, k=5):
            return RetrievalResult(memories=[], retrieval_ms=0.0)

        async def answer(self, user_id, query, k=5):
            raise AssertionError("answer() should not be called after failed ingestion")

        async def reset(self, user_id):
            pass

    dataset = generate_dataset(num_users=1, num_turns=15, seed=5)
    runner = BenchmarkRunner(backend=FailingIngestBackend(), run_id="r", seed=5, model="m", top_k=5)
    traces = await runner.run(dataset)
    assert all(t.success is False for t in traces)
    assert all("disk full" in t.error for t in traces)


@pytest.mark.asyncio
async def test_runner_handles_answer_exception_without_crashing_whole_run():
    class ExplodingBackend(MemoryBackend):
        name = "exploding"

        async def add(self, user_id, messages):
            return IngestResult(user_id=user_id, num_messages=len(messages), latency_ms=0.0)

        async def retrieve(self, user_id, query, k=5):
            return RetrievalResult(memories=[], retrieval_ms=0.0)

        async def answer(self, user_id, query, k=5):
            raise RuntimeError("boom")

        async def reset(self, user_id):
            pass

    dataset = generate_dataset(num_users=1, num_turns=15, seed=5)
    runner = BenchmarkRunner(backend=ExplodingBackend(), run_id="r", seed=5, model="m", top_k=5)
    traces = await runner.run(dataset)
    assert len(traces) == len(dataset.users[0].questions)
    assert all(t.success is False and "boom" in t.error for t in traces)


@pytest.mark.asyncio
async def test_runner_respects_concurrency_bound_across_users():
    dataset = generate_dataset(num_users=6, num_turns=15, seed=8)
    in_flight = 0
    max_in_flight = 0

    class ConcurrencyTrackingBackend(MemoryBackend):
        name = "concurrency-tracker"

        async def add(self, user_id, messages):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            import asyncio
            await asyncio.sleep(0.01)
            in_flight -= 1
            return IngestResult(user_id=user_id, num_messages=len(messages), latency_ms=0.0)

        async def retrieve(self, user_id, query, k=5):
            return RetrievalResult(memories=[], retrieval_ms=0.0)

        async def answer(self, user_id, query, k=5):
            return Answer(text="ok")

        async def reset(self, user_id):
            pass

    runner = BenchmarkRunner(
        backend=ConcurrencyTrackingBackend(), run_id="r", seed=8, model="m", top_k=5, concurrency=2,
    )
    await runner.run(dataset)
    assert max_in_flight <= 2


@pytest.mark.asyncio
async def test_runner_traces_are_json_serializable_round_trip():
    dataset = generate_dataset(num_users=1, num_turns=20, seed=11)
    runner = BenchmarkRunner(backend=_rag_backend(), run_id="r", seed=11, model="local-extractive-v1", top_k=5)
    traces = await runner.run(dataset)
    for t in traces:
        payload = t.model_dump_json()
        restored = Trace.model_validate_json(payload)
        assert restored.query_id == t.query_id
