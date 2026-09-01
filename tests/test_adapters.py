from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pytest

from mem0_bench.datasets.schema import Message
from mem0_bench.llm.client import LocalExtractiveLLM
from mem0_bench.llm.tokenizer import get_tokenizer
from mem0_bench.memory.embeddings import HashingEmbedding
from mem0_bench.memory.full_context import FullContextBackend
from mem0_bench.memory.mem0_adapter import Mem0Adapter
from mem0_bench.memory.rag_adapter import RAGBackend
from mem0_bench.memory.vector_store import NumpyVectorStore

BASE_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _msg(user_id: str, session: str, idx: int, role: str, content: str) -> Message:
    return Message(
        message_id=f"{user_id}_{session}_{idx}", session_id=session, turn_index=idx,
        role=role, content=content, timestamp=BASE_TS + timedelta(minutes=idx),
    )


@pytest.fixture
def tokenizer():
    return get_tokenizer("approximate")


@pytest.fixture
def local_llm():
    return LocalExtractiveLLM()


# ---- RAG backend ----

@pytest.fixture
def rag_backend(local_llm, tokenizer):
    return RAGBackend(
        embedding_provider=HashingEmbedding(dim=64),
        vector_store=NumpyVectorStore(),
        llm=local_llm,
        tokenizer=tokenizer,
        default_top_k=5,
    )


@pytest.mark.asyncio
async def test_rag_add_and_retrieve_roundtrip(rag_backend):
    messages = [_msg("u1", "s1", 0, "user", "I work at Stripe."), _msg("u1", "s1", 1, "assistant", "Got it.")]
    result = await rag_backend.add("u1", messages)
    assert result.ok
    assert result.num_messages == 2

    retrieval = await rag_backend.retrieve("u1", "Where do I work?", k=5)
    assert retrieval.ok
    assert any("Stripe" in m.content for m in retrieval.memories)


@pytest.mark.asyncio
async def test_rag_enforces_user_isolation(rag_backend):
    await rag_backend.add("u1", [_msg("u1", "s1", 0, "user", "I work at Stripe.")])
    await rag_backend.add("u2", [_msg("u2", "s1", 0, "user", "I work at Google.")])

    retrieval_u1 = await rag_backend.retrieve("u1", "Where do I work?", k=5)
    assert all("Google" not in m.content for m in retrieval_u1.memories)

    retrieval_u2 = await rag_backend.retrieve("u2", "Where do I work?", k=5)
    assert all("Stripe" not in m.content for m in retrieval_u2.memories)


@pytest.mark.asyncio
async def test_rag_reset_clears_user_only(rag_backend):
    await rag_backend.add("u1", [_msg("u1", "s1", 0, "user", "I work at Stripe.")])
    await rag_backend.add("u2", [_msg("u2", "s1", 0, "user", "I work at Google.")])
    await rag_backend.reset("u1")

    retrieval_u1 = await rag_backend.retrieve("u1", "Where do I work?", k=5)
    assert retrieval_u1.memories == []
    retrieval_u2 = await rag_backend.retrieve("u2", "Where do I work?", k=5)
    assert len(retrieval_u2.memories) == 1


@pytest.mark.asyncio
async def test_rag_answer_produces_generation_and_timing(rag_backend):
    await rag_backend.add("u1", [_msg("u1", "s1", 0, "user", "I work at Stripe.")])
    answer = await rag_backend.answer("u1", "Where do I work?", k=5)
    assert answer.ok
    assert "Stripe" in answer.text
    assert answer.retrieval_ms >= 0
    assert answer.generation_ms >= 0
    assert answer.output_tokens > 0


@pytest.mark.asyncio
async def test_rag_only_embeds_user_messages_not_assistant_acks(rag_backend):
    messages = [_msg("u1", "s1", 0, "user", "I work at Stripe."), _msg("u1", "s1", 1, "assistant", "Noted!")]
    await rag_backend.add("u1", messages)
    retrieval = await rag_backend.retrieve("u1", "anything", k=10)
    assert all(m.metadata.get("source") == "conversation" for m in retrieval.memories)
    assert len(retrieval.memories) == 1  # only the user turn was chunked


# ---- Full-context backend ----

@pytest.fixture
def full_context_backend(local_llm, tokenizer):
    return FullContextBackend(llm=local_llm, tokenizer=tokenizer, max_context_tokens=50)


@pytest.mark.asyncio
async def test_full_context_stores_and_returns_all_history(full_context_backend):
    messages = [_msg("u1", "s1", i, "user", f"fact number {i}") for i in range(3)]
    await full_context_backend.add("u1", messages)
    retrieval = await full_context_backend.retrieve("u1", "anything", k=5)
    assert len(retrieval.memories) == 3


@pytest.mark.asyncio
async def test_full_context_truncation_is_recorded_not_silent(full_context_backend):
    # max_context_tokens=50 is small; push in enough turns to force truncation.
    messages = [_msg("u1", "s1", i, "user", f"this is fact number {i} about something specific") for i in range(30)]
    await full_context_backend.add("u1", messages)
    answer = await full_context_backend.answer("u1", "what facts do you know?", k=5)
    assert answer.ok
    assert answer.history_total_turns == 30
    assert answer.history_included_turns is not None
    assert answer.history_included_turns < 30
    assert answer.truncated is True
    assert answer.history_dropped_turns == 30 - answer.history_included_turns


@pytest.mark.asyncio
async def test_full_context_no_truncation_when_history_fits(full_context_backend):
    messages = [_msg("u1", "s1", 0, "user", "short fact")]
    await full_context_backend.add("u1", messages)
    answer = await full_context_backend.answer("u1", "what did I say?", k=5)
    assert answer.truncated is False
    assert answer.history_dropped_turns == 0


@pytest.mark.asyncio
async def test_full_context_reset_clears_history(full_context_backend):
    await full_context_backend.add("u1", [_msg("u1", "s1", 0, "user", "fact")])
    await full_context_backend.reset("u1")
    retrieval = await full_context_backend.retrieve("u1", "anything", k=5)
    assert retrieval.memories == []


# ---- Mem0 adapter (fake client injected; no network/API key needed) ----

class FakeMem0Client:
    """Implements the Mem0ClientProtocol without any network calls."""

    def __init__(self) -> None:
        self.storage: Dict[str, List[Dict]] = {}
        self.add_calls: List[Dict] = []

    def add(self, messages: List[Dict[str, str]], *, user_id: str, metadata: Dict) -> Dict:
        self.add_calls.append({"messages": messages, "user_id": user_id, "metadata": metadata})
        bucket = self.storage.setdefault(user_id, [])
        for m in messages:
            if m["role"] == "user":
                bucket.append({"id": f"mem_{len(bucket)}", "memory": m["content"], "score": 1.0})
        return {"results": []}

    def search(self, query: str, *, top_k: int, filters: Dict) -> Dict:
        user_id = filters["user_id"]
        results = self.storage.get(user_id, [])[:top_k]
        return {"results": results}

    def delete_all(self, user_id: str) -> None:
        self.storage.pop(user_id, None)


@pytest.fixture
def mem0_backend(local_llm, tokenizer):
    return Mem0Adapter(llm=local_llm, tokenizer=tokenizer, client=FakeMem0Client())


@pytest.mark.asyncio
async def test_mem0_adapter_add_and_search_roundtrip(mem0_backend):
    result = await mem0_backend.add("u1", [_msg("u1", "s1", 0, "user", "I work at Stripe.")])
    assert result.ok
    retrieval = await mem0_backend.retrieve("u1", "Where do I work?", k=5)
    assert retrieval.ok
    assert any("Stripe" in m.content for m in retrieval.memories)


@pytest.mark.asyncio
async def test_mem0_adapter_passes_session_metadata(mem0_backend):
    await mem0_backend.add("u1", [_msg("u1", "session_7", 0, "user", "hello")])
    call = mem0_backend.client.add_calls[0]
    assert call["metadata"]["session_id"] == "session_7"
    assert call["user_id"] == "u1"


@pytest.mark.asyncio
async def test_mem0_adapter_user_isolation(mem0_backend):
    await mem0_backend.add("u1", [_msg("u1", "s1", 0, "user", "I work at Stripe.")])
    await mem0_backend.add("u2", [_msg("u2", "s1", 0, "user", "I work at Google.")])
    retrieval = await mem0_backend.retrieve("u1", "Where do I work?", k=5)
    assert all("Google" not in m.content for m in retrieval.memories)


@pytest.mark.asyncio
async def test_mem0_adapter_reset(mem0_backend):
    await mem0_backend.add("u1", [_msg("u1", "s1", 0, "user", "I work at Stripe.")])
    await mem0_backend.reset("u1")
    retrieval = await mem0_backend.retrieve("u1", "Where do I work?", k=5)
    assert retrieval.memories == []


class BrokenMem0Client(FakeMem0Client):
    def search(self, query: str, *, top_k: int, filters: Dict) -> Dict:
        raise RuntimeError("simulated backend outage")


@pytest.mark.asyncio
async def test_mem0_adapter_surfaces_errors_without_raising(local_llm, tokenizer):
    backend = Mem0Adapter(llm=local_llm, tokenizer=tokenizer, client=BrokenMem0Client())
    retrieval = await backend.retrieve("u1", "Where do I work?", k=5)
    assert not retrieval.ok
    assert "simulated backend outage" in retrieval.error

    answer = await backend.answer("u1", "Where do I work?", k=5)
    assert not answer.ok
    assert answer.error is not None
