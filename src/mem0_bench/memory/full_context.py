"""Full-context baseline: no retrieval, just the complete conversation history
passed to the answer-generation model on every query.

This is the naive alternative to a memory system. It is included so the
benchmark can identify the crossover point where "just paste everything in
the prompt" stops being viable (see README, "Scale experiments"):
latency, cost, and eventually context-window limits all degrade as history
length grows, while retrieval-based approaches stay roughly flat.
"""

from __future__ import annotations

import time
from typing import Dict, List

from mem0_bench.datasets.schema import Message
from mem0_bench.llm.client import LLMClient
from mem0_bench.llm.tokenizer import Tokenizer
from mem0_bench.memory.base import Answer, IngestResult, Memory, MemoryBackend, RetrievalResult
from mem0_bench.memory.registry import register_backend

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "conversation history provided below. Be concise: answer with the "
    "specific fact requested, not a full sentence restating the question."
)


def _format_turn(m: Message) -> str:
    return f"{m.role}: {m.content}"


@register_backend("full-context")
class FullContextBackend(MemoryBackend):
    """Stores complete history per user; never truncates silently."""

    def __init__(
        self,
        llm: LLMClient,
        tokenizer: Tokenizer,
        max_context_tokens: int = 8000,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self._history: Dict[str, List[Message]] = {}

    async def add(self, user_id: str, messages: List[Message]) -> IngestResult:
        start = time.monotonic()
        self._history.setdefault(user_id, []).extend(messages)
        latency_ms = (time.monotonic() - start) * 1000
        return IngestResult(user_id=user_id, num_messages=len(messages), latency_ms=latency_ms)

    async def retrieve(self, user_id: str, query: str, k: int = 5) -> RetrievalResult:
        # Full-context has no retrieval step; it always uses the whole
        # history. Reported here (with score=None) only so callers that want
        # to inspect "what did the backend use" have something to look at;
        # the evaluator explicitly excludes full-context from retrieval
        # metrics (see benchmark/evaluator.py) rather than treating this as
        # a k-limited retrieval result.
        history = self._history.get(user_id, [])
        start = time.monotonic()
        memories = [
            Memory(memory_id=m.message_id, content=m.content, score=None,
                   metadata={"session_id": m.session_id, "role": m.role})
            for m in history
        ]
        return RetrievalResult(memories=memories, retrieval_ms=(time.monotonic() - start) * 1000)

    async def answer(self, user_id: str, query: str, k: int = 5) -> Answer:
        history = self._history.get(user_id, [])
        total_turns = len(history)

        t0 = time.monotonic()
        # Greedily include the MOST RECENT turns first, within the token
        # budget, then restore chronological order. This is the standard
        # "recency window" truncation strategy for full-context prompting.
        included: List[Message] = []
        used_tokens = 0
        for m in reversed(history):
            turn_tokens = self.tokenizer.count(_format_turn(m))
            if used_tokens + turn_tokens > self.max_context_tokens:
                break
            included.append(m)
            used_tokens += turn_tokens
        included.reverse()
        dropped = total_turns - len(included)
        truncated = dropped > 0

        history_text = "\n".join(_format_turn(m) for m in included)
        prompt = f"Conversation history:\n{history_text}\n\nQuestion: {query}"
        t1 = time.monotonic()
        prompt_ms = (t1 - t0) * 1000

        try:
            response = await self.llm.generate(prompt, system=SYSTEM_PROMPT)
        except Exception as e:  # noqa: BLE001 - surfaced via Answer.error, not swallowed
            total_ms = (time.monotonic() - t0) * 1000
            return Answer(
                text="", retrieval_ms=0.0, prompt_construction_ms=prompt_ms,
                generation_ms=total_ms - prompt_ms, total_ms=total_ms,
                history_total_turns=total_turns, history_included_turns=len(included),
                history_dropped_turns=dropped, truncated=truncated, error=str(e),
            )
        t2 = time.monotonic()

        return Answer(
            text=response.text,
            retrieved_memories=[],
            input_tokens=response.input_tokens,
            retrieved_tokens=0,
            output_tokens=response.output_tokens,
            retrieval_ms=0.0,
            prompt_construction_ms=prompt_ms,
            generation_ms=(t2 - t1) * 1000,
            total_ms=(t2 - t0) * 1000,
            history_total_turns=total_turns,
            history_included_turns=len(included),
            history_dropped_turns=dropped,
            truncated=truncated,
        )

    async def reset(self, user_id: str) -> None:
        self._history.pop(user_id, None)
