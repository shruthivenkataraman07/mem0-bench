"""The common memory backend contract.

Every architecture under test (Mem0, RAG, full-context, or a third-party
backend a user adds) implements :class:`MemoryBackend`. The
:class:`~mem0_bench.benchmark.runner.BenchmarkRunner` depends only on this
abstraction; it contains no backend-specific logic. See README, "Adding a
new memory backend" for how to extend this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field

from mem0_bench.datasets.schema import Message


class Memory(BaseModel):
    """One retrieved memory item, backend-agnostic."""

    memory_id: str
    content: str
    score: Optional[float] = None
    metadata: Dict = Field(default_factory=dict)


class IngestResult(BaseModel):
    user_id: str
    num_messages: int
    latency_ms: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class RetrievalResult(BaseModel):
    memories: List[Memory]
    retrieval_ms: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Answer(BaseModel):
    """The result of a full add-then-ask query, with per-stage instrumentation."""

    text: str
    retrieved_memories: List[Memory] = Field(default_factory=list)

    input_tokens: int = 0
    retrieved_tokens: int = 0
    output_tokens: int = 0

    retrieval_ms: float = 0.0
    prompt_construction_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    # Full-context-specific instrumentation (README section 5).
    history_total_turns: Optional[int] = None
    history_included_turns: Optional[int] = None
    history_dropped_turns: Optional[int] = None
    truncated: bool = False

    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class MemoryBackend(ABC):
    """Abstract interface every memory architecture under test implements."""

    name: ClassVar[str] = "base"

    @abstractmethod
    async def add(self, user_id: str, messages: List[Message]) -> IngestResult:
        """Ingest messages for a user. Callers MUST preserve chronological order."""
        ...

    @abstractmethod
    async def retrieve(self, user_id: str, query: str, k: int = 5) -> RetrievalResult:
        """Retrieve up to `k` relevant memories for a query.

        Backends without a meaningful notion of top-k retrieval (e.g.
        full-context) should return every turn used, with `score=None`.
        """
        ...

    @abstractmethod
    async def answer(self, user_id: str, query: str, k: int = 5) -> Answer:
        """Retrieve (if applicable) and generate a final natural-language answer."""
        ...

    @abstractmethod
    async def reset(self, user_id: str) -> None:
        """Delete all state held for a user. Used for per-user test isolation."""
        ...
