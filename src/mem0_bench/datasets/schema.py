"""Typed dataset schema for realistic multi-session benchmark users.

The schema deliberately separates:

- the raw conversation (``Message`` objects a memory backend ingests),
- the *ground truth* world state (``GoldMemory`` objects, produced by the
  dataset generator's structured world model, never by a backend under
  test), and
- the evaluation protocol (``EvalQuestion`` objects with pre-computed
  expected answers and expected relevant memory ids).

Keeping gold labels independent of any backend's own output is what makes
the benchmark's retrieval/answer metrics non-circular (see README section
"Benchmark validity").
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class QueryCategory(str, Enum):
    PREFERENCE = "preference"
    TEMPORAL = "temporal"
    ENTITY = "entity"
    MULTI_HOP = "multi_hop"
    CONTRADICTION = "contradiction"
    LONG_HORIZON = "long_horizon"
    DISTRACTOR = "distractor"


class Message(BaseModel):
    """A single utterance in a user's conversation history."""

    message_id: str
    session_id: str
    turn_index: int = Field(ge=0)
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {v!r}")
        return v


class GoldMemory(BaseModel):
    """A ground-truth fact derived from the generator's world state.

    ``superseded_by`` links an older fact to the memory_id of the fact that
    replaces it (used for contradiction/update questions), so evaluators can
    tell historical facts from currently-true facts.
    """

    memory_id: str
    content: str
    session_id: str
    message_id: str
    timestamp: datetime
    entity: Optional[str] = None
    fact_type: str
    superseded_by: Optional[str] = None
    is_current: bool = True


class EvalQuestion(BaseModel):
    """One evaluation query with pre-computed, generator-derived ground truth."""

    question_id: str
    category: QueryCategory
    query: str
    expected_answer: str
    expected_memory_ids: List[str]
    asked_after_session: str
    is_distractor_heavy: bool = False
    notes: Optional[str] = None


class UserDataset(BaseModel):
    """All conversation + evaluation data for a single simulated user."""

    user_id: str
    seed: int
    messages: List[Message]
    gold_memories: List[GoldMemory]
    questions: List[EvalQuestion]

    @property
    def sessions(self) -> List[str]:
        seen: List[str] = []
        for m in self.messages:
            if m.session_id not in seen:
                seen.append(m.session_id)
        return seen

    def gold_memory_by_id(self, memory_id: str) -> Optional[GoldMemory]:
        for g in self.gold_memories:
            if g.memory_id == memory_id:
                return g
        return None


class BenchmarkDataset(BaseModel):
    """A full benchmark dataset: many simulated users sharing one seed/version."""

    dataset_id: str
    version: str = "1.0"
    seed: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    users: List[UserDataset]

    @property
    def num_users(self) -> int:
        return len(self.users)

    @property
    def num_messages(self) -> int:
        return sum(len(u.messages) for u in self.users)

    @property
    def num_questions(self) -> int:
        return sum(len(u.questions) for u in self.users)
