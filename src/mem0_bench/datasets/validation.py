"""Structural + referential integrity checks for a ``BenchmarkDataset``.

These catch the class of "benchmark validity" mistakes called out in the
README (e.g. a question pointing at a memory id that doesn't exist, two
users sharing message ids, non-chronological timestamps within a session).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

from mem0_bench.datasets.schema import BenchmarkDataset, UserDataset


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add(self, msg: str) -> None:
        self.errors.append(msg)


def validate_user(user: UserDataset) -> ValidationResult:
    result = ValidationResult()

    msg_ids = [m.message_id for m in user.messages]
    if len(msg_ids) != len(set(msg_ids)):
        result.add(f"[{user.user_id}] duplicate message_id detected")

    session_last_ts: Dict[str, datetime] = {}
    for m in user.messages:
        prev = session_last_ts.get(m.session_id)
        if prev is not None and m.timestamp < prev:
            result.add(
                f"[{user.user_id}] message {m.message_id} is out of chronological "
                f"order within session {m.session_id}"
            )
        session_last_ts[m.session_id] = m.timestamp

    msg_id_set = set(msg_ids)
    mem_ids = [g.memory_id for g in user.gold_memories]
    if len(mem_ids) != len(set(mem_ids)):
        result.add(f"[{user.user_id}] duplicate memory_id detected")
    mem_id_set = set(mem_ids)

    for g in user.gold_memories:
        if g.message_id not in msg_id_set:
            result.add(
                f"[{user.user_id}] gold memory {g.memory_id} references "
                f"unknown message_id {g.message_id}"
            )
        if g.superseded_by is not None and g.superseded_by not in mem_id_set:
            result.add(
                f"[{user.user_id}] gold memory {g.memory_id} superseded_by "
                f"references unknown memory_id {g.superseded_by}"
            )

    session_ids = set(user.sessions)
    for q in user.questions:
        for mem_id in q.expected_memory_ids:
            if mem_id not in mem_id_set:
                result.add(
                    f"[{user.user_id}] question {q.question_id} references "
                    f"unknown expected memory_id {mem_id}"
                )
        if q.asked_after_session not in session_ids:
            result.add(
                f"[{user.user_id}] question {q.question_id} references "
                f"unknown session {q.asked_after_session}"
            )
        if not q.expected_answer.strip():
            result.add(f"[{user.user_id}] question {q.question_id} has an empty expected_answer")

    return result


def validate_dataset(dataset: BenchmarkDataset) -> ValidationResult:
    result = ValidationResult()

    seen_user_ids = set()
    global_message_ids = set()
    global_memory_ids = set()

    for user in dataset.users:
        if user.user_id in seen_user_ids:
            result.add(f"duplicate user_id: {user.user_id}")
        seen_user_ids.add(user.user_id)

        user_result = validate_user(user)
        result.errors.extend(user_result.errors)

        for m in user.messages:
            if m.message_id in global_message_ids:
                result.add(f"message_id {m.message_id} collides across users")
            global_message_ids.add(m.message_id)

        for g in user.gold_memories:
            if g.memory_id in global_memory_ids:
                result.add(f"memory_id {g.memory_id} collides across users")
            global_memory_ids.add(g.memory_id)

    return result
