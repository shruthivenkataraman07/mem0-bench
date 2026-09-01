"""Deterministic synthetic multi-session conversation generator.

Given a seed, this module always produces byte-identical ``BenchmarkDataset``
objects. Gold memories and expected answers are derived directly from the
generator's own structured "world state" for each simulated user, never
from what any memory backend under test retrieves or says. This is what
keeps the benchmark's evaluation non-circular (see README, "Benchmark
validity").
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from mem0_bench.datasets.schema import (
    BenchmarkDataset,
    EvalQuestion,
    GoldMemory,
    Message,
    QueryCategory,
    UserDataset,
)

EMPLOYERS = [
    "Google", "Stripe", "Meta", "Netflix", "Airbnb", "Shopify",
    "Snowflake", "Databricks", "Figma", "Notion", "Anthropic", "Cloudflare",
]
CITIES = [
    "Austin", "Seattle", "New York", "Boston", "Denver", "Chicago",
    "Toronto", "Berlin", "London", "Amsterdam", "Portland", "Vancouver",
]
FOODS = [
    "Thai", "Italian", "Mexican", "Japanese", "Ethiopian",
    "Vietnamese", "Mediterranean", "Korean",
]
HOBBIES = [
    "rock climbing", "pottery", "chess", "distance running",
    "woodworking", "astrophotography", "sailing", "bouldering",
]
PROJECTS = [
    "Atlas", "Nimbus", "Falcon", "Orion", "Meridian", "Vertex", "Comet", "Helios",
]
ROLES = [
    "software engineer", "product manager", "data scientist",
    "designer", "ML engineer", "engineering manager",
]
COLLEAGUES = [
    "Priya", "Marcus", "Elena", "Devon", "Sofia", "Kenji", "Amara", "Liam",
]
GOALS = [
    "run a marathon", "launch a side project", "learn Japanese",
    "get a pilot's license", "write a novel", "start a garden",
]
FILLER_TOPICS = [
    "the weather has been unpredictable lately",
    "I finally organized my desk this week",
    "traffic was brutal this morning",
    "I've been meaning to read more",
    "my phone battery has been dying fast",
    "I rewatched an old favorite movie last night",
    "the coffee shop near me changed its menu",
    "I've been trying to sleep earlier",
]

_BASE_DATE = datetime(2024, 1, 8, 9, 0, tzinfo=timezone.utc)


class _IdGen:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._msg_counter = 0
        self._mem_counter = 0
        self._q_counter = 0

    def message_id(self) -> str:
        self._msg_counter += 1
        return f"{self.user_id}_msg_{self._msg_counter:04d}"

    def memory_id(self) -> str:
        self._mem_counter += 1
        return f"{self.user_id}_mem_{self._mem_counter:04d}"

    def question_id(self) -> str:
        self._q_counter += 1
        return f"{self.user_id}_q_{self._q_counter:03d}"


class _Fact:
    """One structured world-state fact, tied to the message that states it."""

    def __init__(self, fact_type: str, value: str, entity: Optional[str] = None):
        self.fact_type = fact_type
        self.value = value
        self.entity = entity
        self.session_idx: int = -1
        self.message: Optional[Message] = None
        self.memory: Optional[GoldMemory] = None


def _session_id(user_id: str, idx: int) -> str:
    return f"{user_id}_session_{idx:03d}"


def _pick_session_indices(rng: random.Random, count: int, num_sessions: int) -> List[int]:
    """Pick `count` strictly increasing session indices spread across the timeline."""
    if num_sessions <= 1:
        return [0] * count
    if count == 1:
        return [rng.randint(0, num_sessions - 1)]
    # Spread roughly evenly with jitter, then sort + dedupe-forward to stay increasing.
    raw = sorted(rng.sample(range(num_sessions), k=min(count, num_sessions)))
    while len(raw) < count:
        raw.append(raw[-1])
    return raw


def _plan_sessions(rng: random.Random, num_turns: int) -> List[int]:
    """Split `num_turns` user turns into a list of per-session turn counts."""
    sessions: List[int] = []
    remaining = num_turns
    while remaining > 0:
        size = min(remaining, rng.randint(3, 7))
        sessions.append(size)
        remaining -= size
    return sessions or [num_turns]


def _generate_user(user_index: int, num_turns: int, seed: int) -> UserDataset:
    user_id = f"user_{user_index:03d}"
    # A single deterministic int (not a tuple) avoids Python's deprecated
    # hash-based seeding path and keeps generation stable across versions.
    rng = random.Random(seed * 1_000_003 + user_index)
    ids = _IdGen(user_id)

    session_sizes = _plan_sessions(rng, num_turns)
    num_sessions = len(session_sizes)

    # ---- Build structured world-state fact chains ----
    n_employer = 2 if num_turns >= 15 else 1
    n_city = 2 if num_turns >= 40 else 1
    n_projects = 2 if num_turns >= 60 else 1
    project_has_followup = num_turns >= 25

    employer_names = rng.sample(EMPLOYERS, n_employer)
    role_names = [rng.choice(ROLES) for _ in employer_names]
    employer_facts = [_Fact("employer", e) for e in employer_names]
    role_facts = [_Fact("role", r) for r in role_names]

    city_names = rng.sample(CITIES, n_city)
    city_facts = [_Fact("city", c) for c in city_names]

    food = rng.choice(FOODS)
    food_fact = _Fact("food_preference", food)

    hobby = rng.choice(HOBBIES)
    hobby_fact = _Fact("hobby", hobby)

    goal = rng.choice(GOALS)
    goal_fact = _Fact("goal", goal)

    project_names = rng.sample(PROJECTS, n_projects)
    project_facts: List[_Fact] = []
    for pname in project_names:
        intro = _Fact("project_intro", pname, entity=pname)
        project_facts.append(intro)
        if project_has_followup:
            colleague = rng.choice(COLLEAGUES)
            followup = _Fact("project_update", pname, entity=pname)
            followup.value = f"{pname}|{colleague}|shipped the first milestone"
            project_facts.append(followup)

    # Assign session indices, respecting chronological ordering per chain.
    for facts in (employer_facts, role_facts, city_facts):
        idxs = _pick_session_indices(rng, len(facts), num_sessions)
        for f, i in zip(facts, idxs):
            f.session_idx = i
    # role changes happen alongside employer changes
    for rf, ef in zip(role_facts, employer_facts):
        rf.session_idx = ef.session_idx

    food_fact.session_idx = 0
    hobby_fact.session_idx = min(1, num_sessions - 1)
    goal_fact.session_idx = _pick_session_indices(rng, 1, num_sessions)[0]

    for pname in project_names:
        intro = next(f for f in project_facts if f.entity == pname and f.fact_type == "project_intro")
        intro.session_idx = _pick_session_indices(rng, 1, num_sessions)[0]
        followups = [f for f in project_facts if f.entity == pname and f.fact_type == "project_update"]
        for fu in followups:
            fu.session_idx = min(num_sessions - 1, intro.session_idx + max(1, num_sessions // 3))

    # Distractor facts: off-hand mentions that must NOT be treated as the user's own current fact.
    distractor_facts: List[_Fact] = []
    n_distractors = min(6, max(2, num_turns // 15))
    for _ in range(n_distractors):
        kind = rng.choice(["employer", "food"])
        if kind == "employer":
            other = rng.choice([e for e in EMPLOYERS if e not in employer_names])
            df = _Fact("distractor_employer", other)
        else:
            other_food = rng.choice([f for f in FOODS if f != food])
            df = _Fact("distractor_food", other_food)
        df.session_idx = rng.randint(0, num_sessions - 1)
        distractor_facts.append(df)

    all_fact_chains: List[List[_Fact]] = [
        employer_facts, role_facts, city_facts,
        [food_fact], [hobby_fact], [goal_fact], project_facts, distractor_facts,
    ]

    # ---- Render sessions into messages, interleaving facts + filler ----
    messages: List[Message] = []
    facts_by_session: Dict[int, List[_Fact]] = {i: [] for i in range(num_sessions)}
    for chain in all_fact_chains:
        for f in chain:
            facts_by_session[f.session_idx].append(f)

    ts = _BASE_DATE
    for s_idx, size in enumerate(session_sizes):
        sess_id = _session_id(user_id, s_idx)
        session_facts = facts_by_session[s_idx]
        rng.shuffle(session_facts)
        # A session must render every fact assigned to it (never drop a fact
        # silently); if more facts landed here than the planned turn budget,
        # the session simply grows to fit them.
        n_fact_msgs = len(session_facts)
        n_filler = max(0, size - n_fact_msgs)
        slot_is_fact = [True] * n_fact_msgs + [False] * n_filler
        rng.shuffle(slot_is_fact)

        fact_cursor = 0
        turn_idx = 0
        for is_fact in slot_is_fact:
            if is_fact:
                fact = session_facts[fact_cursor]
                fact_cursor += 1
                content = _render_fact_sentence(fact)
            else:
                content = rng.choice(FILLER_TOPICS)

            user_msg = Message(
                message_id=ids.message_id(),
                session_id=sess_id,
                turn_index=turn_idx,
                role="user",
                content=content,
                timestamp=ts,
            )
            messages.append(user_msg)
            if is_fact:
                fact.message = user_msg
            ts += timedelta(minutes=2)
            turn_idx += 1

            ack = Message(
                message_id=ids.message_id(),
                session_id=sess_id,
                turn_index=turn_idx,
                role="assistant",
                content=_ack_sentence(rng),
                timestamp=ts,
            )
            messages.append(ack)
            ts += timedelta(minutes=1)
            turn_idx += 1

        ts += timedelta(days=7)  # next session, a week later

    # ---- Build gold memories from fact chains ----
    gold_memories: List[GoldMemory] = []

    def _add_chain_memories(chain: List[_Fact], fact_type: str, superseding: bool) -> List[GoldMemory]:
        chain_sorted = sorted(chain, key=lambda f: f.session_idx)
        made: List[GoldMemory] = []
        for i, f in enumerate(chain_sorted):
            assert f.message is not None
            mem = GoldMemory(
                memory_id=ids.memory_id(),
                content=_render_fact_sentence(f),
                session_id=f.message.session_id,
                message_id=f.message.message_id,
                timestamp=f.message.timestamp,
                entity=f.entity,
                fact_type=fact_type,
                is_current=(i == len(chain_sorted) - 1),
            )
            f.memory = mem
            made.append(mem)
        for i in range(len(made) - 1):
            made[i].superseded_by = made[i + 1].memory_id
        gold_memories.extend(made)
        return made

    employer_mem = _add_chain_memories(employer_facts, "employer", True)
    role_mem = _add_chain_memories(role_facts, "role", True)
    city_mem = _add_chain_memories(city_facts, "city", True)
    food_mem = _add_chain_memories([food_fact], "food_preference", False)
    hobby_mem = _add_chain_memories([hobby_fact], "hobby", False)
    _add_chain_memories([goal_fact], "goal", False)

    project_mem_by_name: Dict[str, List[GoldMemory]] = {}
    for pname in project_names:
        chain = [f for f in project_facts if f.entity == pname]
        made = _add_chain_memories(chain, "project", False)
        project_mem_by_name[pname] = made

    for df in distractor_facts:
        assert df.message is not None
        gold_memories.append(GoldMemory(
            memory_id=ids.memory_id(),
            content=_render_fact_sentence(df),
            session_id=df.message.session_id,
            message_id=df.message.message_id,
            timestamp=df.message.timestamp,
            entity=None,
            fact_type=df.fact_type,
            is_current=False,
        ))

    # ---- Build evaluation questions from the (now-final) world state ----
    questions: List[EvalQuestion] = []
    last_session = _session_id(user_id, num_sessions - 1)

    questions.append(EvalQuestion(
        question_id=ids.question_id(),
        category=QueryCategory.PREFERENCE,
        query="What kind of food do I prefer?",
        expected_answer=food,
        expected_memory_ids=[food_mem[0].memory_id],
        asked_after_session=last_session,
    ))

    if len(city_mem) >= 2:
        questions.append(EvalQuestion(
            question_id=ids.question_id(),
            category=QueryCategory.TEMPORAL,
            query="What city did I live in before my current city?",
            expected_answer=city_names[-2],
            expected_memory_ids=[city_mem[-2].memory_id],
            asked_after_session=last_session,
            notes="Requires distinguishing a superseded fact from the current one.",
        ))
    else:
        questions.append(EvalQuestion(
            question_id=ids.question_id(),
            category=QueryCategory.TEMPORAL,
            query="What city do I currently live in?",
            expected_answer=city_names[-1],
            expected_memory_ids=[city_mem[-1].memory_id],
            asked_after_session=last_session,
        ))

    for pname, mems in project_mem_by_name.items():
        questions.append(EvalQuestion(
            question_id=ids.question_id(),
            category=QueryCategory.ENTITY,
            query=f"What did I tell you about Project {pname}?",
            expected_answer=_project_summary(pname, project_facts),
            expected_memory_ids=[m.memory_id for m in mems],
            asked_after_session=last_session,
        ))

    questions.append(EvalQuestion(
        question_id=ids.question_id(),
        category=QueryCategory.MULTI_HOP,
        query="What is my current role and where do I currently work?",
        expected_answer=f"{role_names[-1]} at {employer_names[-1]}",
        expected_memory_ids=[role_mem[-1].memory_id, employer_mem[-1].memory_id],
        asked_after_session=last_session,
        notes="Requires combining two independent memories.",
    ))

    if len(employer_mem) >= 2:
        questions.append(EvalQuestion(
            question_id=ids.question_id(),
            category=QueryCategory.CONTRADICTION,
            query="Where do I currently work?",
            expected_answer=employer_names[-1],
            expected_memory_ids=[employer_mem[-1].memory_id],
            asked_after_session=last_session,
            notes=f"Supersedes earlier employer(s): {', '.join(employer_names[:-1])}.",
        ))

    questions.append(EvalQuestion(
        question_id=ids.question_id(),
        category=QueryCategory.LONG_HORIZON,
        query="What hobby did I mention taking up?",
        expected_answer=hobby,
        expected_memory_ids=[hobby_mem[0].memory_id],
        asked_after_session=last_session,
        notes=f"Stated in session 0 of {num_sessions}; asked at the end of the history.",
    ))

    questions.append(EvalQuestion(
        question_id=ids.question_id(),
        category=QueryCategory.DISTRACTOR,
        query="What is my own favorite type of cuisine (not a friend's or something I merely tried)?",
        expected_answer=food,
        expected_memory_ids=[food_mem[0].memory_id],
        asked_after_session=last_session,
        is_distractor_heavy=True,
        notes="Conversation contains several unrelated food/employer mentions as distractors.",
    ))

    return UserDataset(
        user_id=user_id,
        seed=seed,
        messages=messages,
        gold_memories=gold_memories,
        questions=questions,
    )


def _project_summary(pname: str, project_facts: List[_Fact]) -> str:
    updates = [f for f in project_facts if f.entity == pname and f.fact_type == "project_update"]
    if updates:
        _, colleague, status = updates[-1].value.split("|")
        return f"Working on Project {pname} with {colleague}, who {status}"
    return f"Working on Project {pname}"


def _render_fact_sentence(fact: _Fact) -> str:
    ft = fact.fact_type
    v = fact.value
    if ft == "employer":
        return f"I work at {v}."
    if ft == "role":
        return f"My role is {v}."
    if ft == "city":
        return f"I live in {v} now."
    if ft == "food_preference":
        return f"My favorite type of food is {v}."
    if ft == "hobby":
        return f"I recently took up {v} as a hobby."
    if ft == "goal":
        return f"One of my goals this year is to {v}."
    if ft == "project_intro":
        return f"I started working on a new project called {v}."
    if ft == "project_update":
        pname, colleague, status = v.split("|")
        return f"Quick update on Project {pname}: {colleague} and I {status}."
    if ft == "distractor_employer":
        return f"A friend of mine just joined {v}."
    if ft == "distractor_food":
        return f"I tried {v} food last night for the first time, it was okay."
    raise ValueError(f"unknown fact_type: {ft}")


def _ack_sentence(rng: random.Random) -> str:
    return rng.choice([
        "Got it, thanks for sharing.",
        "That's good to know.",
        "Noted!",
        "Thanks for the update.",
        "Makes sense.",
    ])


def generate_dataset(
    num_users: int,
    num_turns: int,
    seed: int,
    dataset_id: Optional[str] = None,
) -> BenchmarkDataset:
    """Deterministically generate a full benchmark dataset.

    Calling this twice with the same arguments always returns an identical
    dataset (aside from ``generated_at``, which is metadata, not data).
    """
    if num_users < 1:
        raise ValueError("num_users must be >= 1")
    if num_turns < 1:
        raise ValueError("num_turns must be >= 1")

    users = [_generate_user(i, num_turns, seed) for i in range(num_users)]
    return BenchmarkDataset(
        dataset_id=dataset_id or f"synthetic_u{num_users}_t{num_turns}_s{seed}",
        seed=seed,
        users=users,
    )
