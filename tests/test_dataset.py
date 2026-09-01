import pytest

from mem0_bench.datasets.generator import generate_dataset
from mem0_bench.datasets.loader import load_dataset, save_dataset
from mem0_bench.datasets.schema import QueryCategory
from mem0_bench.datasets.validation import validate_dataset, validate_user


@pytest.mark.parametrize("turns", [1, 10, 25, 50, 100])
def test_generation_is_valid_across_scales(turns):
    ds = generate_dataset(num_users=2, num_turns=turns, seed=42)
    result = validate_dataset(ds)
    assert result.is_valid, result.errors


def test_generation_is_deterministic():
    ds1 = generate_dataset(num_users=3, num_turns=40, seed=99)
    ds2 = generate_dataset(num_users=3, num_turns=40, seed=99)
    assert ds1.model_dump(exclude={"generated_at"}) == ds2.model_dump(exclude={"generated_at"})


def test_different_seeds_produce_different_datasets():
    ds1 = generate_dataset(num_users=1, num_turns=40, seed=1)
    ds2 = generate_dataset(num_users=1, num_turns=40, seed=2)
    assert ds1.model_dump(exclude={"generated_at"}) != ds2.model_dump(exclude={"generated_at"})


def test_chronological_ingestion_order_within_session(small_dataset):
    for user in small_dataset.users:
        for session_id in user.sessions:
            timestamps = [m.timestamp for m in user.messages if m.session_id == session_id]
            assert timestamps == sorted(timestamps)


def test_no_cross_user_id_collisions(small_dataset):
    all_message_ids = set()
    all_memory_ids = set()
    for user in small_dataset.users:
        for m in user.messages:
            assert m.message_id not in all_message_ids
            all_message_ids.add(m.message_id)
        for g in user.gold_memories:
            assert g.memory_id not in all_memory_ids
            all_memory_ids.add(g.memory_id)


def test_contradiction_question_gold_answer_reflects_latest_fact(small_dataset):
    for user in small_dataset.users:
        contradiction_qs = [q for q in user.questions if q.category == QueryCategory.CONTRADICTION]
        for q in contradiction_qs:
            employer_facts = sorted(
                (g for g in user.gold_memories if g.fact_type == "employer"),
                key=lambda g: g.timestamp,
            )
            assert len(employer_facts) >= 2
            current = employer_facts[-1]
            assert current.is_current
            assert current.superseded_by is None
            # earlier facts must be marked as superseded, not current
            for older in employer_facts[:-1]:
                assert not older.is_current
                assert older.superseded_by is not None
            assert q.expected_memory_ids == [current.memory_id]


def test_every_question_expected_memory_ids_exist_in_gold(small_dataset):
    for user in small_dataset.users:
        gold_ids = {g.memory_id for g in user.gold_memories}
        for q in user.questions:
            for mid in q.expected_memory_ids:
                assert mid in gold_ids


def test_validate_user_flags_duplicate_message_id(small_dataset):
    user = small_dataset.users[0].model_copy(deep=True)
    dup = user.messages[0].model_copy(update={"message_id": user.messages[1].message_id})
    user.messages[0] = dup
    result = validate_user(user)
    assert not result.is_valid
    assert any("duplicate message_id" in e for e in result.errors)


def test_validate_user_flags_dangling_expected_memory_id(small_dataset):
    user = small_dataset.users[0].model_copy(deep=True)
    bad_question = user.questions[0].model_copy(update={"expected_memory_ids": ["does_not_exist"]})
    user.questions[0] = bad_question
    result = validate_user(user)
    assert not result.is_valid


def test_save_and_load_dataset_roundtrip(small_dataset, tmp_path):
    path = tmp_path / "ds.json"
    save_dataset(small_dataset, path)
    loaded = load_dataset(path)
    assert loaded.dataset_id == small_dataset.dataset_id
    assert loaded.num_messages == small_dataset.num_messages
    assert loaded.num_questions == small_dataset.num_questions


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does_not_exist.json")


def test_all_query_categories_appear_with_enough_turns():
    ds = generate_dataset(num_users=1, num_turns=80, seed=7)
    categories = {q.category for q in ds.users[0].questions}
    assert QueryCategory.PREFERENCE in categories
    assert QueryCategory.TEMPORAL in categories
    assert QueryCategory.ENTITY in categories
    assert QueryCategory.MULTI_HOP in categories
    assert QueryCategory.CONTRADICTION in categories
    assert QueryCategory.LONG_HORIZON in categories
    assert QueryCategory.DISTRACTOR in categories
