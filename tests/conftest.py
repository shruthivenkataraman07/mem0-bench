from __future__ import annotations

import pytest

from mem0_bench.datasets.generator import generate_dataset


@pytest.fixture
def small_dataset():
    return generate_dataset(num_users=3, num_turns=30, seed=42)


@pytest.fixture
def tiny_dataset():
    return generate_dataset(num_users=1, num_turns=15, seed=1)
