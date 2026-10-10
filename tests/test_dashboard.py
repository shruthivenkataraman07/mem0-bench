"""Offline regression tests for dashboard summary classification."""

import ast
from pathlib import Path
from typing import Any, Dict

import pytest


def _load_is_attack_summary():
    """Load the helper without importing the Streamlit application."""
    app_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "is_attack_summary"
    )
    namespace = {"Any": Any, "Dict": Dict}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(app_path), "exec"), namespace)
    return namespace["is_attack_summary"]


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ({}, False),
        ({"attack": {}}, False),
        ({"attack": {"attack_success_rate": 1.0}}, False),
        ({"attack": {"poison_retrieval_rate": 0.0}, "backend": {}}, False),
        ({"attack": {"poison_retrieval_rate": 0.0}}, True),
        ({"attack": {"poison_retrieval_rate": None}}, True),
    ],
)
def test_is_attack_summary_requires_every_nonempty_entry_to_have_attack_marker(summary, expected):
    assert _load_is_attack_summary()(summary) is expected
