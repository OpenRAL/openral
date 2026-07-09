"""Unit slice for the execute_rskill prompt fallback.

``ExecuteRskillTool.prompt`` defaults to ``""`` (no ``min_length``), so the LLM
can dispatch a VLA with no task-conditioning text — SmolVLA then writes an empty
``observation["task"]`` and has no instruction. ``_resolve_execute_prompt`` falls
back to the active mission task's text (which *is* the instruction) in that case.
Pure helper, tested like ``_should_offer_subdivision`` / ``_may_subdivide_active``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("openral_msgs")

from openral_reasoner_ros.reasoner_node import _resolve_execute_prompt


def test_keeps_a_nonempty_llm_prompt() -> None:
    assert _resolve_execute_prompt("pick up the milk", "the active task text") == "pick up the milk"


def test_empty_prompt_falls_back_to_active_task_text() -> None:
    assert _resolve_execute_prompt("", "pick up the milk and place it in the basket") == (
        "pick up the milk and place it in the basket"
    )


def test_whitespace_prompt_falls_back_to_active_task_text() -> None:
    assert _resolve_execute_prompt("   ", "pick up the ketchup") == "pick up the ketchup"


def test_empty_prompt_and_no_active_task_yields_empty() -> None:
    # No task to borrow from → empty (the runner/manifest default still applies).
    assert _resolve_execute_prompt("", None) == ""
