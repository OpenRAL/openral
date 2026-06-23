"""Tests for reasoner execution feedback + reflection (ADR-0071 Decision 2.2/2.3).

Covers:
- :func:`~openral_reasoner.context.reflect_on_failure` /
  :func:`~openral_reasoner.context.reflect_on_retry_cap` (deterministic hints).
- :class:`~openral_reasoner.context.ContextRenderer` ``## EXECUTION`` section:
  success + failure feedback, the Reflexion hint on failures, ``seq`` bump, and
  section ordering.

Run with:
    uv run pytest tests/unit/test_reasoner_execution_feedback.py -v
"""

from __future__ import annotations

from openral_reasoner.context import (
    ContextRenderer,
    ExecutionEventRecord,
    reflect_on_failure,
    reflect_on_retry_cap,
)


def test_reflect_on_failure_branches() -> None:
    assert "infeasible" in reflect_on_failure("aborted", "joint limit hit")
    assert "re-check" in reflect_on_failure("canceled", "operator cancel")
    assert "timed out" in reflect_on_failure("failed", "deadline exceeded after 5s")
    assert "timed out" in reflect_on_failure("error", "inference TIMEOUT")
    # Generic failure → "don't repeat the same call".
    assert "different skill" in reflect_on_failure("failed", "grasp slipped")


def test_reflect_on_retry_cap_names_tool_and_cap() -> None:
    hint = reflect_on_retry_cap("execute_rskill__grasp", 3)
    assert "execute_rskill__grasp" in hint
    assert "3+" in hint
    assert "exhausted" in hint


def test_execution_section_renders_success_and_failure() -> None:
    r = ContextRenderer()
    seq0 = r.seq
    r.append_execution(ExecutionEventRecord("grasp", "ok", "trace=abc12345", None, stamp_ns=1))
    r.append_execution(
        ExecutionEventRecord(
            "grasp",
            "failed",
            "object not in gripper",
            reflection=reflect_on_failure("failed", "object not in gripper"),
            stamp_ns=2,
        )
    )
    assert r.seq == seq0 + 2  # each completed skill is an event
    out = r.render(world_state=None)
    assert "## EXECUTION" in out
    assert "[ok] skill=grasp: trace=abc12345" in out
    assert "[failed] skill=grasp: object not in gripper — reflect:" in out
    # Snapshot accessor.
    assert len(r.executions) == 2 and r.executions[0].outcome == "ok"


def test_execution_section_ordered_after_world_state_before_failures() -> None:
    r = ContextRenderer()
    out = r.render(world_state=None)
    assert "## EXECUTION" in out
    assert out.index("## WORLD_STATE") < out.index("## EXECUTION") < out.index("## FAILURES")
    # Empty buffer renders the explicit placeholder.
    assert "## EXECUTION\n(none)" in out
