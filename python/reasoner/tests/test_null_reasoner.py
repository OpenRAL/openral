"""Real-schema plumbing tests for :class:`NullReasoner`.

These tests construct **real** ``WorldState`` instances from
``openral_core`` (per CLAUDE.md §1.11 — no mocks). They exercise the
Protocol surface only; provider-backed reasoners get their own tests
when they land.
"""

from __future__ import annotations

import pytest
from openral_core import JointState, WorldState
from openral_reasoner import NullReasoner, Plan, Reasoner, ToolCall


def _world_state() -> WorldState:
    """Minimal real ``WorldState`` — no placeholders, no mocks."""
    return WorldState(
        stamp_ns=0,
        joint_state=JointState(name=["j1"], position=[0.0], stamp_ns=0),
    )


def test_null_reasoner_emits_single_leaf_plan() -> None:
    reasoner = NullReasoner(default_skill_id="pick_cube_so100")
    plan = reasoner.plan(_world_state(), goal="pick the red cube")

    assert isinstance(plan, Plan)
    assert plan.goal == "pick the red cube"
    assert len(plan.tool_calls) == 1
    assert plan.tool_calls[0].rskill_id == "pick_cube_so100"
    assert plan.confidence == 1.0
    assert plan.bt_xml is None


def test_null_reasoner_satisfies_reasoner_protocol() -> None:
    reasoner = NullReasoner()
    assert isinstance(reasoner, Reasoner)
    assert reasoner.plan_rate_hz == 5.0
    assert reasoner.client is None


def test_plan_rejects_empty_tool_calls() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        Plan(goal="g", tool_calls=[], confidence=0.5)


def test_plan_round_trips_through_json() -> None:
    """Schema round-trip per CLAUDE.md §5.4 (no mocks; real Pydantic v2)."""
    original = Plan(
        goal="pick the red cube",
        tool_calls=[ToolCall(rskill_id="pick_cube_so100", params={"color": "red"})],
        confidence=0.9,
    )
    restored = Plan.model_validate_json(original.model_dump_json())
    assert restored == original


def test_tool_call_forbids_extra_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ToolCall.model_validate({"rskill_id": "x", "unexpected": True})
