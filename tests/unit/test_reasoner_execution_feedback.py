"""Tests for reasoner execution feedback + reflection.

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

from openral_core import RSkillAction
from openral_reasoner.context import (
    ContextRenderer,
    ExecutionEventRecord,
    reflect_on_failure,
    reflect_on_retry_cap,
    reflect_on_reward_plateau,
)
from openral_reasoner.palette import RSkillToolEntry, ToolPalette


def test_reflect_on_failure_branches() -> None:
    assert "infeasible" in reflect_on_failure("aborted", "joint limit hit")
    assert "re-check" in reflect_on_failure("canceled", "operator cancel")
    assert "timed out" in reflect_on_failure("failed", "deadline exceeded after 5s")
    assert "timed out" in reflect_on_failure("error", "inference TIMEOUT")
    # Generic failure → "don't repeat the same call".
    assert "different skill" in reflect_on_failure("failed", "grasp slipped")


def test_reflect_on_failure_prefers_the_typed_timed_out_flag() -> None:
    """``timed_out`` (from the typed failure_kind) overrides the prose probe.

    The default ``None`` keeps the deprecated substring path for callers with no
    typed kind; an explicit flag wins in BOTH directions, so a reason that merely
    mentions a deadline no longer forces the timeout hint.
    """
    # Explicit True on prose that says nothing about time.
    assert "timed out" in reflect_on_failure("aborted", "joint limit hit", timed_out=True)
    # Explicit False on prose that would have tripped the substring probe.
    hint = reflect_on_failure("aborted", "deadline for the grasp was tight", timed_out=False)
    assert "timed out" not in hint
    assert "infeasible" in hint


def test_reflect_on_reward_plateau_says_change_approach_not_shorten() -> None:
    """A reward-plateau (policy ran, reward says not done) must nudge a
    DIFFERENT approach, not a shorter-horizon subdivide (the timeout hint) or a
    blind repeat. A direct replanning probe showed those two wrong signals produce
    repeat/subdivide; this one produces a genuine replan."""
    hint = reflect_on_reward_plateau(0.48)
    assert "0.48" in hint  # carries the evidence
    assert "different approach" in hint.lower()
    assert "do not" in hint.lower() and "same instruction" in hint.lower()
    # must NOT recommend the wrong moves for a clean-execution reward failure
    assert "shorter-horizon" not in hint.lower()
    assert "subdivide" not in hint.lower() or "not " in hint.lower()


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


def _two_skill_palette() -> ToolPalette:
    """A real two-entry palette: one skill that will break, one that won't."""
    return ToolPalette(
        skills=(
            RSkillToolEntry(
                rskill_id="broken",
                description="broken sidecar",
                actions=(RSkillAction.PICK,),
            ),
            RSkillToolEntry(
                rskill_id="healthy",
                description="healthy policy",
                actions=(RSkillAction.PICK,),
            ),
        )
    )


def test_permanent_skill_failure_removes_only_that_skill() -> None:
    """A broken sidecar is not offered again, but runtime failures remain retryable.

    Classification is on the typed ``failure_kind`` uint8 now, not the prose.
    """
    import pytest

    pytest.importorskip("rclpy")
    msgs = pytest.importorskip("openral_msgs.action")
    from openral_reasoner_ros.reasoner_node import _palette_after_rskill_failure

    result_cls = msgs.ExecuteRskill.Result
    palette = _two_skill_palette()

    for permanent in (result_cls.FAILURE_CONFIG_ERROR, result_cls.FAILURE_CAPABILITY_MISMATCH):
        filtered = _palette_after_rskill_failure(palette, "broken", permanent)
        assert filtered.execute_rskill_ids == frozenset({"healthy"})
        assert tuple(skill.rskill_id for skill in filtered.skills) == ("healthy",)

    # Retryable kinds — and the "unclassified" FAILURE_NONE — leave the palette alone.
    for retryable in (
        result_cls.FAILURE_RUNTIME_ERROR,
        result_cls.FAILURE_DEADLINE_MISSED,
        result_cls.FAILURE_SAFETY_ESTOP,
        result_cls.FAILURE_PERCEPTION_STALE,
        result_cls.FAILURE_PLANNING_ERROR,
        result_cls.FAILURE_CANCELLED,
        result_cls.FAILURE_UNKNOWN,
        result_cls.FAILURE_NONE,
    ):
        assert _palette_after_rskill_failure(palette, "broken", retryable) == palette


def test_rskill_failure_kind_reads_the_typed_field() -> None:
    """A real ``ExecuteRskill.Result`` is classified by its uint8, not its prose."""
    import pytest

    pytest.importorskip("rclpy")
    msgs = pytest.importorskip("openral_msgs.action")
    from openral_reasoner_ros.reasoner_node import _rskill_failure_kind

    result_cls = msgs.ExecuteRskill.Result
    result = result_cls()
    result.success = False
    result.failure_kind = result_cls.FAILURE_CAPABILITY_MISMATCH
    # Prose that would have been classified differently by the old prefix match.
    result.failure_reason = "ROSConfigError: this string must not win"
    assert _rskill_failure_kind(result) == result_cls.FAILURE_CAPABILITY_MISMATCH


def test_rskill_failure_kind_falls_back_to_prose_for_old_results() -> None:
    """A pre-``failure_kind`` producer (uint8 left at 0) is still classified.

    Deprecated-in-place path: a runner built against the older IDL sends
    ``failure_kind == FAILURE_NONE`` on a failed result, so the reasoner reads
    the typed prefix the runner has always written onto ``failure_reason``.
    """
    import pytest

    pytest.importorskip("rclpy")
    msgs = pytest.importorskip("openral_msgs.action")
    from openral_reasoner_ros.reasoner_node import _rskill_failure_kind

    result_cls = msgs.ExecuteRskill.Result
    cases = {
        "ROSConfigError: rskill.yaml missing": result_cls.FAILURE_CONFIG_ERROR,
        "ROSCapabilityMismatch: embodiment_tags disjoint": (result_cls.FAILURE_CAPABILITY_MISMATCH),
        "safety_estop:/openral/estop received during goal": result_cls.FAILURE_SAFETY_ESTOP,
        "deadline_exceeded: elapsed=144.5s budget=45.0s": result_cls.FAILURE_DEADLINE_MISSED,
        "cancelled": result_cls.FAILURE_CANCELLED,
        "ROSPerceptionStale: policy_state is stale": result_cls.FAILURE_PERCEPTION_STALE,
        "ROSPlanningError: MoveIt approach failed": result_cls.FAILURE_PLANNING_ERROR,
        "ROSGPUMemoryError: CUDA out of memory": result_cls.FAILURE_RUNTIME_ERROR,
        "ValueError: policy head returned nothing": result_cls.FAILURE_UNKNOWN,
    }
    for reason, expected in cases.items():
        result = result_cls()
        result.success = False
        result.failure_reason = reason
        assert _rskill_failure_kind(result) == expected, reason

    # A terminal result with neither a kind nor a reason (rclpy's default abort)
    # stays unclassified rather than being invented into a kind.
    empty = result_cls()
    empty.success = False
    assert _rskill_failure_kind(empty) == result_cls.FAILURE_NONE
