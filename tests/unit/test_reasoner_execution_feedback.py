"""Tests for reasoner execution feedback + reflection.

Covers:
- ``reflect_on_failure`` /
  ``reflect_on_retry_cap`` (deterministic hints).
- ``ContextRenderer`` ``## EXECUTION`` section:
  success + failure feedback, the Reflexion hint on failures, ``seq`` bump, and
  section ordering.
- Issue #353: ``reflect_on_grasp_miss`` and the per-hand ``[grasp_miss]`` line a
  ``KIND_GRASP_OUTCOME`` miss opens and that hand's next ATTACH closes, on the real
  bimanual OpenArm manifest's hands.

Run with:
    uv run pytest tests/unit/test_reasoner_execution_feedback.py -v
"""

from __future__ import annotations

from typing import Literal

from openral_core import GraspOutcomeEvidence, RobotDescription, RSkillAction
from openral_observability.failure_bus import (
    KIND_GRASP_OUTCOME,
    SEVERITY_FAIL,
    SEVERITY_INFO,
    SEVERITY_WARN,
)
from openral_reasoner.context import (
    ContextRenderer,
    ExecutionEventRecord,
    FailureEventRecord,
    reflect_on_failure,
    reflect_on_grasp_miss,
    reflect_on_retry_cap,
    reflect_on_reward_plateau,
)
from openral_reasoner.mission import MissionState
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


def _openarm_hands() -> list[tuple[str, str, str]]:
    """``(gripper joint, jaw link, attach link)`` per hand of the real bimanual OpenArm."""
    description = RobotDescription.from_yaml("robots/openarm/robot.yaml")
    hands = [
        (joint.name, joint.child_link, joint.parent_link)
        for joint in description.joints
        if joint.role == "gripper"
    ]
    assert len(hands) == 2, hands
    return hands


def _grasp_outcome(
    hand: tuple[str, str, str],
    outcome: Literal["miss", "attach"],
    *,
    misses: int,
    target_id: str = "cell:pick_box",
    severity: int = SEVERITY_FAIL,
    stamp_ns: int = 1,
) -> FailureEventRecord:
    """The ``FailureTrigger`` the HAL bridge publishes, as ``_on_failure`` records it."""
    joint, jaw, attach = hand
    evidence = GraspOutcomeEvidence(
        outcome=outcome,
        gripper_joint=joint,
        jaw_link=jaw,
        hand_link=attach,
        consecutive_misses=misses,
        target_id=target_id,
        short_of_command=0.031,
        settle_spread=0.0004,
    )
    return FailureEventRecord(
        source="hal",
        kind=KIND_GRASP_OUTCOME,
        severity=severity,
        evidence_json=evidence.model_dump_json(),
        rskill_id="restock_pick",
        trace_id="",
        stamp_ns=stamp_ns,
    )


def test_reflect_on_grasp_miss_names_hand_target_and_the_levers() -> None:
    """The hint carries the evidence and points at what execute_rskill can change."""
    hint = reflect_on_grasp_miss("openarm_left_finger_pair", "cell:pick_box", 0.031)
    for part in ("openarm_left_finger_pair", "'cell:pick_box'", "0.031", "nothing was attached"):
        assert part in hint
    for lever in ("grasp_target.object_id", "grasp_target.contact_links", "different skill"):
        assert lever in hint
    assert "identical execute_rskill" in hint  # not a blind repeat
    assert "in a row" not in hint  # a first miss is not called a pattern
    repeated = reflect_on_grasp_miss("openarm_left_finger_pair", "", None, consecutive_misses=3)
    assert "3 misses in a row" in repeated
    assert "an undeclared target" in repeated and "unknown" in repeated


def test_a_grasp_miss_opens_a_reflection_line_for_that_hand() -> None:
    """Issue #353: one ``[grasp_miss]`` line per hand, naming its jaw link and target."""
    left, _right = _openarm_hands()
    r = ContextRenderer()
    seq0 = r.seq
    r.append_failure(_grasp_outcome(left, "miss", misses=1))
    assert r.seq == seq0 + 1
    out = r.render(world_state=None)
    execution = out[out.index("## EXECUTION") : out.index("## FAILURES")]
    assert (
        f"[grasp_miss] hand={left[1]} target=cell:pick_box misses=1 — reflect: "
        f"{reflect_on_grasp_miss(left[1], 'cell:pick_box', 0.031)}"
    ) in execution
    # The event itself is still on the failure buffer, like every other kind.
    assert f"[hal] kind={KIND_GRASP_OUTCOME} severity={SEVERITY_FAIL}" in out


def test_the_next_miss_on_a_hand_replaces_its_line() -> None:
    """A hand has one open miss: the latest, with its running count."""
    left, _right = _openarm_hands()
    r = ContextRenderer()
    r.append_failure(_grasp_outcome(left, "miss", misses=1, stamp_ns=1))
    r.append_failure(_grasp_outcome(left, "miss", misses=2, stamp_ns=2))
    out = r.render(world_state=None)
    assert out.count("[grasp_miss]") == 1
    assert "misses=2" in out and "2 misses in a row" in out


def test_an_attach_closes_only_that_hands_miss() -> None:
    """The recovery ATTACH (``outcome="attach"``) drops its own hand's line, not the other's."""
    left, right = _openarm_hands()
    r = ContextRenderer()
    r.append_failure(_grasp_outcome(left, "miss", misses=1))
    r.append_failure(_grasp_outcome(right, "miss", misses=1, target_id="", severity=SEVERITY_WARN))
    out = r.render(world_state=None)
    assert f"hand={left[1]}" in out and f"hand={right[1]} target=-" in out
    r.append_failure(_grasp_outcome(left, "attach", misses=1, severity=SEVERITY_INFO))
    out = r.render(world_state=None)
    assert f"[grasp_miss] hand={left[1]}" not in out
    assert f"[grasp_miss] hand={right[1]}" in out


def test_open_grasp_misses_drop_on_an_estop_clear_and_a_task_change() -> None:
    """A miss is about the active task: an e-stop clear or a new task makes it stale."""
    left, right = _openarm_hands()
    r = ContextRenderer()
    r.append_failure(_grasp_outcome(left, "miss", misses=1))
    r.clear_failures()
    assert "[grasp_miss]" not in r.render(world_state=None)

    r.set_mission(MissionState(["pick the carton", "place it in the tote"]))
    r.append_failure(_grasp_outcome(right, "miss", misses=1))
    assert "[grasp_miss]" in r.render(world_state=None)
    r.advance_mission(done=False, verdict="abandoned after misses")
    assert "[grasp_miss]" not in r.render(world_state=None)

    r.append_failure(_grasp_outcome(right, "miss", misses=1))
    r.set_mission(MissionState(["restock the shelf"]))
    assert "[grasp_miss]" not in r.render(world_state=None)


def test_a_malformed_grasp_outcome_only_buffers() -> None:
    """A payload the union rejects opens nothing; it still shows raw on the failure line."""
    r = ContextRenderer()
    r.append_failure(
        FailureEventRecord(
            source="hal",
            kind=KIND_GRASP_OUTCOME,
            severity=SEVERITY_FAIL,
            evidence_json='{"kind": "grasp_outcome", "outcome": "miss"}',
            rskill_id="",
            trace_id="",
            stamp_ns=1,
        )
    )
    out = r.render(world_state=None)
    assert "[grasp_miss]" not in out
    assert "evidence=" in out


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
        result_cls.FAILURE_OBJECT_NOT_FOUND,
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
