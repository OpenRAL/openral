"""What the reasoner LLM sees (issue #620): safety state, names, ages, progress, payload.

Real ``ContextRenderer`` + real Pydantic ``WorldState``; the clock is injected.
"""

from __future__ import annotations

from typing import Any

from openral_core import (
    AttachedCollisionObject,
    AttachmentEvidenceKind,
    JointState,
    Pose6D,
    RobotDescription,
    WorldState,
)
from openral_hal._vision_attachment_evidence import jaw_span_primitive
from openral_observability.failure_bus import KIND_COLLISION, SEVERITY_ABORT
from openral_reasoner import ContextRenderer
from openral_reasoner.context import (
    SAFETY_STATUS_LIVENESS_S,
    ExecutionEventRecord,
    FailureEventRecord,
    PerceptionEventRecord,
    SafetyStateRecord,
)

_S = 1_000_000_000


def _renderer(now_s: list[float]) -> ContextRenderer:
    return ContextRenderer(now_ns=lambda: int(now_s[0] * _S))


def test_identical_perception_does_not_bump_seq_or_fill_the_buffer() -> None:
    r = ContextRenderer()
    rec = PerceptionEventRecord(kind="objects", text="3 objects", metadata_json="", stamp_ns=1)
    r.append_perception(rec)
    seq = r.seq
    for i in range(20):
        r.append_perception(
            PerceptionEventRecord(kind="objects", text="3 objects", metadata_json="", stamp_ns=i)
        )
    assert r.seq == seq
    assert len(r.perception_events) == 1
    r.append_perception(
        PerceptionEventRecord(kind="objects", text="4 objects", metadata_json="", stamp_ns=30)
    )
    assert r.seq == seq + 1


def test_failures_render_kind_and_severity_names_with_age() -> None:
    now = [100.0]
    r = _renderer(now)
    r.append_failure(
        FailureEventRecord(
            source="safety",
            kind=KIND_COLLISION,
            severity=SEVERITY_ABORT,
            evidence_json="",
            rskill_id="openral/rskill-pick",
            trace_id="",
            stamp_ns=int(88 * _S),
        )
    )
    out = r.render(world_state=None)
    assert "[safety] kind=COLLISION severity=ABORT age=12.0s" in out


def test_safety_latched_renders_and_bumps_only_on_change() -> None:
    now = [10.0]
    r = _renderer(now)
    seq = r.seq
    r.set_safety_status(SafetyStateRecord(latched=False, reason="", stamp_ns=int(10 * _S)))
    assert r.seq == seq + 1
    assert "## SAFETY\nclear" in r.render(world_state=None)
    # 1 Hz liveness refresh of the same state: not an event.
    now[0] = 11.0
    r.set_safety_status(SafetyStateRecord(latched=False, reason="", stamp_ns=int(11 * _S)))
    assert r.seq == seq + 1
    r.set_safety_status(
        SafetyStateRecord(latched=True, reason="EXTERNAL_ESTOP", stamp_ns=int(11 * _S))
    )
    assert r.seq == seq + 2
    assert r.safety_latched
    assert "LATCHED reason=EXTERNAL_ESTOP" in r.render(world_state=None)


def test_quiet_safety_status_reads_unknown_never_clear() -> None:
    now = [0.0]
    r = _renderer(now)
    # Received at clock time zero: zero is a receipt time like any other.
    r.set_safety_status(SafetyStateRecord(latched=False, reason="", stamp_ns=0))
    assert "## SAFETY\nclear" in r.render(world_state=None)
    now[0] = SAFETY_STATUS_LIVENESS_S + 1.0
    r.refresh_clock_state()
    out = r.render(world_state=None)
    assert "## SAFETY\nUNKNOWN" in out
    assert "## SAFETY\nclear" not in out


def test_safety_status_going_stale_and_coming_back_bumps_seq() -> None:
    """No message announces silence: the liveness crossing itself must wake the model."""
    now = [10.0]
    r = _renderer(now)
    r.set_safety_status(SafetyStateRecord(latched=False, reason="", stamp_ns=int(10 * _S)))
    live = r.seq
    now[0] = 10.0 + SAFETY_STATUS_LIVENESS_S + 0.5
    assert r.seq == live  # reading never changes the renderer
    assert "## SAFETY\nclear" in r.render(world_state=None)  # nor does rendering
    r.refresh_clock_state()
    stale = r.seq
    assert stale == live + 1
    assert "## SAFETY\nUNKNOWN" in r.render(world_state=None)
    r.refresh_clock_state()
    assert r.seq == stale
    r.set_safety_status(SafetyStateRecord(latched=False, reason="", stamp_ns=int(now[0] * _S)))
    assert r.seq == stale + 1
    assert "## SAFETY\nclear" in r.render(world_state=None)


def test_safety_status_without_a_clock_is_never_clear() -> None:
    r = ContextRenderer()
    r.set_safety_status(SafetyStateRecord(latched=False, reason="", stamp_ns=1))
    out = r.render(world_state=None)
    assert "## SAFETY\nUNKNOWN — no clock" in out


def test_inflight_line_shows_elapsed_and_feedback_without_waking() -> None:
    now = [50.0]
    r = _renderer(now)
    r.set_inflight_skill("openral/rskill-pick", stamp_ns=int(20 * _S))
    seq = r.seq
    r.set_inflight_feedback("state=running progress=0.40 chunk=12")
    assert r.seq == seq
    out = r.render(world_state=None)
    assert "running for 30.0s" in out
    assert "latest feedback: state=running progress=0.40 chunk=12" in out
    r.set_inflight_skill(None)
    assert "latest feedback" not in r.render(world_state=None)


def test_note_without_wake_is_rendered_but_does_not_bump_seq() -> None:
    r = ContextRenderer()
    seq = r.seq
    r.append_execution(
        ExecutionEventRecord(
            rskill_id="(emit_prompt)",
            outcome="ok",
            summary="said to the operator: which shelf?",
            reflection=None,
            stamp_ns=1,
        ),
        wake=False,
    )
    assert r.seq == seq
    assert "said to the operator: which shelf?" in r.render(world_state=None)


def test_world_state_renders_age_base_pose_and_what_is_held() -> None:
    # Joint and hand-link names from the real OpenArm manifest, not placeholders.
    description = RobotDescription.from_yaml("robots/openarm/robot.yaml")
    joint = description.joints[0]
    gripper = next(j for j in description.joints if j.role == "gripper")
    now = [5.0]
    r = _renderer(now)
    ws = WorldState(
        stamp_ns=int(4.5 * _S),
        joint_state=JointState(name=[joint.name], position=[0.1], stamp_ns=1),
        base_pose=Pose6D(xyz=(1.0, 2.0, 0.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        attached_objects=[
            AttachedCollisionObject(
                object_id="milk_1",
                attach_link=gripper.parent_link,
                touch_links=[gripper.child_link],
                primitives=[jaw_span_primitive(object_id="milk_1", jaw_span_m=0.06)],
                pose_in_link=Pose6D(
                    xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id=gripper.parent_link
                ),
                confidence=0.9,
                evidence_kind=AttachmentEvidenceKind.GRIPPER_FORCE,
                stamp_ns=1,
            )
        ],
    )
    out = r.render(world_state=ws)
    assert f"stamp_ns: {ws.stamp_ns} age=0.5s" in out
    assert "base_pose: xyz=(+1.000,+2.000,+0.000) frame=map" in out
    assert f"holding: milk_1 on {gripper.parent_link} (confidence 0.90)" in out


def test_an_age_from_another_clock_base_is_omitted() -> None:
    """A sim-time stamp read on a wall clock would give a nonsense age: render none."""
    now = [1_794_858_868.0]  # wall time
    r = _renderer(now)
    r.append_perception(
        PerceptionEventRecord(kind="motion", text="arm moved", metadata_json="", stamp_ns=42 * _S)
    )
    assert "[motion] arm moved" in r.render(world_state=None)


def _core_rig(now: list[float]) -> tuple[ContextRenderer, Any, list[str]]:
    """A real core + renderer sharing one injected clock; the LLM records each context."""
    from openral_core import EmitPromptTool
    from openral_reasoner import MissionState, ReasonerCore, ToolPalette

    from tests.integration.fakes.fake_llm import FakeToolUseClient

    renderer = ContextRenderer(now_ns=lambda: int(now[0] * _S))
    renderer.set_mission(MissionState.from_prompt("pick the cube"))
    contexts: list[str] = []

    def select(context: str, _palette: ToolPalette) -> EmitPromptTool:
        contexts.append(context)
        return EmitPromptTool(target_topic="/openral/prompt", text=str(len(contexts)))

    core = ReasonerCore(
        client=FakeToolUseClient(selector=select), min_interval_s=0.0, clock=lambda: now[0]
    )
    palette = ToolPalette(execute_rskill_ids=frozenset({"openral/rskill-pick"}))

    def tick(force: bool = False) -> Any:
        return core.tick(world_state=None, renderer=renderer, palette=palette, force=force)

    return renderer, tick, contexts


def test_the_model_is_shown_every_safety_liveness_change_without_an_event() -> None:
    """Live -> UNKNOWN -> live, with only timer beats: each change reaches the LLM once."""
    now = [10.0]
    renderer, tick, contexts = _core_rig(now)
    renderer.set_safety_status(SafetyStateRecord(latched=False, reason="", stamp_ns=int(10 * _S)))
    tick(force=True)
    assert "## SAFETY\nclear" in contexts[-1]
    now[0] += 1.0
    assert tick().suppressed_reason == "heartbeat_idle"
    now[0] += SAFETY_STATUS_LIVENESS_S  # the status goes quiet
    assert tick().tool_call is not None
    assert "## SAFETY\nUNKNOWN" in contexts[-1]
    now[0] += 1.0
    assert tick().suppressed_reason == "heartbeat_idle"
    renderer.set_safety_status(  # an unchanged, live status returns
        SafetyStateRecord(latched=False, reason="", stamp_ns=int(now[0] * _S))
    )
    assert tick().tool_call is not None
    assert "## SAFETY\nclear" in contexts[-1]


def test_a_latch_reaches_the_model_once_then_beats_hold() -> None:
    """A latch that landed after the event tick rendered gets one beat; then the gate holds."""
    now = [10.0]
    renderer, tick, contexts = _core_rig(now)
    renderer.set_safety_status(SafetyStateRecord(latched=False, reason="", stamp_ns=int(10 * _S)))
    tick(force=True)  # e.g. the kernel's FailureTrigger tick, before the status latched
    renderer.set_safety_status(
        SafetyStateRecord(latched=True, reason="kind_collision", stamp_ns=int(now[0] * _S))
    )
    now[0] += 1.0
    assert tick().tool_call is not None
    assert "LATCHED reason=kind_collision" in contexts[-1]
    now[0] += 1.0
    assert tick().suppressed_reason == "safety_latched"


def test_a_safety_status_from_the_future_is_unknown() -> None:
    """The graph clock jumped back (sim reset): a negative age proves nothing about freshness."""
    now = [100.0]
    r = _renderer(now)
    r.set_safety_status(SafetyStateRecord(latched=False, reason="", stamp_ns=int(100 * _S)))
    now[0] = 0.0
    r.refresh_clock_state()
    assert "## SAFETY\nUNKNOWN" in r.render(world_state=None)


def test_a_released_object_is_not_reported_as_held() -> None:
    description = RobotDescription.from_yaml("robots/openarm/robot.yaml")
    joint = description.joints[0]
    gripper = next(j for j in description.joints if j.role == "gripper")

    def attached(object_id: str, released_by: str | None) -> AttachedCollisionObject:
        return AttachedCollisionObject(
            object_id=object_id,
            attach_link=gripper.parent_link,
            touch_links=[gripper.child_link],
            primitives=[jaw_span_primitive(object_id=object_id, jaw_span_m=0.06)],
            pose_in_link=Pose6D(
                xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id=gripper.parent_link
            ),
            confidence=0.9,
            evidence_kind=AttachmentEvidenceKind.GRIPPER_FORCE,
            stamp_ns=1,
            released_by_link=released_by,
        )

    ws = WorldState(
        stamp_ns=1,
        joint_state=JointState(name=[joint.name], position=[0.1], stamp_ns=1),
        attached_objects=[attached("milk_1", None), attached("box_2", gripper.child_link)],
    )
    out = ContextRenderer().render(world_state=ws)
    assert "holding: milk_1" in out
    assert "box_2" not in out
