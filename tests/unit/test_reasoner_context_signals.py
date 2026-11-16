"""What the reasoner LLM sees (issue #620): safety state, names, ages, progress, payload.

Real ``ContextRenderer`` + real Pydantic ``WorldState``; the clock is injected.
"""

from __future__ import annotations

from openral_core import (
    AttachedCollisionObject,
    AttachmentEvidenceKind,
    JointState,
    Pose6D,
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
    r.set_safety_status(SafetyStateRecord(latched=False, reason="", stamp_ns=1))
    now[0] = SAFETY_STATUS_LIVENESS_S + 1.0
    out = r.render(world_state=None)
    assert "## SAFETY\nUNKNOWN" in out
    assert "## SAFETY\nclear" not in out


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
    now = [5.0]
    r = _renderer(now)
    ws = WorldState(
        stamp_ns=int(4.5 * _S),
        joint_state=JointState(name=["j1"], position=[0.1], stamp_ns=1),
        base_pose=Pose6D(xyz=(1.0, 2.0, 0.0), quat_xyzw=(0, 0, 0, 1), frame_id="map"),
        attached_objects=[
            AttachedCollisionObject(
                object_id="milk_1",
                attach_link="gripper",
                touch_links=["finger_l"],
                primitives=[jaw_span_primitive(object_id="milk_1", jaw_span_m=0.06)],
                pose_in_link=Pose6D(xyz=(0, 0, 0), quat_xyzw=(0, 0, 0, 1), frame_id="gripper"),
                confidence=0.9,
                evidence_kind=AttachmentEvidenceKind.GRIPPER_FORCE,
                stamp_ns=1,
            )
        ],
    )
    out = r.render(world_state=ws)
    assert f"stamp_ns: {ws.stamp_ns} age=0.5s" in out
    assert "base_pose: xyz=(+1.000,+2.000,+0.000) frame=map" in out
    assert "holding: milk_1 on gripper (confidence 0.90)" in out


def test_an_age_from_another_clock_base_is_omitted() -> None:
    """A sim-time stamp read on a wall clock would give a nonsense age: render none."""
    now = [1_794_858_868.0]  # wall time
    r = _renderer(now)
    r.append_perception(
        PerceptionEventRecord(kind="motion", text="arm moved", metadata_json="", stamp_ns=42 * _S)
    )
    assert "[motion] arm moved" in r.render(world_state=None)
