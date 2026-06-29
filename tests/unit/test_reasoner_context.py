"""Unit tests for :class:`openral_reasoner.ContextRenderer` (ADR-0018 F4).

Real Pydantic schemas + real ContextRenderer — no mocks. Tests assert
deterministic rendering, rolling-buffer behaviour, and the drain-once
contract for the prompt buffer.
"""

from __future__ import annotations

from openral_core import (
    JointState,
    ObjectDetection2D,
    ObjectsMetadata,
    Pose6D,
    TimeoutEvidence,
    WorldState,
)
from openral_reasoner import (
    ContextRenderer,
    FailureEventRecord,
    MissionState,
    PerceptionEventRecord,
    PromptRecord,
)


def _world_state() -> WorldState:
    """Build a realistic WorldState for rendering tests."""
    return WorldState(
        stamp_ns=1_700_000_000_000_000_000,
        joint_state=JointState(
            name=["j1", "j2", "j3"],
            position=[0.1, -0.2, 0.3],
            stamp_ns=1_700_000_000_000_000_000,
        ),
        ee_poses={
            "gripper": Pose6D(
                xyz=(0.3, 0.0, 0.2),
                quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            ),
        },
        diagnostics={"hal": "ok", "sensors": "warn"},
        battery_pct=87.5,
    )


def test_renders_empty_when_no_state() -> None:
    """An empty renderer produces the five section headers + '(none)' filler."""
    text = ContextRenderer().render(world_state=None)
    for header in ("## WORLD_STATE", "## EXECUTION", "## FAILURES", "## PERCEPTION", "## PROMPTS"):
        assert header in text
    assert "(no snapshot yet)" in text
    assert text.count("(none)") == 4  # executions, failures, perception, prompts


def test_renders_world_state_joint_positions() -> None:
    """Joint state lands as ``name=±value`` pairs in sorted order."""
    r = ContextRenderer()
    text = r.render(world_state=_world_state())
    assert "joint_state:" in text
    assert "j1=+0.100" in text
    assert "j2=-0.200" in text
    assert "j3=+0.300" in text


def test_renders_diagnostics_and_battery() -> None:
    """diagnostics and battery_pct land in the WorldState block."""
    text = ContextRenderer().render(world_state=_world_state())
    assert "battery_pct: 87.5" in text
    assert "diagnostics:" in text
    assert "hal=ok" in text and "sensors=warn" in text


def test_renders_ee_pose() -> None:
    """End-effector pose lands as ``ee_pose[<name>]: xyz=... rpy=...``."""
    text = ContextRenderer().render(world_state=_world_state())
    assert "ee_pose[gripper]:" in text
    assert "xyz=(+0.300,+0.000,+0.200)" in text
    assert "frame=base_link" in text


def test_renders_detected_scene_objects() -> None:
    """Lifted scene objects surface as ``scene_objects[<frame>]: label@(x,y,z), …``.

    Without this the LLM only ever learns a goal noun is "not in memory" — it
    never sees the labels the perception lift actually placed (e.g. "bread"), so
    it cannot apply its own semantics to map a goal ("baguette") onto a detected
    object. Surfacing the labels is what lets the reasoner bridge that gap
    (#14). Deduped by label (the open-vocab detector emits overlapping boxes).
    """
    from openral_core import DetectedObject

    ws = WorldState(
        stamp_ns=1_700_000_000_000_000_000,
        joint_state=JointState(name=["j1"], position=[0.0], stamp_ns=1_700_000_000_000_000_000),
        detected_objects=[
            DetectedObject(
                label="bread",
                confidence=0.81,
                pose=Pose6D(
                    xyz=(4.83, -1.00, 0.94), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map"
                ),
            ),
            DetectedObject(
                label="banana",
                confidence=0.74,
                pose=Pose6D(
                    xyz=(4.92, -1.13, 1.42), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map"
                ),
            ),
            # Duplicate label (overlapping detection) collapses to one entry.
            DetectedObject(
                label="bread",
                confidence=0.66,
                pose=Pose6D(
                    xyz=(4.80, -1.02, 0.93), quat_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="map"
                ),
            ),
        ],
    )
    text = ContextRenderer().render(world_state=ws)
    assert "scene_objects[map]:" in text
    assert "bread@(+4.83,-1.00,+0.94)" in text
    assert "banana@(+4.92,-1.13,+1.42)" in text
    # Deduped: the label "bread" appears once in the scene_objects line.
    scene_line = next(line for line in text.splitlines() if line.startswith("scene_objects[map]:"))
    assert scene_line.count("bread@") == 1


def test_failure_buffer_rolls_at_capacity() -> None:
    """Buffer capacity drops the oldest entries; render shows only the latest N."""
    r = ContextRenderer(buffer_size=2)
    for i in range(5):
        r.append_failure(
            FailureEventRecord(
                source="hal",
                kind=0,
                severity=1,
                evidence_json="",
                rskill_id=f"skill_{i}",
                trace_id="0123456789abcdef",
                stamp_ns=i,
            ),
        )
    assert len(r.failures) == 2
    text = r.render(world_state=None)
    assert "skill_3" in text
    assert "skill_4" in text
    assert "skill_0" not in text


def test_failure_render_summarises_evidence() -> None:
    """A real TimeoutEvidence evidence_json is summarised inline."""
    evidence = TimeoutEvidence(operation="skill.step", deadline_s=0.1, elapsed_s=0.2)
    r = ContextRenderer()
    r.append_failure(
        FailureEventRecord(
            source="rskill",
            kind=0,
            severity=2,
            evidence_json=evidence.model_dump_json(),
            rskill_id="pick_cube",
            trace_id="cafe1234deadbeef",
            stamp_ns=0,
        ),
    )
    text = r.render(world_state=None)
    assert "skill.step" in text
    assert "deadline_s" in text


def test_prompts_drain_once() -> None:
    """drain_prompts() empties the prompt buffer (pull-once semantics)."""
    r = ContextRenderer()
    r.append_prompt(PromptRecord(text="pick", metadata_json="", stamp_ns=0))
    r.append_prompt(PromptRecord(text="place", metadata_json="", stamp_ns=1))
    assert len(r.prompts) == 2
    drained = r.drain_prompts()
    assert [p.text for p in drained] == ["pick", "place"]
    assert r.prompts == ()


def test_perception_buffer_renders_per_kind() -> None:
    """Each perception event renders as ``[kind] text`` on its own line."""
    r = ContextRenderer()
    r.append_perception(
        PerceptionEventRecord(
            kind="motion",
            text="motion magnitude=0.030 on cam0",
            metadata_json="",
            stamp_ns=0,
        ),
    )
    r.append_perception(
        PerceptionEventRecord(
            kind="scene_change",
            text="scene_change distance=0.700 on cam0",
            metadata_json="",
            stamp_ns=1,
        ),
    )
    text = r.render(world_state=None)
    assert "[motion] motion magnitude=0.030" in text
    assert "[scene_change] scene_change distance=0.700" in text


def test_high_priority_prompt_overtakes_queued_auto_prompts() -> None:
    """ADR-0018 §3.F10 — a human-source prompt drains before queued auto-prompts."""
    r = ContextRenderer()
    # Auto cascade priority 10 (the reasoner's own EmitPromptTool cascade).
    r.append_prompt(
        PromptRecord(
            text="auto: keep going",
            metadata_json='{"source": "auto", "priority": 10}',
            stamp_ns=0,
        ),
    )
    r.append_prompt(
        PromptRecord(
            text="auto: second cascade",
            metadata_json='{"source": "auto", "priority": 10}',
            stamp_ns=1,
        ),
    )
    # Human source priority 100 — should drain first even though it
    # arrived after the two auto prompts.
    r.append_prompt(
        PromptRecord(
            text="operator: STOP and pick the blue cube",
            metadata_json='{"source": "cli", "priority": 100}',
            stamp_ns=2,
        ),
    )
    drained = [p.text for p in r.drain_prompts()]
    assert drained == [
        "operator: STOP and pick the blue cube",
        "auto: keep going",
        "auto: second cascade",
    ]


def test_priority_extracted_from_metadata_json_when_default_on_record() -> None:
    """When PromptRecord.priority is left at the default, metadata_json wins."""
    r = ContextRenderer()
    r.append_prompt(
        PromptRecord(
            text="cli: priority via metadata",
            metadata_json='{"source": "cli", "priority": 100}',
            stamp_ns=0,
        ),
    )
    drained = r.drain_prompts()
    assert drained[0].priority == 100


def test_explicit_priority_on_record_wins_over_metadata_json() -> None:
    """An explicit ``priority`` field on the record overrides metadata_json."""
    r = ContextRenderer()
    r.append_prompt(
        PromptRecord(
            text="explicit",
            metadata_json='{"priority": 1}',  # ignored
            stamp_ns=0,
            priority=42,  # honoured
        ),
    )
    drained = r.drain_prompts()
    assert drained[0].priority == 42


def test_buffer_eviction_drops_lowest_priority_first() -> None:
    """A high-priority prompt never gets dropped to make room for a low-priority one."""
    r = ContextRenderer(buffer_size=2)
    r.append_prompt(
        PromptRecord(
            text="critical",
            metadata_json='{"priority": 100}',
            stamp_ns=0,
        ),
    )
    r.append_prompt(
        PromptRecord(
            text="low-1",
            metadata_json='{"priority": 10}',
            stamp_ns=1,
        ),
    )
    r.append_prompt(
        PromptRecord(
            text="low-2",
            metadata_json='{"priority": 10}',
            stamp_ns=2,
        ),
    )
    texts = [p.text for p in r.drain_prompts()]
    # The critical prompt stays; one of the low-priority prompts is
    # evicted to enforce buffer_size=2.
    assert "critical" in texts
    assert len(texts) == 2


def test_render_is_deterministic_for_identical_input() -> None:
    """Same input → byte-identical output (no clock leaks, no dict iter order)."""
    r1 = ContextRenderer()
    r2 = ContextRenderer()
    for r in (r1, r2):
        r.append_prompt(PromptRecord(text="hello", metadata_json="", stamp_ns=0))
        r.append_failure(
            FailureEventRecord(
                source="rskill",
                kind=4,
                severity=2,
                evidence_json="",
                rskill_id="x",
                trace_id="0123456789abcdef",
                stamp_ns=0,
            ),
        )
    assert r1.render(world_state=_world_state()) == r2.render(world_state=_world_state())


# ── ADR-0018 amendment 2026-05-25 §2 — seq counter for heartbeat_idle ────────


def test_seq_increments_on_every_append() -> None:
    """Every successful append_* bumps the seq counter.

    ReasonerCore reads ``seq`` to decide whether a heartbeat tick can
    be suppressed (no event since last tick → byte-identical context
    → wasted LLM call).
    """
    r = ContextRenderer()
    assert r.seq == 0
    r.append_prompt(PromptRecord(text="a", metadata_json="", stamp_ns=0))
    assert r.seq == 1
    r.append_failure(
        FailureEventRecord(
            source="rskill",
            kind=0,
            severity=2,
            evidence_json="",
            rskill_id="x",
            trace_id="",
            stamp_ns=0,
        ),
    )
    assert r.seq == 2
    r.append_perception(
        PerceptionEventRecord(kind="motion", text="moved", metadata_json="", stamp_ns=0),
    )
    assert r.seq == 3


def test_seq_is_not_reset_by_drain_prompts() -> None:
    """drain_prompts() clears the buffer but the renderer has still 'seen' the prompt.

    The seq counter is monotonic: it represents *what the renderer
    has observed*, not *what is in the buffer*. ReasonerCore relies
    on that invariant to keep the heartbeat-idle gate from firing
    after a successful tick has drained the prompt that arrived
    between the previous tick and this one.
    """
    r = ContextRenderer()
    r.append_prompt(PromptRecord(text="a", metadata_json="", stamp_ns=0))
    seq_before = r.seq
    r.drain_prompts()
    assert r.seq == seq_before


# ── ADR-0073 §1 — mission (## MISSION) rendering ─────────────────────────────


def test_no_mission_omits_section() -> None:
    r = ContextRenderer()
    assert "## MISSION" not in r.render(world_state=None)


def test_empty_mission_omits_section() -> None:
    r = ContextRenderer()
    r.set_mission(MissionState.from_prompt("   "))  # no tasks
    assert "## MISSION" not in r.render(world_state=None)


def test_set_mission_renders_active_task_and_bumps_seq() -> None:
    r = ContextRenderer()
    seq_before = r.seq
    r.set_mission(MissionState(["pick the bowl", "place the butter"]))
    # A new goal is an event — it must wake an idle heartbeat.
    assert r.seq == seq_before + 1
    out = r.render(world_state=None)
    assert "## MISSION" in out
    assert "▶ t1: pick the bowl" in out
    assert "1 pending task(s)" in out  # t2 still pending


def test_advance_mission_completes_and_activates_next_and_bumps_seq() -> None:
    r = ContextRenderer()
    r.set_mission(MissionState(["pick the bowl", "place the butter"]))
    seq_before = r.seq
    nxt = r.advance_mission(done=True, verdict="success=0.92")
    # Advancing the queue is an event — the new active task wakes the reasoner.
    assert r.seq == seq_before + 1
    assert nxt is not None and nxt.text == "place the butter"
    out = r.render(world_state=None)
    assert "✓ t1: pick the bowl [success=0.92]" in out
    assert "▶ t2: place the butter" in out


def test_advance_mission_abandons_on_done_false() -> None:
    r = ContextRenderer()
    r.set_mission(MissionState(["hard task", "easy task"]))
    r.advance_mission(done=False, verdict="ladder exhausted: stalled@0.73")
    out = r.render(world_state=None)
    assert "✗ t1: hard task" in out
    assert "▶ t2: easy task" in out


def test_advance_mission_with_no_mission_is_noop() -> None:
    r = ContextRenderer()
    seq_before = r.seq
    assert r.advance_mission(done=True, verdict="x") is None
    assert r.seq == seq_before  # no mission → no bump


def test_mission_property_exposes_state_for_in_place_bookkeeping() -> None:
    r = ContextRenderer()
    r.set_mission(MissionState(["a", "b"]))
    # The node records attempts in place (non-waking bookkeeping).
    r.mission.record_attempt(rskill_id="OpenRAL/rskill-smolvla-libero")
    assert r.mission.active().attempts == 1
    assert "attempts=1" in r.render(world_state=None)


def test_mission_finishes_when_last_task_completed() -> None:
    r = ContextRenderer()
    r.set_mission(MissionState.from_prompt("only task"))
    assert r.advance_mission(done=True, verdict="success=0.95") is None
    assert r.mission.is_complete()
    out = r.render(world_state=None)
    assert "✓ t1: only task" in out


def _in_view() -> ObjectsMetadata:
    """A camera-space ObjectsMetadata with stable det_ids (ADR-0076)."""
    return ObjectsMetadata(
        sensor_id="top",
        model_id="omdet-turbo-indoor",
        frame_width=640,
        frame_height=480,
        detections=[
            ObjectDetection2D(
                label="milk", confidence=0.9, bbox_xyxy=(402, 211, 432, 261), det_id=0
            ),
            ObjectDetection2D(
                label="ketchup", confidence=0.8, bbox_xyxy=(370, 230, 406, 272), det_id=1
            ),
        ],
    )


def test_in_view_line_rendered_in_world_state() -> None:
    """ADR-0076 — set_in_view surfaces a camera-space `in_view[<cam>]` line with ids+px."""
    r = ContextRenderer()
    r.set_in_view(_in_view())
    out = r.render(world_state=_world_state())
    assert "in_view[top]: #0 milk @px(417,236), #1 ketchup @px(388,251)" in out


def test_in_view_renders_before_first_world_state_snapshot() -> None:
    """The enumeration is depth-free and may arrive before any WorldState."""
    r = ContextRenderer()
    r.set_in_view(_in_view())
    out = r.render(world_state=None)
    assert "(no snapshot yet)" in out
    assert "in_view[top]: #0 milk" in out


def test_set_in_view_bumps_seq_and_clears() -> None:
    """A new detection snapshot is an event (wakes a heartbeat); None clears the line."""
    r = ContextRenderer()
    seq0 = r.seq
    r.set_in_view(_in_view())
    assert r.seq > seq0
    r.set_in_view(None)
    assert "in_view" not in r.render(world_state=_world_state())


def _located_basket() -> ObjectsMetadata:
    """An open-vocab locate hit for the goal noun the fixed indoor vocab misses."""
    return ObjectsMetadata(
        sensor_id="top",
        model_id="omdet-turbo-locator",
        frame_width=640,
        frame_height=480,
        detections=[
            ObjectDetection2D(label="basket", confidence=0.6, bbox_xyxy=(100, 300, 200, 400)),
        ],
    )


def test_note_located_survives_continuous_in_view_clobber() -> None:
    """ADR-0076 — the deploy locate-loop fix: a goal noun the reasoner confirmed via
    open-vocab locate_in_view (``basket``) must persist on the ``located`` line even
    after the fixed-vocab continuous detector overwrites ``in_view`` (which never
    carries ``basket``), so the LLM can decompose/dispatch instead of re-locating."""
    r = ContextRenderer()
    r.note_located(_located_basket())
    assert "located[top]: basket @px(150,350)" in r.render(world_state=_world_state())
    # Continuous detector clobbers in_view with its clutter — basket NOT in it.
    r.set_in_view(_in_view())
    out = r.render(world_state=_world_state())
    assert "in_view[top]: #0 milk" in out  # continuous line present
    assert "located[top]: basket @px(150,350)" in out  # sticky hit survives the clobber


def test_note_located_latest_wins_and_bumps_seq() -> None:
    """A re-confirmed label updates its bbox (latest wins); a hit wakes a heartbeat."""
    r = ContextRenderer()
    seq0 = r.seq
    r.note_located(_located_basket())
    assert r.seq > seq0
    moved = ObjectsMetadata(
        sensor_id="top",
        model_id="omdet-turbo-locator",
        frame_width=640,
        frame_height=480,
        detections=[
            ObjectDetection2D(label="basket", confidence=0.7, bbox_xyxy=(0, 0, 100, 100)),
        ],
    )
    r.note_located(moved)
    out = r.render(world_state=_world_state())
    assert "located[top]: basket @px(50,50)" in out
    assert out.count("basket @px") == 1  # one entry, not two
    # None / empty is a no-op.
    seq1 = r.seq
    r.note_located(None)
    assert r.seq == seq1
