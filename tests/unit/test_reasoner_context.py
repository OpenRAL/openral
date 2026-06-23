"""Unit tests for :class:`openral_reasoner.ContextRenderer` (ADR-0018 F4).

Real Pydantic schemas + real ContextRenderer — no mocks. Tests assert
deterministic rendering, rolling-buffer behaviour, and the drain-once
contract for the prompt buffer.
"""

from __future__ import annotations

from openral_core import JointState, Pose6D, TimeoutEvidence, WorldState
from openral_reasoner import (
    ContextRenderer,
    FailureEventRecord,
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
