"""Unit tests for the in-flight skill context line, seq gating, and located reset.

Three behaviours added by the supervision-loop fixes:

1. ``set_inflight_skill`` — the LLM must SEE that a skill is running (the
   ``in_flight:`` line in ``## EXECUTION``) and ``ReasonerCore``'s
   heartbeat-idle gate must stay live while one is (mid-run reward polling).
2. ``set_in_view`` bumps ``seq`` only when the rendered enumeration actually
   changes — a continuous detector republishing identical frames must not
   defeat the heartbeat-idle gate forever.
3. ``clear_located`` — a new operator goal drops the sticky open-vocab
   grounding from the previous goal.
"""

from __future__ import annotations

from openral_core import EmitPromptTool, ObjectDetection2D, ObjectsMetadata
from openral_reasoner import ContextRenderer, ReasonerCore
from openral_reasoner.palette import ToolPalette

from tests.integration.fakes.fake_llm import FakeToolUseClient

# ── in-flight line + property ────────────────────────────────────────────────


def test_inflight_skill_renders_and_bumps_seq() -> None:
    r = ContextRenderer()
    seq0 = r.seq
    r.set_inflight_skill("openral/rskill-pick", stamp_ns=123)
    assert r.seq == seq0 + 1
    assert r.inflight_skill == "openral/rskill-pick"
    rendered = r.render(world_state=None)
    assert "in_flight: skill=openral/rskill-pick is RUNNING" in rendered
    assert "stamp_ns=123" in rendered


def test_inflight_clear_bumps_seq_and_removes_line() -> None:
    r = ContextRenderer()
    r.set_inflight_skill("openral/rskill-pick", stamp_ns=1)
    seq_running = r.seq
    r.set_inflight_skill(None)
    assert r.seq == seq_running + 1
    assert r.inflight_skill is None
    assert "in_flight" not in r.render(world_state=None)


def test_inflight_reassert_same_state_is_a_no_op() -> None:
    r = ContextRenderer()
    r.set_inflight_skill("openral/rskill-pick", stamp_ns=1)
    seq = r.seq
    r.set_inflight_skill("openral/rskill-pick", stamp_ns=1)
    assert r.seq == seq
    r.set_inflight_skill(None)
    seq = r.seq
    r.set_inflight_skill(None)
    assert r.seq == seq


def test_heartbeat_stays_live_while_a_skill_is_inflight() -> None:
    """With no new events, a heartbeat is idle-suppressed — unless a skill is
    running, in which case the tick is the only chance to poll the reward
    monitor mid-execution (the system prompt instructs exactly that)."""
    # A non-empty palette (a sensor id suffices) so the palette_empty
    # short-circuit does not mask the gate under test.
    palette = ToolPalette(execute_rskill_ids=frozenset(), sensor_ids=frozenset({"top"}))
    responses = [EmitPromptTool(target_topic="/openral/prompt", text=f"t{i}") for i in range(3)]
    client = FakeToolUseClient(responses=responses)
    core = ReasonerCore(client=client, min_interval_s=0.0)
    r = ContextRenderer()
    r.set_inflight_skill("openral/rskill-vla", stamp_ns=1)

    first = core.tick(world_state=None, renderer=r, palette=palette, force=True)
    assert first.tool_call is not None
    # seq unchanged since the last tick — but a skill is in flight, so the
    # heartbeat must NOT be suppressed as idle.
    second = core.tick(world_state=None, renderer=r, palette=palette)
    assert second.suppressed_reason != "heartbeat_idle"
    assert second.tool_call is not None
    # Once the skill lands (state change bumps seq, then goes quiet), the
    # idle gate resumes suppressing.
    r.set_inflight_skill(None)
    third = core.tick(world_state=None, renderer=r, palette=palette)
    assert third.tool_call is not None  # seq moved when the in-flight line cleared
    fourth = core.tick(world_state=None, renderer=r, palette=palette)
    assert fourth.suppressed_reason == "heartbeat_idle"


# ── in_view change-only seq bump ─────────────────────────────────────────────


def _frame(*labels_at: tuple[str, int]) -> ObjectsMetadata:
    return ObjectsMetadata(
        sensor_id="top",
        model_id="rtdetr",
        frame_width=256,
        frame_height=256,
        detections=[
            ObjectDetection2D(
                det_id=i,
                label=label,
                confidence=0.9,
                bbox_xyxy=(x, 10, x + 20, 30),
            )
            for i, (label, x) in enumerate(labels_at)
        ],
    )


def test_identical_in_view_frames_do_not_bump_seq() -> None:
    r = ContextRenderer()
    r.set_in_view(_frame(("milk", 100), ("basket", 200)))
    seq = r.seq
    for _ in range(5):
        r.set_in_view(_frame(("milk", 100), ("basket", 200)))
    assert r.seq == seq


def test_changed_in_view_frame_bumps_seq() -> None:
    r = ContextRenderer()
    r.set_in_view(_frame(("milk", 100)))
    seq = r.seq
    r.set_in_view(_frame(("milk", 100), ("basket", 200)))  # new object appears
    assert r.seq == seq + 1
    seq = r.seq
    r.set_in_view(_frame(("milk", 140), ("basket", 200)))  # object moved
    assert r.seq == seq + 1
    seq = r.seq
    r.set_in_view(None)  # enumeration cleared
    assert r.seq == seq + 1


# ── clear_located ────────────────────────────────────────────────────────────


def test_clear_located_drops_the_sticky_grounding_and_bumps_seq() -> None:
    r = ContextRenderer()
    r.note_located(_frame(("basket", 200)))
    assert "located[top]: basket" in r.render(world_state=None)
    seq = r.seq
    r.clear_located()
    assert r.seq == seq + 1
    assert "located[" not in r.render(world_state=None)
    # Idempotent: clearing an empty store does not bump.
    seq = r.seq
    r.clear_located()
    assert r.seq == seq


# ── in-flight phase (dispatching vs running) ─────────────────────────────────


def test_dispatching_phase_renders_loading_guidance() -> None:
    """During send→accept (cold policy loads take tens of seconds) the LLM must
    see that the goal is being accepted/loading — it used to read an unchanged
    snapshot and escalate "task is blocked" to the operator mid-load."""
    r = ContextRenderer()
    r.set_inflight_skill("openral/rskill-vla", stamp_ns=5, state="dispatching")
    assert r.inflight_state == "dispatching"
    rendered = r.render(world_state=None)
    assert "state=dispatching" in rendered
    assert "loading its policy" in rendered
    assert "do not escalate to the operator" in rendered


def test_dispatch_to_running_transition_bumps_seq_once() -> None:
    r = ContextRenderer()
    r.set_inflight_skill("openral/rskill-vla", stamp_ns=5, state="dispatching")
    seq = r.seq
    r.set_inflight_skill("openral/rskill-vla", stamp_ns=5, state="running")
    assert r.seq == seq + 1
    assert r.inflight_state == "running"
    assert "state=running" in r.render(world_state=None)
    # Re-asserting the same phase is a no-op.
    seq = r.seq
    r.set_inflight_skill("openral/rskill-vla", stamp_ns=5, state="running")
    assert r.seq == seq


def test_inflight_state_none_when_idle() -> None:
    r = ContextRenderer()
    assert r.inflight_state is None
    r.set_inflight_skill("x", stamp_ns=1, state="dispatching")
    r.set_inflight_skill(None)
    assert r.inflight_state is None
