"""Unit tests for :class:`openral_reasoner.ReasonerCore`.

Real ContextRenderer + real ToolPalette + real Pydantic tool calls;
the LLM endpoint is replaced by the deterministic
:class:`FakeToolUseClient` from
``tests/integration/fakes/fake_llm.py`` (CLAUDE.md §1.11 — fakes are
permitted at process boundaries when named explicitly and under
``tests/<tier>/fakes/``).
"""

from __future__ import annotations

from typing import Any

import pytest
import structlog
from openral_core import (
    EmitPromptTool,
    ExecuteRskillTool,
    LifecycleTransitionTool,
)
from openral_core.exceptions import ROSPlanningError, ROSReasonerInvalidPlan
from openral_reasoner import (
    ContextRenderer,
    MissionState,
    PerceptionEventRecord,
    PromptRecord,
    ReasonerCore,
    ToolPalette,
)

from tests.integration.fakes.fake_llm import FakeToolUseClient


def _palette(*skills: str) -> ToolPalette:
    """Build a palette directly from a list of skill ids."""
    return ToolPalette(
        execute_rskill_ids=frozenset(skills),
        sensor_ids=frozenset(),
        node_ids=frozenset({"/openral/hal/so100"}),
    )


def _renderer_with_prompt(text: str = "pick the cube") -> ContextRenderer:
    """Renderer with one operator prompt enqueued."""
    r = ContextRenderer()
    r.append_prompt(PromptRecord(text=text, metadata_json="", stamp_ns=0))
    return r


class _CaptureProcessor:
    """Minimal structlog processor that buffers events for assertion.

    Drops every event (raises :exc:`structlog.DropEvent`) so test logs
    don't pollute pytest output.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, logger: Any, method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        del logger, method
        name = str(event_dict.pop("event", ""))
        self.events.append((name, dict(event_dict)))
        raise structlog.DropEvent


@pytest.fixture
def log_cap() -> Any:
    """Install a structlog capture processor and restore defaults after.

    Mirrors the fixture pattern in ``test_diagnostics_phase_timer.py``
    (CLAUDE.md §1.11).

    The fixture also rebinds the ``openral_reasoner.core.log`` module
    attribute after reconfiguring structlog so the already-imported
    module-level logger proxy picks up the capture processor even when
    ``cache_logger_on_first_use=True`` had been set by conftest (i.e.
    after the logger was cached on first use).
    """
    import openral_reasoner.core as _core_mod

    proc = _CaptureProcessor()
    structlog.reset_defaults()
    structlog.configure(processors=[proc])
    # Rebind the module-level logger so the new processor applies.
    old_log = _core_mod.log
    _core_mod.log = structlog.get_logger(_core_mod.__name__)
    try:
        yield proc
    finally:
        _core_mod.log = old_log
        structlog.reset_defaults()


def test_tick_dispatches_execute_skill_from_canned_response() -> None:
    """A canned ExecuteRskillTool flows through tick() and is returned verbatim."""
    palette = _palette("openral/rskill-pick-cube-so100")
    client = FakeToolUseClient(
        responses=[
            ExecuteRskillTool(
                rskill_id="openral/rskill-pick-cube-so100",
                prompt="pick the red cube",
            ),
        ],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    result = core.tick(
        world_state=None,
        renderer=_renderer_with_prompt(),
        palette=palette,
    )
    assert result.error is None
    assert result.suppressed_reason == ""
    assert isinstance(result.tool_call, ExecuteRskillTool)
    assert result.tool_call.rskill_id == "openral/rskill-pick-cube-so100"


def test_tick_drains_prompts_on_success() -> None:
    """Operator prompts are pull-once and cleared on a successful tick."""
    palette = _palette()  # empty palette is fine — EmitPrompt is always available
    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="acknowledged")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    renderer = _renderer_with_prompt()
    assert len(renderer.prompts) == 1
    core.tick(world_state=None, renderer=renderer, palette=palette)
    assert renderer.prompts == ()


def test_min_interval_suppresses_back_to_back_tick() -> None:
    """A tick within min_interval_s of the previous one is suppressed."""
    palette = _palette()
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    client = FakeToolUseClient(
        responses=[
            EmitPromptTool(target_topic="/openral/prompt", text="ok"),
            EmitPromptTool(target_topic="/openral/prompt", text="ok"),
        ],
    )
    core = ReasonerCore(client=client, min_interval_s=0.1, clock=clock)
    r1 = core.tick(world_state=None, renderer=_renderer_with_prompt(), palette=palette)
    assert r1.tool_call is not None
    # Advance the clock by less than min_interval_s and try again.
    clock_value = 0.05
    r2 = core.tick(world_state=None, renderer=_renderer_with_prompt(), palette=palette)
    assert r2.tool_call is None
    assert r2.suppressed_reason == "min_interval"


def test_force_bypasses_min_interval() -> None:
    """force=True overrides the min-interval gate (event preemption path)."""
    palette = _palette()
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    client = FakeToolUseClient(
        responses=[
            EmitPromptTool(target_topic="/openral/prompt", text=f"msg{i}") for i in range(3)
        ],
    )
    core = ReasonerCore(client=client, min_interval_s=0.1, clock=clock)
    core.tick(world_state=None, renderer=_renderer_with_prompt(), palette=palette)
    clock_value = 0.01
    r = core.tick(
        world_state=None,
        renderer=_renderer_with_prompt(),
        palette=palette,
        force=True,
    )
    assert r.tool_call is not None


def test_retry_cap_suppresses_after_n_identical_calls() -> None:
    """The identical call (same tool + same args) picked >retry_cap times is suppressed."""
    palette = _palette()
    clock_value = 0.0

    def clock() -> float:
        nonlocal clock_value
        clock_value += 1.0  # always past min_interval
        return clock_value

    # 4 byte-identical EmitPromptTool — cap is 3 by default → 4th tick suppressed.
    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="same") for _ in range(4)],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0, retry_cap_per_kind=3, clock=clock)
    # Reuse one renderer and push a fresh prompt before each tick so
    # ContextRenderer.seq advances — without that the new heartbeat_idle
    # gate suppresses tick 2 before
    # the retry-cap gate ever runs.
    renderer = ContextRenderer()
    results = []
    for i in range(4):
        renderer.append_prompt(PromptRecord(text=f"p{i}", metadata_json="", stamp_ns=i))
        results.append(core.tick(world_state=None, renderer=renderer, palette=palette))
    assert [r.tool_call is not None for r in results] == [True, True, True, False]
    assert results[-1].suppressed_reason == "retry_cap"


def test_retry_cap_ignores_rationale_rephrasing() -> None:
    """A loop that only rephrases its ``rationale`` between retries is still capped."""
    palette = _palette()
    client = FakeToolUseClient(
        responses=[
            EmitPromptTool(target_topic="/openral/prompt", text="same", rationale=f"why {i}")
            for i in range(4)
        ],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0, retry_cap_per_kind=3)
    renderer = ContextRenderer()
    results = []
    for i in range(4):
        renderer.append_prompt(PromptRecord(text=f"p{i}", metadata_json="", stamp_ns=i))
        results.append(core.tick(world_state=None, renderer=renderer, palette=palette))
    assert results[-1].suppressed_reason == "retry_cap"


def test_same_kind_different_args_never_trips_the_cap() -> None:
    """A healthy sequence of *different* calls of the same kind is not a retry storm.

    The pre-fix cap keyed on the bare tool kind, so navigate → pick → place
    (all ``execute_rskill``) tripped it exactly like a genuine loop. Identity
    keying (tool + args) lets legitimate sequential work through.
    """
    palette = _palette()
    client = FakeToolUseClient(
        responses=[
            EmitPromptTool(target_topic="/openral/prompt", text=f"step {i}") for i in range(6)
        ],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0, retry_cap_per_kind=3)
    renderer = ContextRenderer()
    results = []
    for i in range(6):
        renderer.append_prompt(PromptRecord(text=f"p{i}", metadata_json="", stamp_ns=i))
        results.append(core.tick(world_state=None, renderer=renderer, palette=palette))
    assert all(r.tool_call is not None for r in results)


def test_retry_cap_hold_skips_the_llm_call_until_context_moves() -> None:
    """After a retry_cap suppression, an unchanged-context tick is held BEFORE the LLM.

    The pre-fix flow paid for a full LLM round-trip on every capped tick just
    to discard the result. The hold releases the moment ``seq`` moves (the
    node's reflection feedback bumps it).
    """
    palette = _palette()
    same = EmitPromptTool(target_topic="/openral/prompt", text="same")
    client = FakeToolUseClient(responses=[same] * 10)
    core = ReasonerCore(client=client, min_interval_s=0.0, retry_cap_per_kind=2)
    renderer = ContextRenderer()
    results = []
    for i in range(3):
        renderer.append_prompt(PromptRecord(text=f"p{i}", metadata_json="", stamp_ns=i))
        results.append(core.tick(world_state=None, renderer=renderer, palette=palette))
    assert results[-1].suppressed_reason == "retry_cap"
    calls_after_cap = client.calls
    # Context unchanged → held before the LLM; no new client call.
    held = core.tick(world_state=None, renderer=renderer, palette=palette)
    assert held.suppressed_reason == "retry_cap_hold"
    assert client.calls == calls_after_cap
    # Context moves (e.g. the node appended a reflection) → the LLM runs again.
    renderer.append_prompt(PromptRecord(text="new info", metadata_json="", stamp_ns=99))
    resumed = core.tick(world_state=None, renderer=renderer, palette=palette)
    assert resumed.suppressed_reason != "retry_cap_hold"
    assert client.calls == calls_after_cap + 1


def test_reset_kind_streak_lets_the_next_same_kind_tick_through() -> None:
    """reset_kind_streak() clears the streak so the next same-kind tick is not
    suppressed — the mechanism the reasoner_node uses on mission-task advancement
    so a new task is not blocked by the retry_cap the just-finished
    task ended on.
    """
    palette = _palette()
    clock_value = 0.0

    def clock() -> float:
        nonlocal clock_value
        clock_value += 1.0
        return clock_value

    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text=f"m{i}") for i in range(4)],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0, retry_cap_per_kind=3, clock=clock)
    renderer = ContextRenderer()
    results = []
    for i in range(4):
        renderer.append_prompt(PromptRecord(text=f"p{i}", metadata_json="", stamp_ns=i))
        if i == 3:
            core.reset_kind_streak()  # simulate advancing to a new mission task
        results.append(core.tick(world_state=None, renderer=renderer, palette=palette))
    # Without the reset the 4th identical kind is suppressed (see
    # test_retry_cap_suppresses_after_n_identical_kinds); the reset lets it through.
    assert [r.tool_call is not None for r in results] == [True, True, True, True]
    assert results[-1].suppressed_reason == ""  # not suppressed


def test_different_kind_resets_retry_streak() -> None:
    """A different tool kind clears the streak — alternating calls never suppressed."""
    palette = _palette("openral/rskill-x")
    client = FakeToolUseClient(
        responses=[
            EmitPromptTool(target_topic="/openral/prompt", text="a"),
            ExecuteRskillTool(rskill_id="openral/rskill-x"),
            EmitPromptTool(target_topic="/openral/prompt", text="b"),
            ExecuteRskillTool(rskill_id="openral/rskill-x"),
        ],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0, retry_cap_per_kind=2)
    renderer = ContextRenderer()
    for i in range(4):
        # Bump renderer.seq each tick so the heartbeat_idle gate doesn't
        # short-circuit — same reasoning as test_retry_cap_*.
        renderer.append_prompt(PromptRecord(text=f"p{i}", metadata_json="", stamp_ns=i))
        r = core.tick(world_state=None, renderer=renderer, palette=palette)
        assert r.suppressed_reason == ""
        assert r.tool_call is not None


def test_provider_failure_surfaces_as_planning_error() -> None:
    """A provider exception is reported as ROSPlanningError on the tick result."""
    palette = _palette()
    client = FakeToolUseClient(raise_on_call=ROSPlanningError("provider timeout"))
    core = ReasonerCore(client=client, min_interval_s=0.0)
    r = core.tick(world_state=None, renderer=_renderer_with_prompt(), palette=palette)
    assert r.tool_call is None
    assert isinstance(r.error, ROSPlanningError)


def test_execute_skill_outside_palette_is_rejected() -> None:
    """Canned ExecuteRskillTool whose rskill_id is not in the palette → ROSPlanningError."""
    palette = _palette("only/this/one")
    client = FakeToolUseClient(
        responses=[ExecuteRskillTool(rskill_id="not/in/palette")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    r = core.tick(world_state=None, renderer=_renderer_with_prompt(), palette=palette)
    # FakeToolUseClient raises ROSPlanningError directly (the real
    # AnthropicToolUseClient raises ROSReasonerInvalidPlan via the
    # decoder — both are ROSPlanningError subclasses).
    assert r.tool_call is None
    assert isinstance(r.error, ROSPlanningError)


def test_palette_empty_short_circuits_tick() -> None:
    """An empty palette + no prompts → tick suppressed with palette_empty."""
    palette = ToolPalette(execute_rskill_ids=frozenset())
    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="ok")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    r = core.tick(world_state=None, renderer=ContextRenderer(), palette=palette)
    assert r.tool_call is None
    assert r.suppressed_reason == "palette_empty"


def test_force_bypasses_palette_empty_short_circuit() -> None:
    """force=True bypasses the palette-empty gate (event-preemption path).

    The contract of ``force=True`` is "an event demands attention,
    bypass the gating heuristics". A SEVERITY_FAIL preemption on a
    bare reasoner (no installed skills) must still reach the LLM so
    it can pick :class:`EmitPromptTool` to escalate to the operator.
    """
    palette = ToolPalette(execute_rskill_ids=frozenset())
    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="ok")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    r = core.tick(
        world_state=None,
        renderer=ContextRenderer(),
        palette=palette,
        force=True,
    )
    assert r.tool_call is not None
    assert isinstance(r.tool_call, EmitPromptTool)
    assert r.suppressed_reason == ""


def test_min_interval_validation() -> None:
    """min_interval_s must be >= 0; retry_cap_per_kind must be >= 1."""
    client = FakeToolUseClient(responses=[])
    with pytest.raises(ValueError, match="min_interval_s"):
        ReasonerCore(client=client, min_interval_s=-1.0)
    with pytest.raises(ValueError, match="retry_cap_per_kind"):
        ReasonerCore(client=client, retry_cap_per_kind=0)


def test_lifecycle_transition_round_trips_through_tick() -> None:
    """LifecycleTransitionTool flows through unchanged."""
    palette = _palette()
    client = FakeToolUseClient(
        responses=[LifecycleTransitionTool(node="/openral/hal/so100", transition="activate")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    r = core.tick(world_state=None, renderer=_renderer_with_prompt(), palette=palette)
    assert isinstance(r.tool_call, LifecycleTransitionTool)
    assert r.tool_call.node == "/openral/hal/so100"
    # ROSReasonerInvalidPlan is re-exported on core for ergonomic tests.
    assert ROSReasonerInvalidPlan is not None


# ── heartbeat_idle suppression ─────────────────────────────────────────────────


def test_heartbeat_idle_suppresses_when_renderer_unchanged() -> None:
    """A non-forced tick whose renderer hasn't budged since the last tick is suppressed.

    The reasoner is event-driven with a slow heartbeat; the LLM call is
    wasted when no new failure / prompt / perception event has arrived since
    the last successful tick.
    """
    palette = _palette("openral/rskill-x")
    client = FakeToolUseClient(
        responses=[
            ExecuteRskillTool(rskill_id="openral/rskill-x"),
            ExecuteRskillTool(rskill_id="openral/rskill-x"),  # unused — second tick is suppressed
        ],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    renderer = _renderer_with_prompt()
    # First tick consumes the prompt; renderer.seq is already past
    # ContextRenderer.__init__'s sentinel by the time we tick.
    r1 = core.tick(world_state=None, renderer=renderer, palette=palette)
    assert r1.tool_call is not None
    # No new events: renderer.seq unchanged → heartbeat_idle.
    r2 = core.tick(world_state=None, renderer=renderer, palette=palette)
    assert r2.tool_call is None
    assert r2.suppressed_reason == "heartbeat_idle"


def test_forced_tick_bypasses_heartbeat_idle() -> None:
    """force=True (event preemption) bypasses the heartbeat-idle gate.

    The contract of ``force=True`` is "an event demands attention,
    bypass the gating heuristics" — same as it does for the
    palette-empty and min-interval gates.
    """
    palette = _palette("openral/rskill-x")
    client = FakeToolUseClient(
        responses=[
            ExecuteRskillTool(rskill_id="openral/rskill-x"),
            EmitPromptTool(target_topic="/openral/prompt", text="forced"),
        ],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    renderer = _renderer_with_prompt()
    core.tick(world_state=None, renderer=renderer, palette=palette)
    # No new events on the renderer, but force=True must reach the LLM.
    r2 = core.tick(world_state=None, renderer=renderer, palette=palette, force=True)
    assert r2.tool_call is not None
    assert r2.suppressed_reason == ""


def test_new_event_resets_heartbeat_idle_gate() -> None:
    """Any renderer mutation (new failure / prompt / perception) unlocks the next tick."""
    palette = _palette()
    client = FakeToolUseClient(
        responses=[
            EmitPromptTool(target_topic="/openral/prompt", text="a"),
            EmitPromptTool(target_topic="/openral/prompt", text="b"),
        ],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    renderer = _renderer_with_prompt("first")
    r1 = core.tick(world_state=None, renderer=renderer, palette=palette)
    assert r1.tool_call is not None
    # Push a new operator prompt onto the renderer. seq increments;
    # the next heartbeat tick should NOT be suppressed.
    renderer.append_prompt(PromptRecord(text="second", metadata_json="", stamp_ns=1))
    r2 = core.tick(world_state=None, renderer=renderer, palette=palette)
    assert r2.tool_call is not None
    assert r2.suppressed_reason == ""


def test_finished_mission_suppresses_heartbeat_even_when_context_changes() -> None:
    """Perception churn after handoff must not restart planning."""
    palette = _palette()
    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="should not run")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    renderer = ContextRenderer()
    renderer.set_mission(MissionState.from_prompt("find the teapot"))
    renderer.advance_mission(done=False, verdict="search exhausted")
    renderer.append_perception(
        PerceptionEventRecord(
            kind="objects",
            text="scene changed after handoff",
            metadata_json="{}",
            stamp_ns=1,
        )
    )

    result = core.tick(world_state=None, renderer=renderer, palette=palette)

    assert result.suppressed_reason == "mission_finished"
    assert client.calls == 0


# ── Multi-task structured observability ───────────────────────────────────


def test_tick_emits_reasoner_tick_selected_log(log_cap: _CaptureProcessor) -> None:
    """tick() emits a ``reasoner.tick.selected`` structured log on every success.

    This covers the observability requirement added with the multi-task prompt
    path: every successful reasoner decision must emit a structured log event
    with at minimum ``tick_idx``, ``tool``, and ``elapsed_s``.
    """
    palette = _palette("openral/rskill-pick-cube-so100")
    client = FakeToolUseClient(
        responses=[
            ExecuteRskillTool(
                rskill_id="openral/rskill-pick-cube-so100",
                prompt="pick the black bowl",
            ),
        ],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    core.tick(
        world_state=None,
        renderer=_renderer_with_prompt("pick the black bowl"),
        palette=palette,
    )
    selected = [ev for name, ev in log_cap.events if name == "reasoner.tick.selected"]
    assert len(selected) == 1, (
        f"expected exactly 1 reasoner.tick.selected event; got {log_cap.events}"
    )
    ev = selected[0]
    assert "tick_idx" in ev
    assert "tool" in ev
    assert "elapsed_s" in ev
    assert ev["tool"] == "execute_rskill"
    assert ev.get("rskill_id") == "openral/rskill-pick-cube-so100"


def test_tick_selected_log_includes_active_prompt(log_cap: _CaptureProcessor) -> None:
    """``active_prompt`` field in reasoner.tick.selected carries the operator goal text."""
    palette = _palette()  # empty — EmitPrompt is always available
    client = FakeToolUseClient(
        responses=[EmitPromptTool(target_topic="/openral/prompt", text="acknowledged")],
    )
    core = ReasonerCore(client=client, min_interval_s=0.0)
    prompt_text = "pick the black bowl and place it in the basket"
    core.tick(
        world_state=None,
        renderer=_renderer_with_prompt(prompt_text),
        palette=palette,
    )
    selected = [ev for name, ev in log_cap.events if name == "reasoner.tick.selected"]
    assert len(selected) == 1
    ev = selected[0]
    # active_prompt may be truncated to 200 chars; must start with the same text
    active = ev.get("active_prompt", "")
    assert isinstance(active, str)
    assert active.startswith(prompt_text[:30])


def test_search_tools_are_transparent_to_the_retry_cap() -> None:
    """Identical read-only search calls are never capped — their own budgets
    (SearchProgress miss budget, TaskLocateBudget) bound those loops and must
    reach the explicit human-handoff, which the cap's silent hold would preempt
    (the live regression: the recall cascade stalled in retry_cap_hold before
    the search budget could hand off)."""
    from openral_core import RecallObjectTool

    palette = _palette()
    client = FakeToolUseClient(responses=[RecallObjectTool(query="milk") for _ in range(6)])
    core = ReasonerCore(client=client, min_interval_s=0.0, retry_cap_per_kind=3)
    renderer = ContextRenderer()
    results = []
    for i in range(6):
        renderer.append_prompt(PromptRecord(text=f"p{i}", metadata_json="", stamp_ns=i))
        results.append(core.tick(world_state=None, renderer=renderer, palette=palette))
    assert all(r.tool_call is not None for r in results)


def test_interleaved_search_calls_do_not_reset_a_capped_streak() -> None:
    """Transparency, not reset: an alternating <same-call> / <search> pattern
    still accumulates toward the cap — a genuine loop cannot launder its streak
    through a read-only recall between repeats."""
    from openral_core import RecallObjectTool

    palette = _palette()
    same = EmitPromptTool(target_topic="/openral/prompt", text="same")
    responses: list = []
    for _ in range(4):
        responses.append(same)
        responses.append(RecallObjectTool(query="milk"))
    client = FakeToolUseClient(responses=responses)
    core = ReasonerCore(client=client, min_interval_s=0.0, retry_cap_per_kind=3)
    renderer = ContextRenderer()
    results = []
    for i in range(8):
        renderer.append_prompt(PromptRecord(text=f"p{i}", metadata_json="", stamp_ns=i))
        results.append(core.tick(world_state=None, renderer=renderer, palette=palette))
    # The 4th identical emit_prompt (tick index 6) trips the cap even though
    # recall_object calls were interleaved between the repeats.
    suppressed = [r.suppressed_reason for r in results]
    assert "retry_cap" in suppressed


def test_wait_is_transparent_to_the_retry_cap() -> None:
    """Consecutive waits are never capped — the system prompt INSTRUCTS the
    model to keep picking wait during a nominal long skill execution, and every
    wait is byte-identical (rationale is stripped from the call identity), so a
    counted wait would trip the cap after retry_cap heartbeats of prescribed
    behavior and inject a fabricated "retry ladder exhausted" failure into
    context mid-run."""
    from openral_core import WaitTool

    palette = _palette()
    client = FakeToolUseClient(responses=[WaitTool() for _ in range(6)])
    core = ReasonerCore(client=client, min_interval_s=0.0, retry_cap_per_kind=3)
    renderer = ContextRenderer()
    results = []
    for i in range(6):
        renderer.append_prompt(PromptRecord(text=f"p{i}", metadata_json="", stamp_ns=i))
        results.append(core.tick(world_state=None, renderer=renderer, palette=palette))
    assert all(r.tool_call is not None for r in results)
    assert all(not r.suppressed_reason for r in results)
