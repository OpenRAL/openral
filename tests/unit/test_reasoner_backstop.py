"""Event-driven reasoner ticks (issue #620): backstop, ``wait(until=…)``, LLM retries.

Real ``ContextRenderer`` + ``ToolPalette`` + Pydantic tool calls; the LLM is the
deterministic ``FakeToolUseClient`` (CLAUDE.md §1.11 process-boundary fake) and
the clock is injected so every period is exact.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from openral_core import EmitPromptTool, ReasonerToolCall, RSkillAction, WaitTool
from openral_core.exceptions import ROSPlanningError
from openral_observability import semconv
from openral_reasoner import (
    ContextRenderer,
    MissionState,
    PromptRecord,
    ReasonerCore,
    ToolPalette,
)
from openral_reasoner.context import PerceptionEventRecord
from openral_reasoner.core import (
    BACKSTOP_MAX_S,
    BACKSTOP_MIN_S,
    LLM_ERROR_BACKOFF_S,
    PreparedTick,
    ReasonerTickResult,
    inflight_backstop_s,
)
from openral_reasoner.palette import RSkillToolEntry
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from tests.integration.fakes.fake_llm import FakeToolUseClient

_SKILL = "openral/rskill-pick"


def _palette(max_execution_s: float | None = None) -> ToolPalette:
    return ToolPalette(
        skills=(
            RSkillToolEntry(
                rskill_id=_SKILL,
                description="pick",
                actions=(RSkillAction.PICK,),
                max_execution_s=max_execution_s,
            ),
        ),
    )


def _script(*steps: ReasonerToolCall | Exception) -> Callable[[str, ToolPalette], ReasonerToolCall]:
    """A selector replaying ``steps`` in order; an exception step is raised."""
    queue = list(steps)

    def select(_context: str, _palette: ToolPalette) -> ReasonerToolCall:
        step = queue.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    return select


def _emit(text: str = "x") -> EmitPromptTool:
    return EmitPromptTool(target_topic="/openral/prompt", text=text)


class _Rig:
    """A core, a renderer, and a clock the test advances."""

    def __init__(self, *steps: ReasonerToolCall | Exception, palette: ToolPalette | None = None):
        self.now = 0.0
        self.client = FakeToolUseClient(selector=_script(*steps))
        self.core = ReasonerCore(client=self.client, min_interval_s=0.0, clock=lambda: self.now)
        self.renderer = ContextRenderer()
        self.palette = palette or _palette()

    def tick(self, *, force: bool = False, trigger: str = "heartbeat") -> ReasonerTickResult:
        return self.core.tick(
            world_state=None,
            renderer=self.renderer,
            palette=self.palette,
            force=force,
            trigger=trigger,
        )


# ── backstop period ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(None, BACKSTOP_MIN_S), (10.0, BACKSTOP_MIN_S), (100.0, 25.0), (1000.0, BACKSTOP_MAX_S)],
)
def test_inflight_backstop_is_a_clamped_quarter_of_the_budget(
    budget: float | None, expected: float
) -> None:
    assert inflight_backstop_s(budget) == expected


def test_running_skill_with_no_events_calls_the_llm_once_per_backstop_period() -> None:
    """Acceptance (#620): at most one LLM call per backstop period while a skill runs."""
    rig = _Rig(*[_emit(str(i)) for i in range(10)], palette=_palette(max_execution_s=100.0))
    rig.renderer.set_inflight_skill(_SKILL, stamp_ns=1)
    assert rig.tick(force=True, trigger="skill_dispatched").tool_call is not None
    calls = 0
    for _ in range(100):  # 100 s of 1 Hz timer beats
        rig.now += 1.0
        if rig.tick().tool_call is not None:
            calls += 1
    assert calls == 100 // 25  # period = clamp(100 / 4) = 25 s


def test_perception_churn_during_a_skill_is_bounded_by_the_backstop() -> None:
    rig = _Rig(*[_emit(str(i)) for i in range(10)])
    rig.renderer.set_inflight_skill(_SKILL, stamp_ns=1)
    rig.tick(force=True)
    rig.now += 1.0
    rig.renderer.append_perception(
        PerceptionEventRecord(kind="motion", text="arm moved", metadata_json="", stamp_ns=2)
    )
    assert rig.tick().suppressed_reason == "backstop_not_due"
    rig.now += BACKSTOP_MIN_S
    result = rig.tick()
    assert result.tool_call is not None
    assert result.trigger == "context_changed"


def test_idle_with_no_mission_never_calls_the_llm() -> None:
    """Perception churn with no mission, no prompt and nothing in flight costs no call."""
    rig = _Rig(_emit())
    rig.renderer.append_prompt(PromptRecord(text="hi", metadata_json="", stamp_ns=0))
    assert rig.tick(force=True, trigger="prompt").trigger == "prompt"
    for i in range(120):
        rig.now += 1.0
        rig.renderer.append_perception(
            PerceptionEventRecord(kind="motion", text=f"m{i}", metadata_json="", stamp_ns=i)
        )
        assert rig.tick().suppressed_reason == "no_goal"


# ── wait(until=…) ────────────────────────────────────────────────────────────


def test_wait_until_skill_result_holds_through_perception_and_backstop() -> None:
    rig = _Rig(WaitTool(until="skill_result", timeout_s=60.0), _emit())
    rig.renderer.set_inflight_skill(_SKILL, stamp_ns=1)
    rig.tick(force=True)
    assert rig.core.wait_hold == ("skill_result", 60.0)
    for i in range(50):
        rig.now += 1.0
        rig.renderer.append_perception(
            PerceptionEventRecord(kind="motion", text=f"m{i}", metadata_json="", stamp_ns=i)
        )
        assert rig.tick().suppressed_reason == "wait_hold"
    # The skill's terminal result is an event: the node forces a tick for it.
    rig.renderer.set_inflight_skill(None)
    result = rig.tick(force=True, trigger="skill_result")
    assert result.tool_call is not None
    assert result.trigger == "skill_result"
    assert rig.core.wait_hold is None


def test_wait_timeout_releases_one_llm_call() -> None:
    rig = _Rig(WaitTool(until="timeout", timeout_s=20.0), _emit())
    rig.renderer.set_inflight_skill(_SKILL, stamp_ns=1)
    rig.tick(force=True)
    rig.now += 19.0
    assert rig.tick().suppressed_reason == "wait_hold"
    rig.now += 1.0
    result = rig.tick()
    assert result.tool_call is not None
    assert result.trigger == "wait_timeout"


def test_wait_until_next_event_ends_on_new_context() -> None:
    rig = _Rig(WaitTool(), _emit())
    rig.renderer.set_mission(MissionState.from_prompt("watch the shelf"))
    rig.tick(force=True)
    rig.now += 1.0
    assert rig.tick().suppressed_reason == "wait_hold"
    rig.renderer.append_perception(
        PerceptionEventRecord(kind="motion", text="person", metadata_json="", stamp_ns=3)
    )
    rig.now += 1.0
    result = rig.tick()
    assert result.tool_call is not None
    assert result.trigger == "context_changed"


def test_forced_tick_breaks_any_wait() -> None:
    rig = _Rig(WaitTool(until="operator", timeout_s=300.0), _emit())
    rig.renderer.set_inflight_skill(_SKILL, stamp_ns=1)
    rig.tick(force=True)
    rig.now += 1.0
    result = rig.tick(force=True, trigger="failure:safety")
    assert result.tool_call is not None
    assert result.trigger == "failure:safety"


def test_wait_timeout_is_bounded() -> None:
    with pytest.raises(ValidationError):
        WaitTool(timeout_s=0.0)
    with pytest.raises(ValidationError):
        WaitTool(timeout_s=301.0)


# ── LLM transport errors ─────────────────────────────────────────────────────


def test_llm_error_is_retried_with_backoff_then_gives_up() -> None:
    errors = [ROSPlanningError("timeout") for _ in range(len(LLM_ERROR_BACKOFF_S) + 1)]
    rig = _Rig(*errors, _emit())
    rig.renderer.append_prompt(PromptRecord(text="go", metadata_json="", stamp_ns=0))
    assert rig.tick(force=True).error is not None
    for delay in LLM_ERROR_BACKOFF_S:
        rig.now += delay - 0.5
        assert rig.tick().suppressed_reason == "llm_error_backoff"
        rig.now += 0.5
        retried = rig.tick()
        assert retried.error is not None
        assert retried.trigger == "llm_error_retry"
    assert rig.core.llm_gave_up
    # Out of retries: the unchanged context is no longer re-sent on its own.
    rig.now += 100.0
    assert rig.tick().suppressed_reason == "heartbeat_idle"
    # The next event still reaches the model, and success clears the give-up.
    rig.renderer.append_prompt(PromptRecord(text="again", metadata_json="", stamp_ns=1))
    assert rig.tick(force=True).tool_call is not None
    assert not rig.core.llm_gave_up


def test_llm_error_retry_runs_even_when_the_context_is_unchanged() -> None:
    """The model never saw the context the failed call carried: retry it."""
    rig = _Rig(_emit("first"), ROSPlanningError("503"), _emit("retry"))
    rig.renderer.append_prompt(PromptRecord(text="go", metadata_json="", stamp_ns=0))
    assert rig.tick(force=True).tool_call is not None
    assert rig.tick(force=True).error is not None  # same seq, forced
    rig.now += LLM_ERROR_BACKOFF_S[0]
    assert rig.tick().tool_call == _emit("retry")


# ── observability ────────────────────────────────────────────────────────────


def test_trigger_and_wait_land_on_the_tick_span(exporter: InMemorySpanExporter) -> None:
    rig = _Rig(WaitTool(until="skill_result", timeout_s=45.0))
    rig.renderer.append_prompt(PromptRecord(text="go", metadata_json="", stamp_ns=0))
    prep = rig.core.prepare_tick(
        world_state=None,
        renderer=rig.renderer,
        palette=rig.palette,
        force=True,
        trigger="prompt",
        coalesced=("failure:hal",),
    )
    assert isinstance(prep, PreparedTick)
    rig.core.finish_tick(prep, call=rig.core.run_prepared_llm(prep))
    (span,) = exporter.get_finished_spans()
    attrs = span.attributes
    assert attrs is not None
    assert attrs[semconv.REASONER_TRIGGER] == "prompt"
    assert tuple(attrs[semconv.REASONER_COALESCED_TRIGGERS]) == ("failure:hal",)  # type: ignore[arg-type]  # reason: OTel sequence attr
    assert attrs[semconv.REASONER_SKIPPED_SINCE_LAST] == 0
    assert attrs[semconv.REASONER_WAIT_UNTIL] == "skill_result"
    assert attrs[semconv.REASONER_WAIT_TIMEOUT_S] == 45.0


def test_event_ticks_failing_during_the_backoff_do_not_spend_the_retry_budget() -> None:
    """A burst of events in a short outage must not exhaust the 5/10/20 s retries."""
    rig = _Rig(*[ROSPlanningError("503") for _ in range(6)], _emit("back"))
    rig.renderer.set_mission(MissionState.from_prompt("pick the cube"))
    assert rig.tick(force=True, trigger="prompt").error is not None
    for event in ("failure:safety", "skill_failure", "prompt", "failure:hal"):
        rig.now += 0.5
        assert rig.tick(force=True, trigger=event).error is not None
    assert not rig.core.llm_gave_up
    rig.now += LLM_ERROR_BACKOFF_S[0]
    retried = rig.tick()
    assert retried.trigger == "llm_error_retry"
    assert not rig.core.llm_gave_up
