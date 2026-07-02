"""ADR-0018 F4 — :class:`ReasonerCore`.

The transport-agnostic orchestrator that closes
context → LLM → typed tool call. The ROS-side
``openral_reasoner_ros.reasoner_node`` wraps this class with the
lifecycle, subscriptions, and dispatch plumbing; the core itself has
no rclpy dependency so it is fully unit-testable against a
:class:`FakeToolUseClient`.

Per ADR-0018 §4 the reasoner:

* Holds **no** authority over actuation (never publishes
  ``ActionChunk``).
* Picks exactly one of four typed tool calls per tick.
* Enforces a bounded retry counter per failure kind to prevent storms.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable

import structlog
from openral_core import ReasonerToolCall, WorldState
from openral_core.exceptions import (
    ROSPlanningError,
    ROSReasonerInvalidPlan,
)
from openral_observability import reasoner_span, semconv
from openral_observability.propagation import current_traceparent
from opentelemetry.trace import Span

from openral_reasoner.context import ContextRenderer
from openral_reasoner.palette import ToolPalette
from openral_reasoner.tool_use import DEFAULT_SYSTEM_PROMPT, ToolUseClient

__all__ = ["ReasonerCore", "ReasonerTickResult"]

log = structlog.get_logger(__name__)


def _stamp_mission(span: Span, renderer: ContextRenderer) -> None:
    """Stamp the active mission queue on the tick span (ADR-0073).

    Serialized as ``reasoner.mission_json`` so the live dashboard renders the
    task checklist. No-op when no mission is set (a bare operator goal).
    """
    mission = renderer.mission
    if mission is not None and not mission.is_empty():
        span.set_attribute(semconv.REASONER_MISSION_JSON, json.dumps(mission.to_summary()))


@dataclasses.dataclass(frozen=True, slots=True)
class ReasonerTickResult:
    """Outcome of a single :meth:`ReasonerCore.tick` invocation.

    Attributes:
        tool_call: The validated tool call selected by the LLM, or
            ``None`` when the tick was suppressed (rate-limited,
            palette empty, etc.).
        error: A :class:`ROSPlanningError` subclass when the tick
            failed, or ``None`` on success.
        elapsed_s: Wall-clock time the tick took, end-to-end.
        suppressed_reason: When ``tool_call is None and error is None``,
            a short string explaining why the tick was suppressed
            (e.g. ``"min_interval"``, ``"retry_cap"``,
            ``"palette_empty"``, ``"heartbeat_idle"``).
    """

    tool_call: ReasonerToolCall | None
    error: ROSPlanningError | None
    elapsed_s: float
    suppressed_reason: str = ""
    traceparent: str | None = None
    """W3C ``traceparent`` captured while the ``reasoner.tick`` span was
    active. The reasoner_node stamps this onto the outbound
    ``EmitPromptTool`` ``PromptStamped.metadata_json`` so the F7
    bag↔OTel correlator can join the published prompt back to the
    reasoner span that produced it (ADR-0018 §6). ``None`` when no
    real :class:`TracerProvider` is installed."""


class ReasonerCore:
    """Transport-agnostic reasoner orchestrator.

    The class is intentionally narrow: it consumes the current
    :class:`ContextRenderer` + :class:`ToolPalette` and produces a
    typed tool call. Wiring (subscriptions, action clients, service
    calls) lives in the ROS lifecycle node.

    Args:
        client: A :class:`ToolUseClient` instance. In tests this is
            usually :class:`FakeToolUseClient` (under
            ``tests/integration/fakes/``); in production it is one of
            the SDK-backed clients from :mod:`openral_reasoner.tool_use`.
        min_interval_s: Hard lower bound between consecutive ticks, in
            seconds. ADR-0018 §4 mandates 100 ms (0.1 s).
        retry_cap_per_kind: Maximum number of consecutive ticks the
            reasoner may select the same tool kind before being
            suppressed by ``retry_cap`` for one tick. Defaults to 3 —
            matches the replanning ladder guidance in CLAUDE.md §7.6.
        system_prompt: Override the
            :data:`~openral_reasoner.tool_use.DEFAULT_SYSTEM_PROMPT`.
            ``None`` keeps the default.
        clock: Monotonic clock source (seconds). Override in tests.

    Example:
        >>> # End-to-end is exercised in tests/integration/test_reasoner_core.py
        >>> import openral_reasoner.core as core
        >>> hasattr(core, "ReasonerCore")
        True
    """

    def __init__(
        self,
        *,
        client: ToolUseClient,
        min_interval_s: float = 0.1,
        retry_cap_per_kind: int = 3,
        system_prompt: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Stash configuration; no LLM call until :meth:`tick`."""
        if min_interval_s < 0:
            raise ValueError(
                f"ReasonerCore.min_interval_s must be >= 0; got {min_interval_s!r}",
            )
        if retry_cap_per_kind < 1:
            raise ValueError(
                f"ReasonerCore.retry_cap_per_kind must be >= 1; got {retry_cap_per_kind!r}",
            )
        self._client = client
        self._min_interval_s = min_interval_s
        self._retry_cap = retry_cap_per_kind
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._clock = clock
        self._last_tick_s: float = -float("inf")
        self._kind_streak: tuple[str, int] = ("", 0)
        self._tick_idx: int = 0
        # Tracks ``ContextRenderer.seq`` at the time of the last
        # successful (or non-idle-suppressed) tick. A heartbeat tick
        # (force=False) whose renderer hasn't budged since
        # ``_last_seen_seq`` is suppressed with ``heartbeat_idle`` —
        # the LLM call would see byte-identical context and produce
        # the same (or no) tool call. ADR-0018 amendment 2026-05-25 §2.
        self._last_seen_seq: int = -1

    def reset_kind_streak(self) -> None:
        """Reset the consecutive-tool counter used by the retry-cap gate.

        Called by the reasoner_node whenever the situation changes
        materially (new operator prompt, palette refresh, etc.). The
        retry-cap exists to prevent the model from looping on the same
        failure mode against a static context; once the context shifts
        (e.g. an operator types a new task), the previous streak
        carries no information and would otherwise silently swallow
        the next tool call.
        """
        self._kind_streak = ("", 0)

    def tick(
        self,
        *,
        world_state: WorldState | None,
        renderer: ContextRenderer,
        palette: ToolPalette,
        force: bool = False,
        tier: str = "heartbeat",
    ) -> ReasonerTickResult:
        """Run one orchestrator pass.

        Args:
            world_state: Latest WorldState snapshot or ``None``.
            renderer: The reasoner's :class:`ContextRenderer`. Prompts
                are drained on a successful tick.
            palette: Current :class:`ToolPalette`.
            force: When ``True`` bypasses **both** gating heuristics
                (the min-interval rate-limit and the palette-empty
                short-circuit) so an event-preempted tick from a
                ``FailureTrigger.severity >= FAIL`` or a new
                ``PromptStamped`` always reaches the LLM. The
                retry-cap gate still applies — a forced tick is not a
                license to loop on the same kind.
            tier: Trigger tier driving this call. ``"A"`` (safety) /
                ``"B"`` (replan: hal/sensor/rskill/wam) / ``"C"``
                (critic) / ``"D"`` (operator/perception) for event
                preemptions, or ``"heartbeat"`` (default) when the
                periodic timer fired with no preempting callback.
                Recorded on the OTel span as ``reasoner.tier`` for
                trace-filtering on the dashboard — observability only;
                per-tier preemption thresholds live in
                :class:`~openral_reasoner_ros.ReasonerNode`.

        Returns:
            A :class:`ReasonerTickResult`.
        """
        started = self._clock()
        # min-interval gate (ADR-0018 §4) — gate BEFORE opening the
        # OTel span so suppressed ticks don't show up in the trace.
        if not force and started - self._last_tick_s < self._min_interval_s:
            return ReasonerTickResult(
                tool_call=None,
                error=None,
                elapsed_s=0.0,
                suppressed_reason="min_interval",
            )
        # heartbeat-idle gate (ADR-0018 amendment 2026-05-25 §2) — gate
        # BEFORE the OTel span for the same reason. A non-forced tick
        # whose ContextRenderer has not received any new failure /
        # perception / prompt event since the last tick is suppressed:
        # the LLM would see byte-identical context and the call is
        # wasted. Forced ticks (event preemption) bypass this gate by
        # the ``force`` flag itself.
        if not force and renderer.seq == self._last_seen_seq:
            self._last_tick_s = started
            return ReasonerTickResult(
                tool_call=None,
                error=None,
                elapsed_s=0.0,
                suppressed_reason="heartbeat_idle",
            )
        # ADR-0018 §6 / ADR-0017: every tick that reaches the LLM (or
        # the palette-empty short-circuit) opens a ``reasoner.tick``
        # span. The reasoner_node reads ``current_traceparent()`` from
        # inside this scope to stamp the outbound EmitPrompt's
        # metadata_json.
        self._tick_idx += 1
        with reasoner_span(
            tick_idx=self._tick_idx,
            model=getattr(self._client, "model_id", None),
            force=force,
        ) as span:
            span.set_attribute(semconv.REASONER_TIER, tier)
            # ADR-0073 — stamp the active mission queue on every tick span so
            # the dashboard can render the task checklist. Set before the gate
            # short-circuits below so suppressed (retry_cap / error) ticks still
            # carry current mission state. The mission is unchanged within a tick.
            _stamp_mission(span, renderer)
            # palette-empty short-circuit — the LLM call would just
            # timeout / pick a phantom rskill_id; surface the
            # configuration error explicitly.
            #
            # Bypassed when ``force=True``: an event-preempted tick
            # (SEVERITY_FAIL FailureTrigger or a new operator prompt)
            # demands the LLM's attention even when no skills are
            # installed — at minimum the LLM can pick :class:`EmitPromptTool`
            # to escalate to the operator. The contract of ``force=True``
            # is "an event demands attention, bypass the gating
            # heuristics" — gating it here would silently swallow
            # SEVERITY_FAIL preemptions on a bare reasoner.
            if (
                not force
                and not palette.execute_rskill_ids
                and not (palette.sensor_ids or palette.node_ids or renderer.prompts)
            ):
                self._last_tick_s = started
                self._last_seen_seq = renderer.seq
                span.set_attribute(semconv.REASONER_SUPPRESSED_REASON, "palette_empty")
                return ReasonerTickResult(
                    tool_call=None,
                    error=None,
                    elapsed_s=self._clock() - started,
                    suppressed_reason="palette_empty",
                )
            context_text = renderer.render(world_state=world_state)
            try:
                call = self._client.select_tool(
                    context_text=context_text,
                    palette=palette,
                    system_prompt=self._system_prompt,
                )
            except ROSPlanningError as exc:
                self._last_tick_s = started
                self._last_seen_seq = renderer.seq
                span.set_attribute(semconv.REASONER_ERROR_KIND, type(exc).__name__)
                span.record_exception(exc)
                return ReasonerTickResult(
                    tool_call=None,
                    error=exc,
                    elapsed_s=self._clock() - started,
                )
            # retry-cap (ADR-0018 §4 "bounded retry counter per failure kind")
            prev_kind, streak = self._kind_streak
            if call.tool == prev_kind:
                streak += 1
            else:
                streak = 1
            if streak > self._retry_cap:
                self._kind_streak = (call.tool, streak)
                self._last_tick_s = started
                self._last_seen_seq = renderer.seq
                span.set_attribute(semconv.REASONER_TOOL, call.tool)
                span.set_attribute(semconv.REASONER_SUPPRESSED_REASON, "retry_cap")
                return ReasonerTickResult(
                    tool_call=None,
                    error=None,
                    elapsed_s=self._clock() - started,
                    suppressed_reason="retry_cap",
                )
            self._kind_streak = (call.tool, streak)
            self._last_tick_s = started
            self._last_seen_seq = renderer.seq
            span.set_attribute(semconv.REASONER_TOOL, call.tool)
            # ExecuteRskillTool carries a rskill_id worth surfacing on the
            # span for trace-search drill-down; other variants don't have
            # an obvious "primary key" to record.
            rskill_id = getattr(call, "rskill_id", None)
            if rskill_id:
                span.set_attribute(semconv.REASONER_RSKILL_ID, rskill_id)
            # Structured log for every successful tick so the multi-task
            # deploy flow can be inspected from the structured log stream
            # without opening Jaeger (Criterion 4 of the libero multi-task
            # goal: subtask, rSkill, localization, VLA execution, reward
            # evaluation, and final outcome are all emitted here).
            _log_kwargs: dict[str, object] = {
                "tick_idx": self._tick_idx,
                "tool": call.tool,
                "tier": tier,
                "elapsed_s": round(self._clock() - started, 4),
            }
            if rskill_id:
                _log_kwargs["rskill_id"] = rskill_id
            rationale = getattr(call, "rationale", None)
            if rationale:
                _log_kwargs["rationale"] = rationale
            # Surface the active prompt so the caller can correlate which
            # operator goal drove this tool selection (multi-task tracing).
            if renderer.prompts:
                _log_kwargs["active_prompt"] = renderer.prompts[0].text[:200]
            log.info("reasoner.tick.selected", **_log_kwargs)
            # Capture the active traceparent WHILE the span is still in
            # scope so reasoner_node._dispatch (which runs after this
            # function returns) can stamp it onto outbound PromptStamped
            # metadata_json (ADR-0018 §6 "OTel context is the truth").
            traceparent = current_traceparent()
            # Drain operator prompts on a successful tick (pull-once semantics).
            renderer.drain_prompts()
            return ReasonerTickResult(
                tool_call=call,
                error=None,
                elapsed_s=self._clock() - started,
                traceparent=traceparent,
            )


# Re-export :class:`ROSReasonerInvalidPlan` so test code can ``from
# openral_reasoner.core import ROSReasonerInvalidPlan`` without
# reaching into ``openral_core.exceptions``.
__all__ += ["ROSReasonerInvalidPlan"]
