""":class:`ReasonerCore`.

The transport-agnostic orchestrator that closes
context → LLM → typed tool call. The ROS-side
``openral_reasoner_ros.reasoner_node`` wraps this class with the
lifecycle, subscriptions, and dispatch plumbing; the core itself has
no rclpy dependency so it is fully unit-testable against a
:class:`FakeToolUseClient`.

The reasoner:

* Holds **no** authority over actuation (never publishes
  ``ActionChunk``).
* Picks exactly one typed :data:`~openral_core.ReasonerToolCall` per tick.
* Enforces a bounded retry counter per identical call to prevent storms.
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
from openral_observability import semconv, start_reasoner_span
from openral_observability.propagation import current_traceparent
from opentelemetry.trace import Span, use_span

from openral_reasoner.context import ContextRenderer, PromptRecord
from openral_reasoner.palette import ToolPalette
from openral_reasoner.tool_use import DEFAULT_SYSTEM_PROMPT, ToolUseClient

__all__ = ["PreparedTick", "ReasonerCore", "ReasonerTickResult"]

log = structlog.get_logger(__name__)


def _stamp_mission(span: Span, renderer: ContextRenderer) -> None:
    """Stamp the active mission queue on the tick span.

    Serialized as ``reasoner.mission_json`` so the live dashboard renders the
    task checklist. No-op when no mission is set (a bare operator goal).
    """
    mission = renderer.mission
    if mission is not None and not mission.is_empty():
        span.set_attribute(semconv.REASONER_MISSION_JSON, json.dumps(mission.to_summary()))


# Read-only search tools are TRANSPARENT to the retry cap — neither counted
# nor streak-resetting. Each search loop already has its own dedicated,
# operator-facing bound (SearchProgress's miss budget and the per-task
# TaskLocateBudget, both terminating in an explicit human-handoff); letting
# the identity cap fire first would stop the loop in a silent retry_cap_hold
# BEFORE the search budget can hand off (caught live by
# test_active_search_cascade_is_bounded_and_hands_off). Transparency (rather
# than resetting) also keeps an alternating <same-call> / <search> loop
# accumulating toward the cap.
#
# "wait" is exempt for a different reason: the system prompt and the in_flight
# context line INSTRUCT the model to keep picking wait during a nominal long
# skill execution, and _call_identity strips rationale so every wait is
# byte-identical — counting it would trip the cap after retry_cap heartbeats
# of exactly the prescribed behavior and inject a fabricated "retry ladder
# exhausted" failure into context mid-run. Waiting is already bounded by the
# skill's own deadline/patience machinery, not the identity cap.
_RETRY_CAP_EXEMPT_TOOLS: frozenset[str] = frozenset(
    {"recall_object", "resolve_place", "locate_in_view", "wait"}
)


def _call_identity(call: ReasonerToolCall) -> str:
    """Canonical identity of a tool call for the retry-cap streak.

    The full argument payload minus ``rationale`` (the model rephrases its
    rationale freely between identical retries, so including it would let a
    genuine loop dodge the cap), JSON-serialised with sorted keys so the
    identity is byte-stable.
    """
    payload = {k: v for k, v in call.model_dump(mode="json").items() if k != "rationale"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclasses.dataclass(slots=True)
class PreparedTick:
    """In-flight state of a phased tick, prepare_tick → finish_tick.

    Carries everything the blocking LLM phase needs (context text, palette
    snapshot, system prompt) plus the bookkeeping captured at prepare time
    (``seq``, ``prompts``, ``started``, the open OTel span). The ``seq`` and
    ``prompts`` snapshots matter: events that arrive while the LLM call is in
    flight were **not** rendered into the model's context, so
    :meth:`~ReasonerCore.finish_tick` must mark seen / drain only what the
    model actually saw.
    """

    context_text: str
    palette: ToolPalette
    system_prompt: str
    renderer: ContextRenderer
    span: Span
    started: float
    seq: int
    prompts: tuple[PromptRecord, ...]
    force: bool
    tier: str
    llm_s: float = 0.0
    """Wall-clock of the LLM round-trip alone, written by
    :meth:`~ReasonerCore.run_prepared_llm` (the only field the LLM phase
    mutates) and read back by :meth:`~ReasonerCore.finish_tick`. Split out
    from the tick's ``elapsed_s`` because a slow tick is otherwise
    unattributable: provider time and reasoner overhead look identical."""
    prompt_tokens: int | None = None
    """Provider-reported prompt-token count for that call, or ``None`` when
    the client does not surface usage."""


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
            ``"retry_cap_hold"``, ``"palette_empty"``,
            ``"heartbeat_idle"``).
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
    reasoner span that produced it. ``None`` when no
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
            seconds. Mandated as 100 ms (0.1 s).
        retry_cap_per_kind: Maximum number of consecutive ticks the
            reasoner may select the **identical tool call** (same tool +
            same arguments, ``rationale`` excluded) before being
            suppressed by ``retry_cap``. Defaults to 3 — matches the
            replanning ladder guidance in CLAUDE.md §7.6. Keying on the
            full call identity (rather than the bare tool kind) means a
            healthy sequence of *different* skills — navigate → pick →
            place, all ``execute_rskill`` — never trips the cap, while a
            genuine loop re-issuing the same call against unchanged
            context still does. The read-only search tools
            (:data:`_RETRY_CAP_EXEMPT_TOOLS`) are transparent to the cap:
            their own budgets (``SearchProgress``, ``TaskLocateBudget``)
            bound them and terminate in an explicit human-handoff, which
            the cap's silent hold must never preempt.
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
        # (tool kind, call identity JSON, consecutive count). Identity is
        # the call's full argument payload minus ``rationale`` — see
        # ``retry_cap_per_kind`` in the class docstring.
        self._call_streak: tuple[str, str, int] = ("", "", 0)
        # ``renderer.seq`` at the moment of the last ``retry_cap``
        # suppression, or ``None``. While the context has not moved past
        # that seq, a non-forced tick is held *before* the LLM call
        # (``retry_cap_hold``) — the model would see byte-identical context
        # and re-emit the same capped call, so the call is pure waste. The
        # node's reflection feedback (and any real event) bumps ``seq`` and
        # releases the hold.
        self._retry_cap_hold_seq: int | None = None
        self._tick_idx: int = 0
        # Tracks ``ContextRenderer.seq`` at the time of the last
        # successful (or non-idle-suppressed) tick. A heartbeat tick
        # (force=False) whose renderer hasn't budged since
        # ``_last_seen_seq`` is suppressed with ``heartbeat_idle`` —
        # the LLM call would see byte-identical context and produce
        # the same (or no) tool call.
        self._last_seen_seq: int = -1

    @property
    def retry_cap(self) -> int:
        """The configured consecutive-identical-call cap (read-only)."""
        return self._retry_cap

    @property
    def streak_tool(self) -> str:
        """Tool kind of the current consecutive-call streak (``""`` when none)."""
        return self._call_streak[0]

    def reset_kind_streak(self) -> None:
        """Reset the consecutive-call counter used by the retry-cap gate.

        Called by the reasoner_node whenever the situation changes
        materially (new operator prompt, mission advance, decompose, etc.).
        The retry-cap exists to prevent the model from looping on the same
        failure mode against a static context; once the context shifts
        (e.g. an operator types a new task), the previous streak
        carries no information and would otherwise silently swallow
        the next tool call. Also releases the pre-call ``retry_cap_hold``.
        """
        self._call_streak = ("", "", 0)
        self._retry_cap_hold_seq = None

    def tick(
        self,
        *,
        world_state: WorldState | None,
        renderer: ContextRenderer,
        palette: ToolPalette,
        force: bool = False,
        tier: str = "heartbeat",
    ) -> ReasonerTickResult:
        """Run one orchestrator pass, synchronously.

        Composition of the three phases (:meth:`prepare_tick` →
        :meth:`run_prepared_llm` → :meth:`finish_tick`) on the calling
        thread. The ROS node runs the LLM phase on a worker thread instead
        (issue #21 — a blocking ``select_tool`` starves the rclpy executor);
        this method remains the single-threaded contract for tests and
        non-ROS embedders.

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
        prep = self.prepare_tick(
            world_state=world_state,
            renderer=renderer,
            palette=palette,
            force=force,
            tier=tier,
        )
        if isinstance(prep, ReasonerTickResult):
            return prep
        try:
            call = self.run_prepared_llm(prep)
        except Exception as exc:
            # finish_tick returns a result for ROSPlanningError and re-raises
            # anything else after closing the span — same surface as before
            # the phase split.
            return self.finish_tick(prep, error=exc)
        return self.finish_tick(prep, call=call)

    def prepare_tick(
        self,
        *,
        world_state: WorldState | None,
        renderer: ContextRenderer,
        palette: ToolPalette,
        force: bool = False,
        tier: str = "heartbeat",
    ) -> PreparedTick | ReasonerTickResult:
        """Phase 1 of a tick: run the suppression gates and render the context.

        Cheap and non-blocking — safe (and required) on the rclpy executor
        thread, since it reads and snapshots state that executor callbacks
        mutate (the renderer, the palette reference). Returns a suppressed
        :class:`ReasonerTickResult` when a gate fires, else a
        :class:`PreparedTick` whose blocking LLM phase
        (:meth:`run_prepared_llm`) may run on any thread and whose
        :meth:`finish_tick` must run back on the owning thread.

        Args/semantics are those of :meth:`tick`.
        """
        started = self._clock()
        # min-interval gate — gate BEFORE opening the
        # OTel span so suppressed ticks don't show up in the trace.
        if not force and started - self._last_tick_s < self._min_interval_s:
            return ReasonerTickResult(
                tool_call=None,
                error=None,
                elapsed_s=0.0,
                suppressed_reason="min_interval",
            )
        # retry-cap hold — gate BEFORE the LLM call (and the OTel span).
        # After a ``retry_cap`` suppression the model would see the same
        # context and re-emit the same capped call, so paying for the LLM
        # round-trip just to discard the result is pure waste. Held until
        # the context moves past the seq the cap fired at (the node's
        # reflection feedback bumps it) or a reset/forced tick.
        if (
            not force
            and self._retry_cap_hold_seq is not None
            and renderer.seq == self._retry_cap_hold_seq
        ):
            self._last_tick_s = started
            return ReasonerTickResult(
                tool_call=None,
                error=None,
                elapsed_s=0.0,
                suppressed_reason="retry_cap_hold",
            )
        # A terminal mission has nothing left to plan. World-state and
        # perception updates may keep changing ``renderer.seq`` after handoff,
        # so the ordinary heartbeat-idle gate cannot suppress these calls.
        # A new operator goal rebuilds the mission before forcing a tick; safety
        # and other urgent events still bypass this gate via ``force=True``.
        mission = renderer.mission
        if (
            not force
            and mission is not None
            and mission.active() is None
            and renderer.inflight_skill is None
        ):
            self._last_tick_s = started
            self._last_seen_seq = renderer.seq
            return ReasonerTickResult(
                tool_call=None,
                error=None,
                elapsed_s=0.0,
                suppressed_reason="mission_finished",
            )
        # heartbeat-idle gate — gate
        # BEFORE the OTel span for the same reason. A non-forced tick
        # whose ContextRenderer has not received any new failure /
        # perception / prompt event since the last tick is suppressed:
        # the LLM would see byte-identical context and the call is
        # wasted. Forced ticks (event preemption) bypass this gate by
        # the ``force`` flag itself. Exception: while a skill is IN
        # FLIGHT the heartbeat stays live even on an unchanged seq —
        # the tick is the reasoner's only opportunity to poll
        # ``query_task_progress`` mid-execution (the system prompt
        # instructs exactly that), and "nothing new arrived" is not
        # the same as "nothing to supervise".
        if not force and renderer.seq == self._last_seen_seq and renderer.inflight_skill is None:
            self._last_tick_s = started
            return ReasonerTickResult(
                tool_call=None,
                error=None,
                elapsed_s=0.0,
                suppressed_reason="heartbeat_idle",
            )
        # Every tick that reaches the LLM (or
        # the palette-empty short-circuit) opens a ``reasoner.tick``
        # span. Non-attaching (start_reasoner_span) so the span can travel
        # prepare → worker LLM phase → finish without leaking into
        # executor callbacks that run between the phases; finish_tick
        # re-attaches it to capture the traceparent.
        self._tick_idx += 1
        span = start_reasoner_span(
            tick_idx=self._tick_idx,
            model=getattr(self._client, "model_id", None),
            force=force,
        )
        span.set_attribute(semconv.REASONER_TIER, tier)
        # Stamp the active mission queue on every tick span so
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
            span.end()
            return ReasonerTickResult(
                tool_call=None,
                error=None,
                elapsed_s=self._clock() - started,
                suppressed_reason="palette_empty",
            )
        context_text = renderer.render(world_state=world_state)
        # Snapshot seq + the rendered prompt records NOW: anything appended
        # while the LLM phase is in flight was not in the model's context,
        # so finish_tick must neither mark it seen nor drain it.
        return PreparedTick(
            context_text=context_text,
            palette=palette,
            system_prompt=self._system_prompt,
            renderer=renderer,
            span=span,
            started=started,
            seq=renderer.seq,
            prompts=renderer.prompts,
            force=force,
            tier=tier,
        )

    def run_prepared_llm(self, prep: PreparedTick) -> ReasonerToolCall:
        """Phase 2 of a tick: the blocking LLM round-trip.

        The **only** phase safe to run off the owning thread — it touches
        nothing but the client and the immutable snapshots inside ``prep``.
        Runs under the tick span (per-thread attach) so client-internal
        spans nest correctly; exception recording is deferred to
        :meth:`finish_tick` (single recorder).

        Raises:
            ROSPlanningError: Provider/transport/decode failures, exactly
                as :meth:`ToolUseClient.select_tool` raises them.
        """
        with use_span(
            prep.span, end_on_exit=False, record_exception=False, set_status_on_exception=False
        ):
            llm_started = self._clock()
            try:
                return self._client.select_tool(
                    context_text=prep.context_text,
                    palette=prep.palette,
                    system_prompt=prep.system_prompt,
                )
            finally:
                # ``finally``: a timed-out / failed call is exactly the one
                # whose duration matters most.
                prep.llm_s = self._clock() - llm_started
                # Optional client attribute — absent on clients that don't
                # surface provider usage, hence getattr rather than a
                # Protocol member every implementation would have to carry.
                prep.prompt_tokens = getattr(self._client, "last_prompt_tokens", None)

    def finish_tick(
        self,
        prep: PreparedTick,
        *,
        call: ReasonerToolCall | None = None,
        error: BaseException | None = None,
    ) -> ReasonerTickResult:
        """Phase 3 of a tick: bookkeeping + span close, back on the owning thread.

        Applies the retry-cap ladder, marks the context seen **up to the
        prepare-time snapshot** (``prep.seq`` / ``prep.prompts`` — events
        that arrived mid-flight stay unseen for the next tick), captures
        the traceparent, and ends the tick span.

        Args:
            prep: The matching :meth:`prepare_tick` output.
            call: The selected tool call (success path).
            error: The exception the LLM phase raised. A
                :class:`ROSPlanningError` becomes ``ReasonerTickResult.error``;
                anything else is recorded on the span and **re-raised** —
                the pre-split behavior of an unexpected client bug.
        """
        with use_span(
            prep.span, end_on_exit=True, record_exception=False, set_status_on_exception=False
        ):
            span = prep.span
            # Stamped before the error/suppression branches: an errored or
            # capped tick still burned the LLM round-trip, and those are the
            # ticks worth attributing.
            span.set_attribute(semconv.REASONER_LLM_S, round(prep.llm_s, 4))
            if prep.prompt_tokens is not None:
                span.set_attribute(semconv.REASONER_PROMPT_TOKENS, prep.prompt_tokens)
            if error is not None:
                self._last_tick_s = prep.started
                self._last_seen_seq = prep.seq
                span.record_exception(error)
                if not isinstance(error, ROSPlanningError):
                    raise error
                span.set_attribute(semconv.REASONER_ERROR_KIND, type(error).__name__)
                return ReasonerTickResult(
                    tool_call=None,
                    error=error,
                    elapsed_s=self._clock() - prep.started,
                )
            if call is None:
                raise ValueError("finish_tick: pass exactly one of `call` or `error`")
            # retry-cap ("bounded retry counter per failure kind") — keyed
            # on the full call identity (tool + args minus rationale), so
            # only a *verbatim* repeat accumulates; a different skill /
            # different params is progress, not a retry. Read-only search
            # tools are transparent to the cap (see _RETRY_CAP_EXEMPT_TOOLS):
            # their own budgets bound them and must reach the explicit
            # human-handoff, never a silent hold.
            if call.tool not in _RETRY_CAP_EXEMPT_TOOLS:
                identity = _call_identity(call)
                _prev_tool, prev_identity, streak = self._call_streak
                streak = streak + 1 if identity == prev_identity else 1
                self._call_streak = (call.tool, identity, streak)
            else:
                streak = 0  # transparent: streak untouched, never capped
            if streak > self._retry_cap:
                # Arm the pre-call hold at the seq this cap fired against so
                # subsequent unchanged-context ticks skip the LLM entirely.
                self._retry_cap_hold_seq = prep.seq
                self._last_tick_s = prep.started
                self._last_seen_seq = prep.seq
                span.set_attribute(semconv.REASONER_TOOL, call.tool)
                span.set_attribute(semconv.REASONER_SUPPRESSED_REASON, "retry_cap")
                return ReasonerTickResult(
                    tool_call=None,
                    error=None,
                    elapsed_s=self._clock() - prep.started,
                    suppressed_reason="retry_cap",
                )
            self._retry_cap_hold_seq = None
            self._last_tick_s = prep.started
            self._last_seen_seq = prep.seq
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
                "tier": prep.tier,
                "elapsed_s": round(self._clock() - prep.started, 4),
                # elapsed_s minus llm_s is the reasoner-side overhead; without
                # the split a growing tick can't be blamed on the provider or
                # on us (issue #92).
                "llm_s": round(prep.llm_s, 4),
            }
            if prep.prompt_tokens is not None:
                _log_kwargs["prompt_tokens"] = prep.prompt_tokens
            if rskill_id:
                _log_kwargs["rskill_id"] = rskill_id
            rationale = getattr(call, "rationale", None)
            if rationale:
                _log_kwargs["rationale"] = rationale
            # Surface the active prompt so the caller can correlate which
            # operator goal drove this tool selection (multi-task tracing).
            if prep.prompts:
                _log_kwargs["active_prompt"] = prep.prompts[0].text[:200]
            log.info("reasoner.tick.selected", **_log_kwargs)
            # Capture the active traceparent WHILE the span is still in
            # scope so reasoner_node._dispatch (which runs after this
            # function returns) can stamp it onto outbound PromptStamped
            # metadata_json ("OTel context is the truth").
            traceparent = current_traceparent()
            # Drain the operator prompts the model actually saw (pull-once
            # semantics); mid-flight arrivals survive for the next tick.
            prep.renderer.drain_prompts(seen=prep.prompts)
            return ReasonerTickResult(
                tool_call=call,
                error=None,
                elapsed_s=self._clock() - prep.started,
                traceparent=traceparent,
            )


# Re-export :class:`ROSReasonerInvalidPlan` so test code can ``from
# openral_reasoner.core import ROSReasonerInvalidPlan`` without
# reaching into ``openral_core.exceptions``.
__all__ += ["ROSReasonerInvalidPlan"]
