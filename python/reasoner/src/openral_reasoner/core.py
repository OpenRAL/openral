"""``ReasonerCore``.

The transport-agnostic orchestrator that closes
context → LLM → typed tool call. The ROS-side
``openral_reasoner_ros.reasoner_node`` wraps this class with the
lifecycle, subscriptions, and dispatch plumbing; the core itself has
no rclpy dependency so it is fully unit-testable against a
``FakeToolUseClient``.

The reasoner:

* Holds **no** authority over actuation (never publishes
  ``ActionChunk``).
* Picks exactly one typed ``ReasonerToolCall`` per tick.
* Enforces a bounded retry counter per identical call to prevent storms.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable

import structlog
from openral_core import EmitPromptTool, ReasonerToolCall, WaitTool, WorldState
from openral_core.exceptions import (
    ROSPlanningError,
    ROSReasonerInvalidPlan,
)
from openral_observability import semconv, start_reasoner_span
from openral_observability.compliance import free_text_may_leave_host
from openral_observability.propagation import current_traceparent
from opentelemetry.trace import Span, use_span

from openral_reasoner.context import ContextRenderer, PromptRecord
from openral_reasoner.palette import ToolPalette
from openral_reasoner.skill_subset import SkillSubset, select_skill_subset
from openral_reasoner.tool_use import DEFAULT_SYSTEM_PROMPT, ToolUseClient

__all__ = [
    "BACKSTOP_MAX_S",
    "BACKSTOP_MIN_S",
    "LLM_ERROR_BACKOFF_S",
    "QUIET_SUPPRESSIONS",
    "PreparedTick",
    "ReasonerCore",
    "ReasonerTickResult",
    "inflight_backstop_s",
]

log = structlog.get_logger(__name__)


def _stamp_mission(span: Span, renderer: ContextRenderer) -> None:
    """Stamp the active mission queue on the tick span.

    Serialized as ``reasoner.mission_json`` so the live dashboard renders the
    task checklist. No-op when no mission is set (a bare operator goal).
    """
    mission = renderer.mission
    if mission is not None and not mission.is_empty():
        span.set_attribute(semconv.REASONER_MISSION_JSON, json.dumps(mission.to_summary()))


def _stamp_rationale(span: Span, call: ReasonerToolCall) -> None:
    """Stamp the call's rationale (capped) so the dashboard trace can say why it chose it.

    An ``EmitPromptTool`` also stamps its prompt text (capped), so the console folds
    only identical prompts. Free model text that can paraphrase the operator's prompt:
    stamped only where it may leave the host (a local collector, or the deployer's
    consent; CLAUDE.md §3).
    """
    if not free_text_may_leave_host():
        return
    if call.rationale:
        span.set_attribute(
            semconv.REASONER_RATIONALE, call.rationale[: semconv.REASONER_RATIONALE_MAX]
        )
    if isinstance(call, EmitPromptTool):
        span.set_attribute(semconv.REASONER_MESSAGE, call.text[: semconv.REASONER_MESSAGE_MAX])


# Read-only search tools (recall_object/resolve_place/locate_in_view) are
# TRANSPARENT to the retry cap — each has its own bound (SearchProgress's
# miss budget, the per-task TaskLocateBudget) terminating in an explicit
# human-handoff; the identity cap firing first would silently
# retry_cap_hold before that handoff (see
# test_active_search_cascade_is_bounded_and_hands_off). Transparency (not
# reset) also lets an alternating <same-call>/<search> loop still accumulate.
#
# "wait" is exempt separately: the system prompt instructs the model to keep
# picking wait during a long skill execution, and identity is
# rationale-stripped so every wait is byte-identical — counting it would
# fabricate a "retry ladder exhausted" failure mid-run. Waiting is already
# bounded by the skill's own deadline/patience machinery.
_RETRY_CAP_EXEMPT_TOOLS: frozenset[str] = frozenset(
    {"recall_object", "resolve_place", "locate_in_view", "wait"}
)


#: Clamp for the in-flight backstop (issue #620): while a skill runs and nothing
#: forces a tick, the heartbeat calls the LLM at most once per
#: ``clamp(max_execution_s / 4, BACKSTOP_MIN_S, BACKSTOP_MAX_S)`` seconds.
BACKSTOP_MIN_S: float = 15.0
BACKSTOP_MAX_S: float = 30.0

#: Delays before re-trying an LLM call that failed in transport (timeout, 5xx,
#: refusal to answer). One entry per retry; after the last the context is marked
#: seen and ``ReasonerCore.llm_gave_up`` turns true so the node hands off.
LLM_ERROR_BACKOFF_S: tuple[float, ...] = (5.0, 10.0, 20.0)

#: Suppression reasons that are a timer beat's steady state, not news: a node logs
#: them at DEBUG and keeps its one-shot retry-cap warning latched across them (no
#: LLM call ran). Every gate in ``ReasonerCore._timer_gate`` returns one of these.
QUIET_SUPPRESSIONS: frozenset[str] = frozenset(
    {
        "min_interval",
        "safety_latched",
        "llm_error_backoff",
        "retry_cap_hold",
        "mission_finished",
        "wait_hold",
        "backstop_not_due",
        "heartbeat_idle",
        "no_goal",
        "llm_gave_up",
    }
)


def inflight_backstop_s(max_execution_s: float | None) -> float:
    """Heartbeat period while a skill is in flight and no event forces a tick.

    A quarter of the skill's execution budget, clamped to
    ``[BACKSTOP_MIN_S, BACKSTOP_MAX_S]``; ``BACKSTOP_MIN_S`` when the manifest
    declares no budget.

    Example:
        >>> inflight_backstop_s(None), inflight_backstop_s(80.0), inflight_backstop_s(600.0)
        (15.0, 20.0, 30.0)
    """
    if max_execution_s is None:
        return BACKSTOP_MIN_S
    return min(max(max_execution_s / 4.0, BACKSTOP_MIN_S), BACKSTOP_MAX_S)


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

    Carries everything the blocking LLM phase needs (context text, the palette
    offered this tick — narrowed by ``ReasonerCore.offered_palette`` when a
    skill-tool cap is set — and the system prompt) plus the bookkeeping captured at prepare time
    (``seq``, ``prompts``, ``started``, the open OTel span). The ``seq`` and
    ``prompts`` snapshots matter: events that arrive while the LLM call is in
    flight were **not** rendered into the model's context, so
    ``ReasonerCore.finish_tick`` must mark seen / drain only what the
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
    trigger: str = "heartbeat"
    """What started this tick: the event name a forced tick was raised for
    (``prompt``, ``skill_result``, ``failure:<source>``, …) or, for a timer
    beat, the gate that let it through (``context_changed``, ``backstop``,
    ``wait_timeout``, ``llm_error_retry``)."""
    coalesced: tuple[str, ...] = ()
    """Triggers merged into this tick while the previous one was in flight."""
    llm_s: float = 0.0
    """Wall-clock of the LLM round-trip alone, written by
    ``ReasonerCore.run_prepared_llm`` (the only field the LLM phase
    mutates) and read back by ``ReasonerCore.finish_tick``. Split out
    from the tick's ``elapsed_s`` because a slow tick is otherwise
    unattributable: provider time and reasoner overhead look identical."""
    prompt_tokens: int | None = None
    """Provider-reported prompt-token count for that call, or ``None`` when
    the client does not surface usage."""
    served_model: str | None = None
    """The model the endpoint reported serving the call (an alias resolves
    server-side), or ``None`` when the client does not surface it."""


@dataclasses.dataclass(frozen=True, slots=True)
class ReasonerTickResult:
    """Outcome of a single ``ReasonerCore.tick`` invocation.

    Attributes:
        tool_call: The validated tool call selected by the LLM, or
            ``None`` when the tick was suppressed (rate-limited,
            palette empty, etc.).
        error: A ``ROSPlanningError`` subclass when the tick
            failed, or ``None`` on success.
        elapsed_s: Wall-clock time the tick took, end-to-end.
        suppressed_reason: When ``tool_call is None and error is None``,
            a short string explaining why the tick was suppressed
            (e.g. ``"min_interval"``, ``"retry_cap"``,
            ``"retry_cap_hold"``, ``"palette_empty"``,
            ``"heartbeat_idle"``, ``"backstop_not_due"``, ``"wait_hold"``,
            ``"llm_error_backoff"``, ``"safety_latched"``, ``"no_goal"``).
        trigger: ``PreparedTick.trigger`` of a tick that reached the LLM
            (``""`` when suppressed before it).
    """

    tool_call: ReasonerToolCall | None
    error: ROSPlanningError | None
    elapsed_s: float
    suppressed_reason: str = ""
    trigger: str = ""
    traceparent: str | None = None
    """W3C ``traceparent`` captured while the ``reasoner.tick`` span was
    active. The reasoner_node stamps this onto the outbound
    ``EmitPromptTool`` ``PromptStamped.metadata_json`` so the F7
    bag↔OTel correlator can join the published prompt back to the
    reasoner span that produced it. ``None`` when no
    real ``TracerProvider`` is installed."""


class ReasonerCore:
    """Transport-agnostic reasoner orchestrator.

    The class is intentionally narrow: it consumes the current
    ``ContextRenderer`` + ``ToolPalette`` and produces a
    typed tool call. Wiring (subscriptions, action clients, service
    calls) lives in the ROS lifecycle node.

    Args:
        client: A ``ToolUseClient`` instance. In tests this is
            usually ``FakeToolUseClient`` (under
            ``tests/integration/fakes/``); in production it is one of
            the SDK-backed clients from ``openral_reasoner.tool_use``.
        min_interval_s: Hard lower bound between consecutive ticks, in
            seconds. Mandated as 100 ms (0.1 s).
        retry_cap_per_kind: Maximum number of consecutive ticks the
            reasoner may select the **identical tool call** (same tool +
            same arguments, ``rationale`` excluded) before being
            suppressed by ``retry_cap``. Defaults to 3 — matches the
            replanning ladder guidance in CLAUDE.md §3 Reasoner & dispatch. Keying on the
            full call identity (rather than the bare tool kind) means a
            healthy sequence of *different* skills — navigate → pick →
            place, all ``execute_rskill`` — never trips the cap, while a
            genuine loop re-issuing the same call against unchanged
            context still does. The read-only search tools
            (``_RETRY_CAP_EXEMPT_TOOLS``) are transparent to the cap:
            their own budgets (``SearchProgress``, ``TaskLocateBudget``)
            bound them and terminate in an explicit human-handoff, which
            the cap's silent hold must never preempt.
        system_prompt: Override the
            ``DEFAULT_SYSTEM_PROMPT``.
            ``None`` keeps the default.
        max_skill_tools: Offer at most this many skill tools per tick, ranked
            against the mission (``select_skill_subset``); ``None`` offers the
            whole palette. Set from ``resolve_max_skill_tools`` for models that
            break on many near-identical skill tools (Cosmos3-Edge, #494). The
            subset is recomputed only when the palette, mission text or sticky
            skills change, so the tool prefix stays cacheable.
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
        max_skill_tools: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Stash configuration; no LLM call until ``tick``."""
        if min_interval_s < 0:
            raise ValueError(
                f"ReasonerCore.min_interval_s must be >= 0; got {min_interval_s!r}",
            )
        if retry_cap_per_kind < 1:
            raise ValueError(
                f"ReasonerCore.retry_cap_per_kind must be >= 1; got {retry_cap_per_kind!r}",
            )
        if max_skill_tools is not None and max_skill_tools < 1:
            raise ValueError(
                f"ReasonerCore.max_skill_tools must be >= 1 or None; got {max_skill_tools!r}",
            )
        self._client = client
        self._max_skill_tools = max_skill_tools
        # (full palette, sticky ids asked for, subset) for the last tick. Reused while
        # the palette object, query (``subset.query``) and sticky ids hold; its
        # narrowed palette object is reused while the offered set holds, so the
        # clients' per-palette-object tool render (and the provider's prefix cache)
        # survive across ticks.
        self._subset: tuple[ToolPalette, tuple[str, ...], SkillSubset] | None = None
        # rskill_id of the last ExecuteRskillTool this core returned; the
        # ``recent`` slot when no mission task is active.
        self._last_dispatched: str | None = None
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
        # Start of the last tick that reached the LLM (any trigger); the
        # in-flight backstop is measured from it.
        self._last_llm_s: float = -float("inf")
        # Timer beats suppressed since the last LLM call (span attribute).
        self._skipped_since_llm: int = 0
        # (until, monotonic deadline) of the last ``wait`` the model chose.
        self._wait: tuple[str, float] | None = None
        # Consecutive transport failures of the LLM call, and when the next
        # timer beat may retry it.
        self._llm_errors: int = 0
        self._llm_retry_at: float | None = None

    @property
    def llm_gave_up(self) -> bool:
        """True once every ``LLM_ERROR_BACKOFF_S`` retry failed in a row.

        Cleared by the next successful LLM call. The node hands the mission to
        the operator when it flips.
        """
        return self._llm_errors > len(LLM_ERROR_BACKOFF_S)

    @property
    def wait_hold(self) -> tuple[str, float] | None:
        """``(until, monotonic deadline)`` of the active ``wait``, or ``None``."""
        return self._wait

    @property
    def max_skill_tools(self) -> int | None:
        """The configured per-tick skill-tool cap, or ``None`` (read-only)."""
        return self._max_skill_tools

    def offered_palette(self, palette: ToolPalette, renderer: ContextRenderer) -> ToolPalette:
        """The palette the LLM is shown this tick: ``palette`` narrowed to the skill cap.

        Ranked against the active mission task plus every pending operator prompt
        (an operator reply mid-mission reaches the LLM through PROMPTS without
        rebuilding the mission, so it must be able to change the offer too). The
        skill in flight stays offered so the model can wait on it; next, the best
        match for the highest-priority pending prompt keeps a slot so the
        operator's ask is not outvoted by a long task text (with ``k`` >= 2 it is
        always offered on the tick the prompt arrives, before that tick drains
        it); then the active task's last dispatched skill, so the model can
        re-dispatch it. Equal scores prefer skills not yet tried for the active task
        (so substituting a skill reaches beyond the top-k), then higher weight
        precision. Without a cap the palette is returned unchanged.

        The ranking (query, every score, offered and kept ids) is logged as
        ``reasoner.skill_subset`` each time it is recomputed, i.e. whenever the
        palette, query or sticky ids change.
        """
        if self._max_skill_tools is None:
            return palette
        mission = renderer.mission
        active = mission.active() if mission is not None else None
        texts = ([active.text] if active is not None else []) + [p.text for p in renderer.prompts]
        query = "\n".join(texts)
        cached = self._subset
        if not query.strip() and cached is not None:
            # No task and no pending prompt (mission finished, or the prompt was
            # drained last tick) but a failure / perception event woke the tick:
            # keep ranking on the last real query instead of an empty one, which
            # would offer the alphabetical first k and refuse the retry.
            query = cached[2].query
        inflight = renderer.inflight_skill
        # Outside a mission the last execute_rskill this core dispatched stays
        # offered, so a retry after its failure is not refused off-palette.
        recent = active.last_rskill_id if active is not None else self._last_dispatched
        tried = active.tried_rskill_ids if active is not None else ()
        sticky = (inflight or "", recent or "", *tried)
        # The buffer is priority-ordered (highest first), arrival order within a
        # priority: the newest prompt of the top band is the operator's latest ask
        # ("no, the cup instead" outranks the "pick the bowl" it corrects).
        prompts = renderer.prompts
        focus = (
            next(p for p in reversed(prompts) if p.priority == prompts[0].priority).text
            if prompts
            else ""
        )
        if (
            cached is not None
            and cached[0] is palette
            and (cached[2].query, cached[1]) == (query, sticky)
        ):
            return cached[2].palette
        subset = select_skill_subset(
            palette,
            query=query,
            k=self._max_skill_tools,
            sticky=(inflight,),
            focus=focus,
            recent=(recent,),
            tried=tried,
        )
        if cached is not None and cached[2].palette == subset.palette:
            subset = subset.model_copy(update={"palette": cached[2].palette})
        log.info(
            "reasoner.skill_subset",
            k=subset.k,
            offered=list(subset.offered),
            total=len(subset.scores),
            query=query[:400],
            sticky=list(subset.sticky),
            focused=list(subset.focused),
            support=list(subset.support),
            tried=list(subset.tried),
            max_score=subset.max_score,
            scores=dict(subset.scores),
        )
        self._subset = (palette, sticky, subset)
        return subset.palette

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

        Called whenever the situation changes materially (new operator
        prompt, mission advance, decompose, etc.) so a stale streak cannot
        silently swallow the next tool call. Also releases the pre-call
        ``retry_cap_hold``.
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
        trigger: str = "heartbeat",
    ) -> ReasonerTickResult:
        """Run one orchestrator pass, synchronously.

        Composition of the three phases (``prepare_tick`` →
        ``run_prepared_llm`` → ``finish_tick``) on the calling
        thread. The ROS node runs the LLM phase on a worker thread instead
        (issue #21 — a blocking ``select_tool`` starves the rclpy executor);
        this method remains the single-threaded contract for tests and
        non-ROS embedders.

        Args:
            world_state: Latest WorldState snapshot or ``None``.
            renderer: The reasoner's ``ContextRenderer``. Prompts
                are drained on a successful tick.
            palette: Current ``ToolPalette``.
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
                ``ReasonerNode``.
            trigger: Event name of a forced tick (see ``PreparedTick.trigger``).

        Returns:
            A ``ReasonerTickResult``.
        """
        prep = self.prepare_tick(
            world_state=world_state,
            renderer=renderer,
            palette=palette,
            force=force,
            tier=tier,
            trigger=trigger,
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

    def _suppressed(self, reason: str) -> ReasonerTickResult:
        """A gate fired before the LLM call: count the skipped beat and report it."""
        self._skipped_since_llm += 1
        return ReasonerTickResult(
            tool_call=None, error=None, elapsed_s=0.0, suppressed_reason=reason
        )

    def _timer_gate(
        self, renderer: ContextRenderer, palette: ToolPalette, started: float
    ) -> ReasonerTickResult | str:
        """Gates of a timer beat: a suppressed result, or the trigger that lets it through."""
        if started - self._last_tick_s < self._min_interval_s:
            return self._suppressed("min_interval")
        self._last_tick_s = started
        held = self._held_reason(renderer, started)
        if held is not None:
            return self._suppressed(held)
        if self._llm_retry_at is not None:  # not held: the retry is due
            self._llm_retry_at = None
            return "llm_error_retry"
        waited = self._wait_gate(renderer, started)
        if waited == "wait_hold":
            return self._suppressed(waited)
        if waited is not None:
            return waited
        return self._idle_gate(renderer, palette, started)

    def _held_reason(self, renderer: ContextRenderer, started: float) -> str | None:
        """Why a timer beat must not call the LLM regardless of the context, or ``None``."""
        if self._llm_retry_at is not None:
            # A scheduled transport retry runs even while latched: the failed call
            # may have been the latch notification itself.
            return "llm_error_backoff" if started < self._llm_retry_at else None
        if self.llm_gave_up:
            # Every retry failed: no timer beat re-sends (not even a skill's
            # backstop); only a new event reaches the model.
            return "llm_gave_up"
        if renderer.safety_latched and renderer.seq == self._last_seen_seq:
            # A latched stop refuses every motion until the operator resets it:
            # nothing new to show the model means nothing to decide. A context
            # that moved (the latch itself arriving after the kernel's failure
            # tick, an operator note) still gets through once.
            return "safety_latched"
        if self._retry_cap_hold_seq is not None and renderer.seq == self._retry_cap_hold_seq:
            # After a ``retry_cap`` the model would see the same context and
            # re-emit the same capped call. Held until the context moves past the
            # seq the cap fired at (the node's reflection bumps it) or a reset.
            return "retry_cap_hold"
        mission = renderer.mission
        if mission is not None and mission.active() is None and renderer.inflight_skill is None:
            # A terminal mission has nothing left to plan; perception keeps moving
            # ``seq`` after hand-off, so the idle gate alone cannot hold it. A new
            # operator goal rebuilds the mission before forcing a tick.
            self._last_seen_seq = renderer.seq
            self._wait = None
            return "mission_finished"
        return None

    def _wait_gate(self, renderer: ContextRenderer, started: float) -> str | None:
        """The model's own ``wait``: ``"wait_hold"``, ``"wait_timeout"`` (due), or ``None``."""
        if self._wait is None:
            return None
        until, deadline = self._wait
        inflight = renderer.inflight_skill
        if started >= deadline:
            self._wait = None
            # A lapsed wait re-decides when there is something to decide about,
            # or when the model asked for exactly that (``until="timeout"``: a
            # bare prompt with no mission keeps its promised reconsideration).
            # A default wait, idle with no task, falls through to the idle gate.
            mission = renderer.mission
            busy = inflight is not None or (mission is not None and mission.active() is not None)
            return "wait_timeout" if busy or until == "timeout" else None
        changed = renderer.seq != self._last_seen_seq
        if (until == "next_event" and changed) or (until == "skill_result" and inflight is None):
            self._wait = None
            return None
        return "wait_hold"

    def _idle_gate(
        self, renderer: ContextRenderer, palette: ToolPalette, started: float
    ) -> ReasonerTickResult | str:
        """Whether the context gives a timer beat anything new to decide about."""
        changed = renderer.seq != self._last_seen_seq
        inflight = renderer.inflight_skill
        if inflight is not None:
            # While a skill runs the beat is a backstop, not a poll: one LLM call
            # per ``inflight_backstop_s``. A result-only skill (wrapped Nav2-shape
            # action) has no progress to poll, so an unchanged context keeps it
            # quiet unless a reward monitor can still be asked (2026-11-11: 103 of
            # 109 LLM calls were `wait` ticks during one drive).
            entry = palette.skill(inflight)
            result_only = entry is not None and entry.result_only
            if not (changed or palette.task_progress_available or not result_only):
                return self._suppressed("heartbeat_idle")
            period = inflight_backstop_s(entry.max_execution_s if entry else None)
            if started - self._last_llm_s < period:
                return self._suppressed("backstop_not_due")
            return "context_changed" if changed else "backstop"
        if not changed:
            # Nothing new since the last tick: byte-identical context.
            return self._suppressed("heartbeat_idle")
        if renderer.mission is None and not renderer.prompts:
            # No goal, nothing in flight, no prompt: perception churn alone gives
            # the model nothing to decide (#620). Events still force their ticks.
            self._last_seen_seq = renderer.seq
            return self._suppressed("no_goal")
        return "context_changed"

    def prepare_tick(
        self,
        *,
        world_state: WorldState | None,
        renderer: ContextRenderer,
        palette: ToolPalette,
        force: bool = False,
        tier: str = "heartbeat",
        trigger: str = "heartbeat",
        coalesced: tuple[str, ...] = (),
    ) -> PreparedTick | ReasonerTickResult:
        """Phase 1 of a tick: run the suppression gates and render the context.

        Cheap and non-blocking — safe (and required) on the rclpy executor
        thread, since it reads and snapshots state that executor callbacks
        mutate (the renderer, the palette reference). Returns a suppressed
        ``ReasonerTickResult`` when a gate fires, else a
        ``PreparedTick`` whose blocking LLM phase
        (``run_prepared_llm``) may run on any thread and whose
        ``finish_tick`` must run back on the owning thread.

        A forced tick (an event) skips every gate below and ends an active
        ``wait``. A timer beat (``force=False``) reaches the LLM only when one
        of these is true (issue #620), recorded as its ``trigger``:

        * ``llm_error_retry`` — the backoff after a failed LLM call lapsed;
        * ``wait_timeout`` — the model's ``wait`` lapsed with a skill in
          flight or a mission task active;
        * ``backstop`` — a skill is in flight and ``inflight_backstop_s`` has
          passed since the last LLM call (perception churn alone never calls
          more often than that);
        * ``context_changed`` — nothing is in flight, the context moved, and
          there is a mission or a pending prompt (with neither it is ``no_goal``).

        Args/semantics are those of ``tick``; ``coalesced`` lists the triggers
        merged into this tick while the previous one was in flight.
        """
        started = self._clock()
        # Fold time-dependent context (a safety status going stale) into ``seq``
        # once per tick, before any gate reads it and before the snapshot below,
        # so what the gates track and what is rendered come from one evaluation.
        renderer.refresh_clock_state()
        if force:
            # An event the model has not seen yet ends any wait it chose.
            self._wait = None
        else:
            # Gate BEFORE opening the OTel span so suppressed beats don't show
            # up in the trace.
            gate = self._timer_gate(renderer, palette, started)
            if isinstance(gate, ReasonerTickResult):
                return gate
            trigger = gate
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
        inflight = renderer.inflight_skill
        if inflight:
            span.set_attribute(semconv.REASONER_INFLIGHT_SKILL, inflight)
            span.set_attribute(semconv.REASONER_INFLIGHT_STATE, renderer.inflight_state or "")
        span.set_attribute(semconv.REASONER_TRIGGER, trigger)
        if coalesced:
            span.set_attribute(semconv.REASONER_COALESCED_TRIGGERS, list(coalesced))
        span.set_attribute(semconv.REASONER_SKIPPED_SINCE_LAST, self._skipped_since_llm)
        # Stamp the active mission queue on every tick span so
        # the dashboard can render the task checklist. Set before the gate
        # short-circuits below so suppressed (retry_cap / error) ticks still
        # carry current mission state. The mission is unchanged within a tick.
        _stamp_mission(span, renderer)
        # palette-empty short-circuit — the LLM call would just timeout /
        # pick a phantom rskill_id; surface the configuration error
        # explicitly. Bypassed when ``force=True``: an event-preempted tick
        # (SEVERITY_FAIL FailureTrigger or a new operator prompt) demands
        # the LLM's attention even with no skills installed — at minimum it
        # can pick ``EmitPromptTool`` to escalate to the operator;
        # gating here would silently swallow SEVERITY_FAIL preemptions on a
        # bare reasoner.
        # A due transport retry or a lapsed ``wait`` is a decision the model or an
        # event already demanded, and its deadline was cleared when it became
        # due: dropping it here would lose it for good.
        if (
            not force
            and trigger not in ("llm_error_retry", "wait_timeout")
            and not palette.execute_rskill_ids
            and not (palette.sensor_ids or palette.node_ids or renderer.prompts)
        ):
            self._last_tick_s = started
            self._last_seen_seq = renderer.seq
            span.set_attribute(semconv.REASONER_SUPPRESSED_REASON, "palette_empty")
            span.end()
            self._skipped_since_llm += 1
            return ReasonerTickResult(
                tool_call=None,
                error=None,
                elapsed_s=self._clock() - started,
                suppressed_reason="palette_empty",
            )
        offered = self.offered_palette(palette, renderer)
        if self._max_skill_tools is not None:
            span.set_attribute(semconv.REASONER_OFFERED_SKILLS, sorted(offered.execute_rskill_ids))
        # ``seq`` and the render both reflect the ``refresh_clock_state`` above;
        # neither reading changes the renderer.
        seq = renderer.seq
        context_text = renderer.render(world_state=world_state)
        # Snapshot the rendered prompt records NOW too: anything appended
        # while the LLM phase is in flight was not in the model's context,
        # so finish_tick must neither mark it seen nor drain it.
        return PreparedTick(
            context_text=context_text,
            palette=offered,
            system_prompt=self._system_prompt,
            renderer=renderer,
            span=span,
            started=started,
            seq=seq,
            prompts=renderer.prompts,
            force=force,
            tier=tier,
            trigger=trigger,
            coalesced=coalesced,
        )

    def run_prepared_llm(self, prep: PreparedTick) -> ReasonerToolCall:
        """Phase 2 of a tick: the blocking LLM round-trip.

        The **only** phase safe to run off the owning thread — it touches
        nothing but the client and the immutable snapshots inside ``prep``.
        Runs under the tick span (per-thread attach) so client-internal
        spans nest correctly; exception recording is deferred to
        ``finish_tick`` (single recorder).

        Raises:
            ROSPlanningError: Provider/transport/decode failures, exactly
                as ``ToolUseClient.select_tool`` raises them.
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
                prep.served_model = getattr(self._client, "served_model", None)

    def _record_llm_error(self, error: ROSPlanningError, seq: int) -> None:
        """Bookkeeping for an LLM call that returned no usable tool call."""
        if isinstance(error, ROSReasonerInvalidPlan):
            # The model answered, just not with a valid call; the node feeds the
            # reason back as context (which moves seq).
            self._last_seen_seq = seq
            self._llm_errors = 0
            self._llm_retry_at = None
            return
        # Transport failure: the model never saw this context, so it stays unseen
        # and a timer beat retries it after a backoff. Out of retries -> mark it
        # seen (``llm_gave_up``); the next event still reaches the model.
        if self._llm_retry_at is not None:
            # An event tick failed while a retry is already scheduled: the
            # backoff stands. Only scheduled retries spend the budget, so a burst
            # of events during a short outage cannot exhaust it in seconds.
            return
        self._llm_errors += 1
        if self._llm_errors <= len(LLM_ERROR_BACKOFF_S):
            self._llm_retry_at = self._clock() + LLM_ERROR_BACKOFF_S[self._llm_errors - 1]
        else:
            self._llm_retry_at = None
            self._last_seen_seq = seq

    def _arm_after_call(self, call: ReasonerToolCall, span: Span) -> None:
        """Book a good call: rationale on the span, LLM-error run over, wait hold set.

        Runs before the retry-cap branch, so a capped tick says why too. A ``wait``
        arms its hold; any other call ends one.
        """
        _stamp_rationale(span, call)
        self._llm_errors = 0
        self._llm_retry_at = None
        if isinstance(call, WaitTool):
            self._wait = (call.until, self._clock() + call.timeout_s)
            span.set_attribute(semconv.REASONER_WAIT_UNTIL, call.until)
            span.set_attribute(semconv.REASONER_WAIT_TIMEOUT_S, call.timeout_s)
        else:
            self._wait = None

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
            prep: The matching ``prepare_tick`` output.
            call: The selected tool call (success path).
            error: The exception the LLM phase raised. A
                ``ROSPlanningError`` becomes ``ReasonerTickResult.error``;
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
            usage: dict[str, int | str | None] = {
                semconv.REASONER_PROMPT_TOKENS: prep.prompt_tokens,
                semconv.REASONER_SERVED_MODEL: prep.served_model,
            }
            span.set_attributes({k: v for k, v in usage.items() if v is not None})
            # Every outcome below spent an LLM call at ``prep.started``.
            self._last_llm_s = self._last_tick_s = prep.started
            self._skipped_since_llm = 0
            if error is not None:
                span.record_exception(error)
                if not isinstance(error, ROSPlanningError):
                    self._last_seen_seq = prep.seq
                    raise error
                span.set_attribute(semconv.REASONER_ERROR_KIND, type(error).__name__)
                self._record_llm_error(error, prep.seq)
                return ReasonerTickResult(
                    tool_call=None,
                    error=error,
                    elapsed_s=self._clock() - prep.started,
                    trigger=prep.trigger,
                )
            if call is None:
                raise ValueError("finish_tick: pass exactly one of `call` or `error`")
            self._arm_after_call(call, span)
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
                self._last_seen_seq = prep.seq
                span.set_attribute(semconv.REASONER_TOOL, call.tool)
                span.set_attribute(semconv.REASONER_SUPPRESSED_REASON, "retry_cap")
                return ReasonerTickResult(
                    tool_call=None,
                    error=None,
                    elapsed_s=self._clock() - prep.started,
                    suppressed_reason="retry_cap",
                    trigger=prep.trigger,
                )
            self._retry_cap_hold_seq = None
            self._last_seen_seq = prep.seq
            span.set_attribute(semconv.REASONER_TOOL, call.tool)
            # ExecuteRskillTool carries a rskill_id worth surfacing on the
            # span for trace-search drill-down; other variants don't have
            # an obvious "primary key" to record.
            rskill_id = getattr(call, "rskill_id", None)
            if rskill_id:
                span.set_attribute(semconv.REASONER_RSKILL_ID, rskill_id)
                self._last_dispatched = rskill_id
            # Structured log for every successful tick so the multi-task
            # deploy flow can be inspected from the structured log stream
            # without opening Jaeger (Criterion 4 of the libero multi-task
            # goal: subtask, rSkill, localization, VLA execution, reward
            # evaluation, and final outcome are all emitted here).
            _log_kwargs: dict[str, object] = {
                "tick_idx": self._tick_idx,
                "tool": call.tool,
                "tier": prep.tier,
                "trigger": prep.trigger,
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
            # Free text rides the OTLP log export too: the span's consent rule applies.
            rationale = getattr(call, "rationale", None)
            if rationale and free_text_may_leave_host():
                _log_kwargs["rationale"] = rationale
            # Surface the active prompt so the caller can correlate which
            # operator goal drove this tool selection (multi-task tracing).
            if prep.prompts and free_text_may_leave_host():
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
                trigger=prep.trigger,
                traceparent=traceparent,
            )


# Re-export ``ROSReasonerInvalidPlan`` so test code can ``from
# openral_reasoner.core import ROSReasonerInvalidPlan`` without
# reaching into ``openral_core.exceptions``.
__all__ += ["ROSReasonerInvalidPlan"]
