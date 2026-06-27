"""Shared base for inference runners.

:class:`InferenceRunnerBase` owns the rate-limited loop, the OTel
``rskill.tick`` parent span, the per-tick :class:`TickResult` collection,
the aggregate :class:`RunResult` (mean / p99 timings, budget violations,
trace id), and the deadline-overrun policy. Subclasses implement
:meth:`_tick_impl` which performs one actual tick and returns its
:class:`TickResult`; the base records the per-stage timings on the parent
span and decides whether the cadence was honoured.

The base is plain Python: it does not import HAL, sensors, or ROS. The
two concrete runners — :class:`SimRunner` (in ``openral_sim``, future
PR) and :class:`DeployRunner` (in ``openral_runner.deploy_runner``,
PR F) — wire their respective input / output stacks into the
:meth:`_tick_impl` hook.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import structlog
from openral_core import DeadlineOverrunPolicy, RunResult, TickResult
from openral_core.exceptions import ROSDeadlineMissed
from openral_observability import metrics as ral_metrics
from openral_observability import rskill_span, semconv
from openral_observability.tracing_lttng import TP_RUNNER_TICK, lttng_tracepoint
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from openral_runner.clock import sleep_until

__all__ = ["InferenceRunnerBase"]

log = structlog.get_logger(__name__)


def _percentile(samples: list[float], q: float) -> float:
    """Linear-interpolation percentile of a non-empty sample list.

    ``q`` is in [0.0, 1.0]. Returns 0.0 for an empty list (callers gate on
    ``n_ticks > 0`` before reading the field).
    """
    if not samples:
        return 0.0
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    idx_f = q * (len(s) - 1)
    lo = int(idx_f)
    hi = min(lo + 1, len(s) - 1)
    frac = idx_f - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


class InferenceRunnerBase(ABC):
    """Abstract base class for inference runners.

    Concrete subclasses override :meth:`_tick_impl` to perform one tick
    against their input/output stack (sim env vs real HAL + sensors). The
    base class provides:

    * Rate-limited :meth:`run` using
      :func:`~openral_runner.clock.sleep_until`.
    * One OTel ``rskill.tick`` parent span per tick (via
      :func:`~openral_observability.rskill_span`). The base attaches
      per-stage timing attributes lifted from the returned :class:`TickResult`
      so child spans (``inference_span`` / ``safety_span``) automatically
      correlate.
    * :class:`RunResult` aggregation: mean / p99 inference and tick latencies,
      budget-violation count, OTel trace id, save_dir, and arbitrary
      ``metadata``.
    * Deadline-overrun policy: ``warn`` logs + records on the parent span;
      ``drop`` is reported (the subclass is responsible for the action
      itself); ``raise`` raises :class:`ROSDeadlineMissed` (test mode).

    Args:
        rate_hz: Foreground tick rate. Default 30 Hz.
        deadline_overrun_policy: What to do when ``tick_ms > 1 / rate_hz``.
        runner_name: Span ``skill.id`` attribute — useful when multiple
            runners share a trace.
        latency_budget_ms: If set, ticks whose ``tick_ms`` exceeds this
            count toward :attr:`RunResult.budget_violations`. ``None``
            disables the check.
        save_dir: Optional artefact directory, forwarded into
            :class:`RunResult`. The base class does not write anything; the
            subclass owns that.
    """

    rate_hz: float

    def __init__(
        self,
        *,
        rate_hz: float = 30.0,
        deadline_overrun_policy: DeadlineOverrunPolicy = DeadlineOverrunPolicy.WARN,
        runner_name: str = "inference_runner",
        latency_budget_ms: float | None = None,
        save_dir: str | None = None,
    ) -> None:
        """Initialise the base; subclasses MUST call ``super().__init__(...)``."""
        if rate_hz <= 0:
            raise ValueError(f"InferenceRunnerBase.rate_hz must be > 0; got {rate_hz}")
        self.rate_hz = rate_hz
        self.deadline_overrun_policy = deadline_overrun_policy
        self._runner_name = runner_name
        self._latency_budget_ms = latency_budget_ms
        self._save_dir = save_dir
        self._tick_idx = 0
        self._active = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def activate(self) -> None:
        """Mark the runner as ready to tick. Idempotent. Subclasses may extend."""
        self._tick_idx = 0
        self._active = True

    def deactivate(self) -> None:
        """Stop ticking. Idempotent. Subclasses may extend."""
        self._active = False

    # ── Hot path ────────────────────────────────────────────────────────────

    @abstractmethod
    def _tick_impl(self, tick_idx: int) -> TickResult:
        """Run one tick and return a populated :class:`TickResult`.

        The base class wraps every call in a ``rskill.tick`` span and lifts
        the returned timings onto that span. Subclasses should not open the
        parent span themselves.

        Args:
            tick_idx: 0-indexed tick counter inside this run.

        Returns:
            The tick's record (timings, safety violations, action_applied).
        """

    def _should_terminate(self) -> bool:
        """Subclass hook: early-exit signal evaluated after each tick.

        Default returns ``False`` so :class:`DeployRunner` runs until
        ``max_ticks`` (or :meth:`deactivate`) as before. :class:`SimRunner`
        overrides this to stop once ``n_episodes`` have completed without
        depending on the caller picking the exact tick budget. The hook is
        consulted after the tick is recorded and deadline-overrun
        bookkeeping is done, before :func:`sleep_until` waits on the next
        deadline.
        """
        return False

    # ── ADR-0019 PR3 — explicit episode boundary API ────────────────────────
    #
    # Sim derives episodes from the env's terminated / truncated flags;
    # hardware has no such signal. The episode boundary on a real robot
    # is "the operator pressed go" → "the BT executor reported done", so
    # the runner exposes explicit start/end hooks. Both default to
    # NotImplementedError so a subclass that needs them must opt in;
    # SimRunner overrides them as no-ops because its `_finalize_episode`
    # path already drives the attached RolloutRecorder.

    def episode_start(self, task_string: str) -> int:
        """Begin a new episode on this runner.

        Hardware path uses this to open a new episode on the attached
        :class:`openral_dataset.RolloutRecorder` (and, transitively,
        on the :class:`Rosbag2Sink` or :class:`LeRobotDatasetSink`).

        Args:
            task_string: Natural-language task instruction; lands on
                the bag's ``/openral/episode`` ``task_string`` field.

        Returns:
            The new ``episode_idx``.

        Raises:
            NotImplementedError: Subclasses without an explicit
                episode-boundary contract reject the call cleanly so
                a wiring bug surfaces immediately, rather than silently
                producing a malformed dataset.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement episode_start; "
            "hardware-style runners must override this method"
        )

    def episode_end(self, *, success: bool) -> None:
        """Close the current episode on this runner with the success flag.

        Args:
            success: Episode-level outcome. The downstream
                :class:`openral_dataset.RolloutRecorder` tags every
                frame's ``next.success`` from this value at conversion
                time (PR4).

        Raises:
            NotImplementedError: As for :meth:`episode_start` —
                subclasses without an episode boundary contract reject
                the call so wiring bugs are loud.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement episode_end; "
            "hardware-style runners must override this method"
        )

    def tick(self) -> TickResult:
        """Single-tick entry point (public Protocol method).

        Wraps :meth:`_tick_impl` in a ``rskill.tick`` OTel parent span and
        attaches the per-stage timing attributes returned in the
        :class:`TickResult`. Records the tick on the
        ``openral.tick.duration`` histogram and increments
        ``openral.safety.violations`` on any safety violation. Does
        **not** enforce cadence — call :meth:`run` for that.
        """
        idx = self._tick_idx
        # ADR-0018 F9 — LTTng entry/exit around the whole tick. No-op
        # when OPENRAL_ROS2_TRACING is unset; on, the begin/end pair
        # frames a ~30 Hz interval that babeltrace2 can correlate
        # against kernel scheduler events.
        with (
            rskill_span(
                "rskill.tick",
                rskill_id=self._runner_name,
                tick_idx=idx,
                rate_hz=self.rate_hz,
            ) as span,
            lttng_tracepoint(
                TP_RUNNER_TICK,
                rskill_id=self._runner_name,
                tick_idx=idx,
                rate_hz=self.rate_hz,
            ),
        ):
            result = self._tick_impl(idx)
            span.set_attribute(semconv.RSKILL_TICK_MS, result.tick_ms)
            span.set_attribute(semconv.RSKILL_INFERENCE_MS, result.inference_ms)
            span.set_attribute(semconv.RSKILL_SENSORS_MS, result.sensors_ms)
            span.set_attribute(semconv.RSKILL_WORLD_STATE_MS, result.world_state_ms)
            span.set_attribute(semconv.RSKILL_SAFETY_MS, result.safety_ms)
            span.set_attribute(semconv.RSKILL_HAL_MS, result.hal_ms)
            span.set_attribute(semconv.RSKILL_ACTION_APPLIED, result.action_applied)
            span.set_attribute(semconv.TICK_IDX, idx)
            if result.safety_violations:
                span.set_attribute(semconv.RSKILL_SAFETY_VIOLATIONS, result.safety_violations)
            # Sim-only fields. Skipped on hardware
            # where they're left at None.
            if result.episode_idx is not None:
                span.set_attribute(semconv.RSKILL_EPISODE_IDX, result.episode_idx)
            if result.step_idx is not None:
                span.set_attribute(semconv.RSKILL_STEP_IDX, result.step_idx)
            if result.reward is not None:
                span.set_attribute(semconv.RSKILL_REWARD, result.reward)
            if result.terminated is not None:
                span.set_attribute(semconv.RSKILL_TERMINATED, result.terminated)
            if result.truncated is not None:
                span.set_attribute(semconv.RSKILL_TRUNCATED, result.truncated)

        # Record tick latency + per-stage timings on the histogram outside
        # the span so a slow exporter never blocks the hot path. Labels
        # follow the closed-set whitelist in design §9.
        base_attrs = {semconv.LABEL_RSKILL_ID: self._runner_name}
        # The configured latency budget rides along as a per-data-point threshold
        # so the dashboard can draw a budget line + breach coloring on the
        # tick.duration sparkline. Constant per runner, so no extra cardinality.
        tick_attrs: dict[str, str | float] = dict(base_attrs)
        if self._latency_budget_ms is not None:
            tick_attrs[semconv.METRIC_THRESHOLD_MS] = self._latency_budget_ms
        ral_metrics.record_histogram_ms(ral_metrics.get_tick_duration(), result.tick_ms, tick_attrs)
        ral_metrics.record_histogram_ms(
            ral_metrics.get_inference_duration(),
            result.inference_ms,
            base_attrs,
        )
        if result.safety_violations:
            # ``safety_violations`` is a list of human-readable strings on
            # ``TickResult``; the counter just records that the tick had
            # at least one. Per-exception-type counters are emitted at the
            # supervisor boundary in ``DeployRunner._tick_impl``.
            ral_metrics.get_safety_violations().add(
                len(result.safety_violations),
                {
                    semconv.LABEL_CHECK_NAME: "runtime",
                    semconv.LABEL_SEVERITY: "violation",
                },
            )

        self._tick_idx = idx + 1
        return result

    def run(self, max_ticks: int | None = None) -> RunResult:
        """Rate-limited tick loop.

        Iterates :meth:`tick` at :attr:`rate_hz` until ``max_ticks`` is
        reached or :meth:`deactivate` is called from elsewhere. After each
        tick, the next deadline is computed as ``previous_deadline +
        1 / rate_hz`` and :func:`sleep_until` waits for it. When the tick
        overruns the deadline, the configured
        :class:`DeadlineOverrunPolicy` decides whether to ``warn`` / ``drop``
        / ``raise``.

        Args:
            max_ticks: Stop after this many ticks. ``None`` means "run
                until :meth:`deactivate`".

        Returns:
            Aggregated :class:`RunResult` (mean / p99 timings, budget
            violations, trace id).

        Raises:
            ROSDeadlineMissed: When the configured policy is ``RAISE`` and a
                tick exceeds the period.
        """
        if not self._active:
            self.activate()

        period = 1.0 / self.rate_hz
        deadline = time.perf_counter()
        results: list[TickResult] = []
        budget_violations = 0
        trace_id = self._current_trace_id()

        try:
            while self._active and (max_ticks is None or self._tick_idx < max_ticks):
                pre = time.perf_counter()
                result = self.tick()
                results.append(result)
                if self._latency_budget_ms is not None and result.tick_ms > self._latency_budget_ms:
                    budget_violations += 1
                    ral_metrics.get_tick_budget_violations().add(
                        1, {semconv.LABEL_RSKILL_ID: self._runner_name}
                    )

                # Cadence enforcement.
                deadline += period
                if pre + result.tick_ms / 1e3 > deadline:
                    self._on_deadline_overrun(result)
                if self._should_terminate():
                    break
                sleep_until(deadline)
        finally:
            # Note: do not auto-deactivate — leaves the runner re-runnable.
            pass

        return self._build_run_result(
            results, budget_violations=budget_violations, trace_id=trace_id
        )

    # ── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _current_trace_id() -> str | None:
        """Return the active OTel trace id (hex) or ``None`` when no span is active."""
        span = trace.get_current_span()
        if span is None:
            return None
        ctx = span.get_span_context()
        if not ctx.is_valid:
            return None
        return f"{ctx.trace_id:032x}"

    def _on_deadline_overrun(self, result: TickResult) -> None:
        """Apply the configured :class:`DeadlineOverrunPolicy`.

        Always increments ``openral.tick.deadline_misses`` and emits a
        ``openral.event.deadline_missed`` span event on the current
        ``rskill.tick`` span (it has already exited by the time this
        method runs, so use the inactive-span-safe ``record_exception``
        / ``add_event`` on the current span if any). The policy decides
        whether to also log, raise, or just record.
        """
        budget_ms = 1000.0 / self.rate_hz
        ral_metrics.get_tick_deadline_misses().add(1, {semconv.LABEL_RSKILL_ID: self._runner_name})
        # If we're still inside a higher-level parent span (the CLI root),
        # surface the miss there too — Jaeger reviewers expect to see the
        # event without drilling into every child.
        current = trace.get_current_span()
        if current.get_span_context().is_valid:
            current.add_event(
                semconv.EVENT_DEADLINE_MISSED,
                attributes={
                    semconv.TICK_IDX: result.tick_idx,
                    semconv.RSKILL_TICK_MS: result.tick_ms,
                    semconv.TICK_DEADLINE_MS: budget_ms,
                },
            )

        if self.deadline_overrun_policy == DeadlineOverrunPolicy.WARN:
            log.warning(
                "inference_runner.deadline_missed",
                runner=self._runner_name,
                tick_idx=result.tick_idx,
                tick_ms=result.tick_ms,
                budget_ms=budget_ms,
            )
        elif self.deadline_overrun_policy == DeadlineOverrunPolicy.DROP:
            log.warning(
                "inference_runner.deadline_missed.drop",
                runner=self._runner_name,
                tick_idx=result.tick_idx,
                tick_ms=result.tick_ms,
                budget_ms=budget_ms,
                action_applied=result.action_applied,
            )
        elif self.deadline_overrun_policy == DeadlineOverrunPolicy.RAISE:
            exc = ROSDeadlineMissed(
                f"Tick {result.tick_idx} exceeded {budget_ms:.3f} ms "
                f"deadline: {result.tick_ms:.3f} ms"
            )
            if current.get_span_context().is_valid:
                current.record_exception(exc)
                current.set_status(StatusCode.ERROR, str(exc))
            raise exc

    def _build_run_result(
        self,
        results: list[TickResult],
        *,
        budget_violations: int,
        trace_id: str | None,
    ) -> RunResult:
        """Aggregate per-tick records into a :class:`RunResult`."""
        n = len(results)
        if n == 0:
            return RunResult(
                n_ticks=0,
                budget_violations=budget_violations,
                trace_id=trace_id,
                save_dir=self._save_dir,
            )
        inference_ms = [r.inference_ms for r in results]
        tick_ms = [r.tick_ms for r in results]
        return RunResult(
            n_ticks=n,
            budget_violations=budget_violations,
            avg_inference_ms=sum(inference_ms) / n,
            p99_inference_ms=_percentile(inference_ms, 0.99),
            avg_tick_ms=sum(tick_ms) / n,
            p99_tick_ms=_percentile(tick_ms, 0.99),
            trace_id=trace_id,
            save_dir=self._save_dir,
        )
