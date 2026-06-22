"""Hardware inference runner.

:class:`DeployRunner` is the first concrete subclass of
:class:`~openral_runner.InferenceRunnerBase` and closes the
inference loop end-to-end on real hardware (or a digital twin):

::

    for each tick at rate_hz:
        for each SensorReader:
            frame = reader.read_latest()
            aggregator.update_image(frame.sensor_id, frame.topic, frame.stamp_wall_ns)
        aggregator.update_joint_state(hal.read_state())
        snapshot = aggregator.snapshot()
        action = skill.step(snapshot)
        try:
            safety_client.check_action(action)        # CLAUDE.md §10
            hal.send_action(action)
        except ROSSafetyViolation as exc:
            record on TickResult.safety_violations
            do NOT call hal.send_action — set action_applied=False

The runner does not manage the :class:`~openral_rskill.Skill`
lifecycle: callers must :meth:`Skill.configure` + :meth:`Skill.activate`
before constructing the runner. The runner does manage HAL connection
and SensorReader open/close as part of its own
:meth:`activate` / :meth:`deactivate`.

In-process image frames flow through the inference hot path
unchanged (``SensorReader.read_latest()`` → ``WorldState.image_frames``
via the aggregator). When a host needs the same frames on a ROS topic
— for the rosbag2 recorder (PR3 of ADR-0019), Foxglove, or
``rqt_image_view`` — :class:`openral_sensors.ros_publisher.SensorRosPublisher`
runs as a *parallel* consumer of the same reader from its own thread.
The GStreamer backend additionally provides a zero-copy tee via
:class:`openral_runner.backends.gstreamer.ros_tee.RosImagePublisher`;
the sensors-side publisher is the universal-but-copying fallback for
OpenCV / RealSense / mock readers.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import structlog
from openral_core import TickResult
from openral_core.exceptions import ROSConfigError, ROSPerceptionStale, ROSSafetyViolation
from openral_hal.protocol import HAL
from openral_observability import metrics as ral_metrics
from openral_observability import producer as ral_producer
from openral_observability import semconv
from openral_observability.tracing_lttng import (
    TP_HAL_READ_STATE,
    TP_HAL_SEND_ACTION,
    lttng_tracepoint,
)
from openral_rskill.base import rSkillBase
from openral_world_state.aggregator import WorldStateAggregator
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from openral_runner.base import InferenceRunnerBase
from openral_runner.safety import NullSafetyClient, SafetyClient
from openral_runner.sensor_reader import SensorReader

__all__ = ["DeployRunner"]

log = structlog.get_logger(__name__)


_modality_for_encoding = ral_producer.modality_for_encoding


class DeployRunner(InferenceRunnerBase):
    """Compose HAL + Skill + WorldStateAggregator + SensorReaders + SafetyClient.

    Mirrors the simulator runner (``openral_sim.SimRunner``) but
    drives real hardware via a :class:`HAL` adapter instead of a sim env.
    Subclass of :class:`InferenceRunnerBase` so the rate-limited loop,
    ``rskill.tick`` OTel parent span, and :class:`RunResult` aggregation
    come for free.

    The runner is the safety-supervisor boundary for the inference loop
    (CLAUDE.md §10): when :meth:`SafetyClient.check_action` raises
    :class:`ROSSafetyViolation`, the runner records the violation on the
    :class:`TickResult` and skips the :meth:`HAL.send_action` call; the
    exception is **not** re-raised because the runner already mitigated
    by withholding the action. Future PRs hook this into the real
    E-stop / incident-log path when the C++ safety kernel lands.

    Args:
        hal: A :class:`~openral_hal.protocol.HAL` adapter. The runner
            calls :meth:`HAL.connect` in :meth:`activate` and
            :meth:`HAL.disconnect` in :meth:`deactivate`.
        skill: A :class:`~openral_rskill.Skill` instance that is
            **already configured + activated**. The runner does not
            re-load weights (that is the caller's responsibility — weight
            loading is heavy and not idempotent).
        aggregator: The :class:`WorldStateAggregator` instance the runner
            feeds joint state + sensor topic refs into each tick.
        sensor_readers: Sequence of :class:`SensorReader` instances. The
            runner opens each in :meth:`activate` and closes in
            :meth:`deactivate`. Frames whose carry-mode is ``topic`` are
            forwarded to :meth:`WorldStateAggregator.update_image`; frames
            carrying inline ``data`` or ``handle`` are noted but not yet
            attached to the snapshot (follow-up).
        safety_client: Optional :class:`SafetyClient`. Defaults to a
            :class:`NullSafetyClient` so digital-twin runs still emit
            ``safety.check`` spans even without the C++ kernel.
        thumbnail_hz: Per-camera rate at which a JPEG thumbnail is encoded
            onto the ``sensors.read_latest`` span for the dashboard.
            Default 25 Hz, decoupled from the tick ``rate_hz``; ``0`` disables
            thumbnails. Throttling keeps ticks above the rate free of image
            bytes so the trace pipeline stays light.
        **base_kwargs: Forwarded to
            :class:`InferenceRunnerBase.__init__` (``rate_hz``,
            ``deadline_overrun_policy``, ``runner_name``,
            ``latency_budget_ms``, ``save_dir``).

    Example:
        >>> # Exercised in tests/unit/test_deploy_runner.py against a
        >>> # real SO100FollowerHAL backed by SO100DigitalTwin.
        >>> pass
    """

    def __init__(
        self,
        *,
        hal: HAL,
        skill: rSkillBase,
        aggregator: WorldStateAggregator,
        sensor_readers: Sequence[SensorReader] = (),
        safety_client: SafetyClient | None = None,
        recorder: object | None = None,
        thumbnail_hz: float = 25.0,
        **base_kwargs: object,
    ) -> None:
        """Initialise the runner; does not open any I/O until :meth:`activate`.

        ``recorder`` is an optional
        :class:`openral_dataset.RolloutRecorder` (ADR-0019 PR3). When set,
        :meth:`episode_start` / :meth:`episode_end` drive the recorder's
        lifecycle and (PR3 follow-up wiring inside ``_tick_impl``) every
        per-tick state + frame + action lands on the attached sinks.
        Typed as ``object`` here so ``openral_runner`` does not import
        ``openral_dataset`` (the dataset package imports observability,
        not the other way around — keeping the dep DAG one-way).
        """
        super().__init__(**base_kwargs)  # type: ignore[arg-type]
        self._hal = hal
        self._skill = skill
        self._aggregator = aggregator
        self._sensor_readers: list[SensorReader] = list(sensor_readers)
        self._safety_client: SafetyClient = (
            safety_client if safety_client is not None else NullSafetyClient()
        )
        self._recorder = recorder
        # Tracks whether an episode is currently "open" on the recorder.
        # episode_start sets to True; episode_end clears. deactivate
        # closes a still-open episode as a failure (mirrors SimRunner).
        self._recorder_episode_open: bool = False
        # Cached episode counter for episode_start's return value. Sim
        # exposes _episode_idx as a public-ish field; hardware tracks it
        # internally because we have no env-driven episode signal.
        self._hw_episode_idx: int = -1
        # Short, low-cardinality label for HAL spans + metrics — the
        # adapter class name (``SO100FollowerHAL`` → ``so100_followerhal``)
        # is short enough for Prom labels and unique per backend. The
        # full module path is not — it bloats label sets.
        self._hal_adapter_label = type(hal).__name__.lower()
        # Dashboard thumbnail throttle. Encoding a JPEG on every tick (up to
        # rate_hz) bloats every sensors.read_latest span and the trace
        # pipeline; the dashboard only needs an at-a-glance preview. Gate per
        # camera to thumbnail_hz (0 disables), decoupled from rate_hz, using a
        # wall-clock (time.monotonic) deadline per sensor_id.
        self._thumbnail_hz = float(thumbnail_hz)
        self._thumb_next_due: dict[str, float] = {}
        # Monotonic clock the thumbnail gate reads. A seam (not a config knob):
        # production always uses ``time.monotonic``; tests inject a deterministic
        # clock so the per-camera emit count is independent of real wall-clock
        # tick throughput under load.
        self._thumbnail_clock: Callable[[], float] = time.monotonic

    # ── ADR-0019 PR3 — episode boundary API ────────────────────────────────

    def episode_start(self, task_string: str) -> int:
        """Open a new episode on the attached :class:`RolloutRecorder`.

        Idempotent within an episode: a second call without
        :meth:`episode_end` raises :class:`RuntimeError` (the same
        contract the recorder enforces internally).

        Args:
            task_string: Task instruction; lands on the bag's
                ``/openral/episode`` ``task_string`` field.

        Returns:
            The new ``episode_idx``. ``-1`` when no recorder is
            attached (caller can detect "no recorder" if needed).
        """
        if self._recorder is None:
            return -1
        if self._recorder_episode_open:
            raise RuntimeError(
                f"episode {self._hw_episode_idx} is still open on the recorder; "
                "call episode_end() first"
            )
        idx_raw = self._recorder.episode_start(task_string=task_string)  # type: ignore[attr-defined]
        idx = int(idx_raw)
        self._hw_episode_idx = idx
        self._recorder_episode_open = True
        return idx

    def episode_end(self, *, success: bool) -> None:
        """Close the current episode on the recorder with the success flag.

        No-op when no recorder is attached.

        Raises:
            RuntimeError: When called without a matching
                :meth:`episode_start`.
        """
        if self._recorder is None:
            return
        if not self._recorder_episode_open:
            raise RuntimeError("no recorder episode open; call episode_start() first")
        self._recorder.episode_end(success=success)  # type: ignore[attr-defined]
        self._recorder_episode_open = False

    @property
    def _tracer(self) -> trace.Tracer:
        # Per-call resolution — see also ``aggregator._tracer``: caching at
        # module / __init__ time binds to the TracerProvider that was
        # global when ``DeployRunner`` was first constructed and silently
        # drops spans after a provider swap.
        return trace.get_tracer("openral")

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def activate(self) -> None:
        """Open HAL connection + every SensorReader. Asserts Skill is active.

        Idempotent: re-activating after :meth:`deactivate` re-opens
        everything. Subclasses may override; call ``super().activate()``
        first so the tick counter is reset and the active flag is set.
        """
        super().activate()
        self._hal.connect()
        for reader in self._sensor_readers:
            reader.open()
        log.info(
            "deploy_runner.activated",
            runner=self._runner_name,
            robot=self._hal.description.name,
            sensor_count=len(self._sensor_readers),
        )

    def deactivate(self) -> None:
        """Close every SensorReader, disconnect HAL. Idempotent. Best-effort.

        Reader / HAL teardown errors are logged but do not propagate —
        deactivation must succeed even when one path is degraded so the
        next activation has a clean slate.
        """
        for reader in self._sensor_readers:
            try:
                reader.close()
            except Exception as exc:  # reason: log + continue on teardown
                log.warning(
                    "deploy_runner.sensor_close_error",
                    runner=self._runner_name,
                    sensor_id=reader.sensor_id,
                    exc=str(exc),
                )
        try:
            self._hal.disconnect()
        except Exception as exc:  # reason: log + continue on teardown
            log.warning(
                "deploy_runner.hal_disconnect_error",
                runner=self._runner_name,
                exc=str(exc),
            )
        # ADR-0019 PR3 — finalize the recorder. Close any still-open
        # episode as a failure (matches the SimRunner contract) so
        # sinks see a complete lifecycle.
        if self._recorder is not None:
            if self._recorder_episode_open:
                try:
                    self._recorder.episode_end(success=False)  # type: ignore[attr-defined]
                except (ValueError, RuntimeError) as exc:
                    log.warning(
                        "deploy_runner.recorder_episode_end_failed",
                        exc=str(exc),
                    )
                self._recorder_episode_open = False
            try:
                self._recorder.finalize()  # type: ignore[attr-defined]
            except (ValueError, RuntimeError) as exc:
                log.warning("deploy_runner.recorder_finalize_failed", exc=str(exc))
        super().deactivate()

    def _thumbnail_due(self, sensor_id: str, now: float) -> bool:
        """Per-camera rate gate for dashboard thumbnails.

        Returns ``True`` (and advances the per-camera deadline) when a
        thumbnail is due at ``thumbnail_hz``; ``False`` otherwise. ``0`` Hz
        disables. Deadlines advance from the *previous* deadline (not from
        ``now``), so the long-run average holds at ``thumbnail_hz`` even when
        ticks run only slightly faster than it — advancing from ``now`` would
        quantise the emit rate down to a tick subharmonic (e.g. 25 Hz against a
        28 Hz tick collapses to ~14 Hz). A deadline that has fallen more than a
        period behind (cold start / long stall) is resynced to ``now + period``
        so the gate never bursts to catch up.
        """
        if self._thumbnail_hz <= 0.0:
            return False
        due = self._thumb_next_due.get(sensor_id)
        if due is not None and now < due:
            return False
        period = 1.0 / self._thumbnail_hz
        nxt = (due + period) if due is not None else (now + period)
        if nxt <= now:
            nxt = now + period
        self._thumb_next_due[sensor_id] = nxt
        return True

    # ── Hot path ────────────────────────────────────────────────────────────

    def _tick_impl(  # noqa: PLR0915  # reason: five sequential tick phases each carry their own bookkeeping
        self, tick_idx: int
    ) -> TickResult:
        """Read sensors + HAL, run Skill, gate on safety, dispatch action.

        Per-stage wall-times are recorded into the :class:`TickResult`;
        :class:`InferenceRunnerBase.tick` lifts them onto the
        ``rskill.tick`` OTel parent span.
        """
        tick_start = time.perf_counter()
        stamp_ns = time.time_ns()
        safety_violations: list[str] = []
        action_applied = True

        # ── 1. Sensors ──────────────────────────────────────────────────────
        sensors_t0 = time.perf_counter()
        tick_wall_ns = stamp_ns
        for reader in self._sensor_readers:
            try:
                with self._tracer.start_as_current_span(
                    semconv.SPAN_SENSORS_READ_LATEST,
                    attributes={
                        semconv.SENSORS_SOURCE: reader.sensor_id,
                        semconv.TICK_IDX: tick_idx,
                    },
                ) as sensor_span:
                    frame = reader.read_latest()
                    # Per-source age at read time. ``stamp_wall_ns`` is the
                    # frame's capture-time stamp; subtract from the tick's
                    # wall-clock to get the freshness budget.
                    age_ms = (tick_wall_ns - frame.stamp_wall_ns) / 1e6
                    modality = _modality_for_encoding(frame.encoding)
                    # Per-camera thumbnail gate (dashboard preview): encode +
                    # attach a JPEG only when due at ``thumbnail_hz``, so most
                    # ticks carry zero image bytes and skip the encode — the
                    # trace path stays light while the dashboard refreshes at
                    # ~thumbnail_hz. ``encode_frame_thumbnail`` returns None for
                    # unrenderable frames (DEPTH16/CUDA_NV12/RAW) or topic-ref /
                    # GPU-handle frames; the gate has already advanced, so a
                    # non-renderable camera does not retry the encode each tick.
                    thumb_bytes = (
                        ral_producer.encode_frame_thumbnail(frame)
                        if self._thumbnail_due(reader.sensor_id, self._thumbnail_clock())
                        else None
                    )
                    ral_producer.record_sensor_frame_attrs(
                        sensor_span,
                        modality=modality,
                        encoding=frame.encoding.value,
                        width=frame.width,
                        height=frame.height,
                        channels=frame.channels,
                        age_ms=age_ms,
                        thumbnail_bytes=thumb_bytes,
                    )
                    ral_metrics.record_histogram_ms(
                        ral_metrics.get_sensors_age_ms(),
                        age_ms,
                        {semconv.LABEL_MODALITY: modality},
                    )
            except ROSPerceptionStale as exc:
                # Stale sensor: leave the aggregator's last topic ref in
                # place. The diagnostics field on WorldState will surface
                # the staleness when the aggregator next snapshots.
                current = trace.get_current_span()
                if current.get_span_context().is_valid:
                    current.add_event(
                        semconv.EVENT_SENSOR_STALE,
                        attributes={
                            semconv.SENSORS_SOURCE: reader.sensor_id,
                            semconv.TICK_IDX: tick_idx,
                        },
                    )
                ral_metrics.get_sensors_stale_reads().add(1, {semconv.LABEL_MODALITY: "unknown"})
                log.debug(
                    "deploy_runner.sensor_stale",
                    sensor_id=reader.sensor_id,
                    exc=str(exc),
                )
                continue
            if frame.topic is not None:
                self._aggregator.update_image(frame.sensor_id, frame.topic, frame.stamp_wall_ns)
            # frame.data / frame.handle carry-modes are not yet wired into
            # WorldState.image_frames — that's a follow-up PR (image_frames
            # field exists on the schema today, just no producer).
        sensors_ms = (time.perf_counter() - sensors_t0) * 1e3

        # ── 2. HAL → WorldState snapshot ───────────────────────────────────
        ws_t0 = time.perf_counter()
        hal_adapter = self._hal_adapter_label
        read_attrs = {
            semconv.LABEL_HAL_ADAPTER: hal_adapter,
            semconv.LABEL_ROBOT_MODEL: self._hal.description.name,
        }
        with self._tracer.start_as_current_span(
            semconv.SPAN_HAL_READ_STATE,
            attributes={
                semconv.HAL_ADAPTER: hal_adapter,
                semconv.HAL_ROBOT_MODEL: self._hal.description.name,
                semconv.TICK_IDX: tick_idx,
            },
        ) as hal_read_span:
            read_t0 = time.perf_counter()
            with lttng_tracepoint(TP_HAL_READ_STATE, tick_idx=tick_idx, adapter=hal_adapter):
                state = self._hal.read_state()
            ral_metrics.record_histogram_ms(
                ral_metrics.get_hal_read_state_duration(),
                (time.perf_counter() - read_t0) * 1e3,
                read_attrs,
            )
            # Surface joint-level reality on the span so the dashboard
            # can render the per-joint reality strip + ee/gripper card.
            joint_specs = self._hal.description.joints
            ral_producer.record_joint_state(
                hal_read_span,
                names=state.name,
                positions=state.position,
                velocities=state.velocity,
                efforts=state.effort,
                position_limits=[j.position_limits for j in joint_specs],
                velocity_limits=[j.velocity_limit for j in joint_specs],
                effort_limits=[j.effort_limit for j in joint_specs],
                stamp_ns=state.stamp_ns,
            )
        self._aggregator.update_joint_state(state)
        snapshot = self._aggregator.snapshot()
        world_state_ms = (time.perf_counter() - ws_t0) * 1e3

        # ── 3. Inference ───────────────────────────────────────────────────
        inf_t0 = time.perf_counter()
        step_result = self._skill.step(snapshot)
        inference_ms = (time.perf_counter() - inf_t0) * 1e3
        # ADR-0028b — ``step()`` may return ``Action | list[Action]``.
        # DeployRunner is the live-hardware path; today its safety
        # + HAL dispatch + observability code assumes a single Action.
        # Until ADR-0028c lands per-mode HAL handlers on real hardware,
        # surface list returns as a typed config error instead of
        # silently picking the first one.
        if isinstance(step_result, list):
            raise ROSConfigError(
                f"DeployRunner: skill {self._skill.name!r} returned "
                f"{len(step_result)} Actions (ADR-0028b multi-surface). "
                "Real-hardware HALs gain per-mode dispatch in ADR-0028c; "
                "run this skill via rskill_runner_node (sim path) until then."
            )
        action = step_result

        # ── 4. Safety gate ─────────────────────────────────────────────────
        safety_t0 = time.perf_counter()
        try:
            self._safety_client.check_action(action)
        except ROSSafetyViolation as exc:
            # Supervisor boundary: record + withhold action. We do NOT
            # re-raise — withholding the action IS the mitigation today.
            # When the C++ safety kernel lands, this branch will also
            # trigger the E-stop / incident-log path.
            safety_violations.append(str(exc))
            action_applied = False
            current = trace.get_current_span()
            if current.get_span_context().is_valid:
                # ``record_exception`` adds an ``exception`` event with
                # full ``exception.type`` / ``exception.stacktrace`` —
                # the type survives even when the runner records the
                # violation onto the TickResult's plain list of strings.
                current.record_exception(exc)
                current.set_status(StatusCode.ERROR, str(exc))
                current.add_event(
                    semconv.EVENT_SAFETY_VIOLATION,
                    attributes={
                        semconv.SAFETY_CHECK_NAME: type(exc).__name__,
                        semconv.SAFETY_SEVERITY: "violation",
                        semconv.TICK_IDX: tick_idx,
                    },
                )
            ral_metrics.get_safety_violations().add(
                1,
                {
                    semconv.LABEL_CHECK_NAME: type(exc).__name__,
                    semconv.LABEL_SEVERITY: "violation",
                },
            )
            log.warning(
                "deploy_runner.safety_violation",
                runner=self._runner_name,
                tick_idx=tick_idx,
                exc=str(exc),
                exc_type=type(exc).__name__,
            )
        safety_ms = (time.perf_counter() - safety_t0) * 1e3

        # ── 5. HAL dispatch ────────────────────────────────────────────────
        hal_t0 = time.perf_counter()
        if action_applied:
            send_attrs = {
                semconv.LABEL_HAL_ADAPTER: self._hal_adapter_label,
                semconv.LABEL_CONTROL_MODE: action.control_mode.value,
            }
            with self._tracer.start_as_current_span(
                semconv.SPAN_HAL_SEND_ACTION,
                attributes={
                    semconv.HAL_ADAPTER: self._hal_adapter_label,
                    semconv.HAL_CONTROL_MODE: action.control_mode.value,
                    semconv.TICK_IDX: tick_idx,
                },
            ) as hal_send_span:
                send_t0 = time.perf_counter()
                with lttng_tracepoint(
                    TP_HAL_SEND_ACTION,
                    tick_idx=tick_idx,
                    adapter=self._hal_adapter_label,
                    control_mode=action.control_mode.value,
                ):
                    self._hal.send_action(action)
                ral_metrics.record_histogram_ms(
                    ral_metrics.get_hal_send_action_duration(),
                    (time.perf_counter() - send_t0) * 1e3,
                    send_attrs,
                )
                # Surface the row of the action chunk we just applied so
                # the dashboard can overlay command-vs-reality. We only
                # record one row (not the full horizon x dim chunk).
                next_row: list[float] | None = None
                if action.joint_targets:
                    next_row = list(action.joint_targets[0])
                elif action.joint_velocities:
                    next_row = list(action.joint_velocities[0])
                elif action.joint_torques:
                    next_row = list(action.joint_torques[0])
                gripper_next = action.gripper[0] if action.gripper else None
                ral_producer.record_action(
                    hal_send_span,
                    next_row=next_row,
                    dim=len(next_row) if next_row else None,
                    horizon=action.horizon,
                    applied=action_applied,
                    gripper_position=gripper_next,
                )
        hal_ms = (time.perf_counter() - hal_t0) * 1e3

        tick_ms = (time.perf_counter() - tick_start) * 1e3
        result = TickResult(
            stamp_ns=stamp_ns,
            tick_idx=tick_idx,
            sensors_ms=sensors_ms,
            world_state_ms=world_state_ms,
            inference_ms=inference_ms,
            safety_ms=safety_ms,
            hal_ms=hal_ms,
            tick_ms=tick_ms,
            chunk_index=None,
            safety_violations=safety_violations,
            action_applied=action_applied,
        )
        # ADR-0019 PR3 — fan out to the attached RolloutRecorder. Only
        # when a recorder is attached AND an episode is open; otherwise
        # this is a no-op and the hot path stays unchanged.
        if self._recorder is not None and self._recorder_episode_open:
            self._record_tick_to_recorder(snapshot, action, action_applied, stamp_ns)
        return result

    def _record_tick_to_recorder(
        self,
        snapshot: object,
        action: object,
        action_applied: bool,
        stamp_ns: int,
    ) -> None:
        """Fan one tick's state + frame + action to the attached recorder.

        Splits out of ``_tick_impl`` to keep that method readable.
        Errors are logged + swallowed: the hardware path must never
        crash on a dataset-recording side-effect.
        """
        # ``numpy`` is a heavy import we keep out of module top so the
        # runner stays import-safe on hosts without numpy (the dataset
        # path is optional).
        import numpy as np  # noqa: PLC0415

        try:
            joint_state = getattr(snapshot, "joint_state", None)
            positions = getattr(joint_state, "position", None) if joint_state else None
            if positions is None:
                # Snapshot was unavailable; build a zero placeholder of
                # the shape the recorder expects so the row still lands.
                expected_shape = self._recorder.expected_state_shape  # type: ignore[union-attr]
                state_array = np.zeros(expected_shape, dtype=np.float32)
            else:
                state_array = np.asarray(list(positions), dtype=np.float32)

            # Action vector: first row of the action chunk if present.
            joint_targets = getattr(action, "joint_targets", None)
            if joint_targets:
                action_array = np.asarray(joint_targets[0], dtype=np.float32)
            else:
                action_dim = state_array.shape[0]
                action_array = np.zeros(action_dim, dtype=np.float32)

            # Per-camera images. Hardware reads frames from
            # SensorReader; here we forward whatever was captured this
            # tick. Frame keys come from the recorder so the schema
            # matches the LeRobot v3 features dict.
            images: dict[str, np.ndarray[Any, Any]] = {}
            for cam_key in self._recorder.expected_image_keys():  # type: ignore[union-attr]
                # Hardware-side per-camera frames are recorded via the
                # ROS publisher (PR2) which writes its own dataset rows;
                # the in-line path here doesn't have access to the raw
                # uint8 buffers. Provide a zero placeholder so the
                # schema is satisfied. PR4's converter will
                # post-process by joining /openral/tick with the camera
                # streams from the bag for the authoritative frames.
                images[cam_key] = np.zeros((1, 1, 3), dtype=np.uint8)

            self._recorder.record_frame(  # type: ignore[union-attr]
                observation_state=state_array,
                images=images,
                action=action_array,
                reward=0.0,
                terminated=False,
                truncated=False,
                stamp_ns=stamp_ns,
            )
        except (ValueError, RuntimeError, AttributeError) as exc:
            log.warning(
                "hardware.recorder_record_frame_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                episode_idx=self._hw_episode_idx,
            )
            _ = action_applied  # parameter retained for signature symmetry
