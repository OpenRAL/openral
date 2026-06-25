"""DeployRunner throttles dashboard thumbnail emission to thumbnail_hz/camera.

Real twin + real skill + real synthetic camera readers (no mocks per
CLAUDE.md §1.11). A :class:`_SyntheticRgbReader` is a genuine
:class:`~openral_runner.sensor_reader.SensorReader` implementation — it
produces real RGB8 :class:`~openral_core.SensorFrame` objects that flow
through the real Pillow encode path — analogous to a videotestsrc camera.
The gate is wall-clock (``time.monotonic``) based, so the rate assertions
allow jitter rather than demanding an exact count.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from openral_core import Action, ControlMode, SensorFrame
from openral_core.schemas import FrameEncoding, WorldState
from openral_hal.so100_follower import SO100FollowerHAL
from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig
from openral_observability import semconv
from openral_rskill.base import rSkillBase
from openral_runner import DeployRunner
from openral_world_state.aggregator import WorldStateAggregator
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class _SyntheticRgbReader:
    """A real SensorReader that always returns a fresh RGB8 test frame."""

    def __init__(self, sensor_id: str, *, w: int = 320, h: int = 240) -> None:
        self.sensor_id = sensor_id
        self.is_open = False
        self._w = w
        self._h = h
        self._data = bytes([64, 128, 192] * (w * h))

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def read_latest(self, max_age_ms: int | None = None) -> SensorFrame:
        del max_age_ms
        now = time.time_ns()
        return SensorFrame(
            sensor_id=self.sensor_id,
            stamp_monotonic_ns=time.monotonic_ns(),
            stamp_wall_ns=now,
            encoding=FrameEncoding.RGB8,
            width=self._w,
            height=self._h,
            channels=3,
            data=self._data,
        )


class _NoOpSkill(rSkillBase):
    def __init__(self) -> None:
        super().__init__(name="throttle_test_skill", embodiment_tags=["so100_follower"])

    def _configure_impl(self) -> None:
        return None

    def _activate_impl(self) -> None:
        return None

    def _deactivate_impl(self) -> None:
        return None

    def _shutdown_impl(self) -> None:
        return None

    def _step_impl(self, world_state: WorldState) -> Action:
        del world_state
        return Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[0.0] * 6],
            confidence=1.0,
        )


@pytest.fixture
def memory_exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace.set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        exporter.clear()


def _build_runner(
    *, thumbnail_hz: float, readers: list[_SyntheticRgbReader]
) -> tuple[DeployRunner, _NoOpSkill]:
    skill = _NoOpSkill()
    skill.configure()
    skill.activate()
    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    hal = SO100FollowerHAL(robot=twin)
    aggregator = WorldStateAggregator(hal.description)
    runner = DeployRunner(
        hal=hal,
        skill=skill,
        aggregator=aggregator,
        sensor_readers=readers,
        rate_hz=30.0,
        thumbnail_hz=thumbnail_hz,
    )
    return runner, skill


def _shutdown(skill: _NoOpSkill) -> None:
    if skill.info.state.value == "active":
        skill.deactivate()
    if skill.info.state.value != "finalized":
        skill.shutdown()


def _thumbnails_by_source(exporter: InMemorySpanExporter) -> dict[str, int]:
    counts: dict[str, int] = {}
    for span in exporter.get_finished_spans():
        if span.name != semconv.SPAN_SENSORS_READ_LATEST:
            continue
        attrs = span.attributes or {}
        if semconv.SENSORS_THUMBNAIL_JPEG_B64 in attrs:
            src = str(attrs.get(semconv.SENSORS_SOURCE, "?"))
            counts[src] = counts.get(src, 0) + 1
    return counts


class _FakeClock:
    """Deterministic monotonic clock the test advances by hand.

    Injected into ``runner._thumbnail_clock`` so the per-camera emit count is
    a function of simulated time only — not real wall-clock tick throughput,
    which varies with CPU load (the old loop-on-``time.monotonic`` form flaked
    under full-suite contention). The gate *math* is covered separately by the
    ``test_thumbnail_due_*`` cases; these exercise it through the real
    ``tick()`` path with a controllable clock.
    """

    def __init__(self, t0: float = 1000.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t


def test_gate_emits_at_thumbnail_hz(memory_exporter: InMemorySpanExporter) -> None:
    reader = _SyntheticRgbReader("cam_top")
    runner, skill = _build_runner(thumbnail_hz=5.0, readers=[reader])
    clock = _FakeClock()
    runner._thumbnail_clock = clock
    runner.activate()
    # 5 Hz gate (period 0.2 s). Tick step 0.07 s is chosen NOT to divide the
    # period, so no tick ever lands exactly on a deadline (which float fuzz
    # could then drop): emits at the first tick past each deadline →
    # t ≈ 1000.00, .21, .42, .63, .84, 1.05 = 6 over 16 ticks.
    dt = 0.07
    n_ticks = 16
    try:
        for i in range(n_ticks):
            clock.t = 1000.0 + i * dt
            runner.tick()
    finally:
        runner.deactivate()
        _shutdown(skill)

    n = _thumbnails_by_source(memory_exporter).get("cam_top", 0)
    assert n == 6, f"expected exactly 6 thumbnails at 5 Hz over 16 ticks, got {n}"


def test_gate_is_per_camera(memory_exporter: InMemorySpanExporter) -> None:
    readers = [_SyntheticRgbReader("cam_a"), _SyntheticRgbReader("cam_b")]
    runner, skill = _build_runner(thumbnail_hz=5.0, readers=readers)
    clock = _FakeClock()
    runner._thumbnail_clock = clock
    runner.activate()
    dt = 0.07
    n_ticks = 16
    try:
        for i in range(n_ticks):
            clock.t = 1000.0 + i * dt
            runner.tick()
    finally:
        runner.deactivate()
        _shutdown(skill)

    counts = _thumbnails_by_source(memory_exporter)
    assert set(counts) == {"cam_a", "cam_b"}
    # Each camera is gated independently at 5 Hz → 6 emits over the 16 ticks.
    assert all(v == 6 for v in counts.values()), counts


def test_zero_disables_thumbnails(memory_exporter: InMemorySpanExporter) -> None:
    reader = _SyntheticRgbReader("cam_top")
    runner, skill = _build_runner(thumbnail_hz=0.0, readers=[reader])
    runner.activate()
    try:
        for _ in range(60):
            runner.tick()
    finally:
        runner.deactivate()
        _shutdown(skill)

    assert _thumbnails_by_source(memory_exporter) == {}


def test_thumbnail_due_holds_rate_near_tick_rate() -> None:
    # 25 Hz target sampled by a 28 Hz tick must deliver ~25 Hz, NOT a tick
    # subharmonic (~14 Hz). Regression for the `now + period` quantisation bug:
    # advancing the deadline from `now` collapsed the rate when the tick period
    # was only slightly shorter than the gate period.
    runner, skill = _build_runner(thumbnail_hz=25.0, readers=[])
    try:
        tick_dt = 1.0 / 28.0
        t = 1000.0
        emits = 0
        n_ticks = int(4.0 * 28)  # ~4 simulated seconds
        for _ in range(n_ticks):
            if runner._thumbnail_due("cam", t):
                emits += 1
            t += tick_dt
        rate = emits / (n_ticks * tick_dt)
        assert 23.0 <= rate <= 27.0, f"expected ~25 Hz, got {rate:.1f}"
    finally:
        _shutdown(skill)


def test_thumbnail_due_no_cold_start_burst() -> None:
    # First call (no prior deadline) emits once; a tick a few ms later must NOT
    # emit — the deadline starts at `now + period`, never a zero/epoch value
    # that would burst-fire every tick to catch up.
    runner, skill = _build_runner(thumbnail_hz=5.0, readers=[])
    try:
        t = 5000.0
        assert runner._thumbnail_due("cam", t) is True
        assert runner._thumbnail_due("cam", t + 0.01) is False  # 10 ms < 200 ms
        assert runner._thumbnail_due("cam", t + 0.21) is True  # past the 200 ms period
    finally:
        _shutdown(skill)


def test_thumbnail_due_disabled_returns_false() -> None:
    runner, skill = _build_runner(thumbnail_hz=0.0, readers=[])
    try:
        assert runner._thumbnail_due("cam", 1000.0) is False
        assert runner._thumbnail_due("cam", 2000.0) is False
    finally:
        _shutdown(skill)
