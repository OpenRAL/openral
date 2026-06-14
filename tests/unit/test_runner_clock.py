"""Unit tests for :mod:`openral_runner.clock`.

Property tested: :func:`precise_sleep` waits at least the requested duration
and overshoots by less than a configurable tolerance — the cadence accuracy
the inference runner depends on. No mocks; the test exercises the real
``time.perf_counter`` clock.
"""

from __future__ import annotations

import time

import pytest
from openral_runner import precise_sleep
from openral_runner.clock import sleep_until


@pytest.mark.parametrize("duration_s", [0.0, -0.01, -1.0])
def test_precise_sleep_returns_immediately_for_non_positive_durations(
    duration_s: float,
) -> None:
    """A non-positive duration must be a no-op (no clock read, no busy-wait)."""
    t0 = time.perf_counter()
    precise_sleep(duration_s)
    elapsed = time.perf_counter() - t0
    # 2 ms is generous slack for jitter; the actual return is sub-microsecond.
    assert elapsed < 2e-3


@pytest.mark.parametrize("duration_s", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
def test_precise_sleep_meets_or_exceeds_target(duration_s: float) -> None:
    """The wait must never finish earlier than the requested duration."""
    t0 = time.perf_counter()
    precise_sleep(duration_s)
    elapsed = time.perf_counter() - t0
    assert elapsed >= duration_s, (
        f"precise_sleep returned early: requested {duration_s * 1e3:.3f} ms, "
        f"got {elapsed * 1e3:.3f} ms"
    )


@pytest.mark.parametrize("duration_s", [5e-3, 1e-2, 2e-2])
def test_precise_sleep_overshoot_is_bounded(duration_s: float) -> None:
    """The wait must overshoot the target by no more than ~3 ms on a healthy host.

    3 ms is generous slack for non-RT Linux kernels; on a dedicated CI runner
    the typical overshoot is < 200 µs. If this test starts flaking, the host
    has scheduling issues, not the helper.
    """
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        precise_sleep(duration_s)
        samples.append(time.perf_counter() - t0)
    median = sorted(samples)[len(samples) // 2]
    overshoot_ms = (median - duration_s) * 1e3
    assert overshoot_ms < 3.0, (
        f"precise_sleep median overshoot {overshoot_ms:.3f} ms exceeds 3 ms "
        f"budget for {duration_s * 1e3:.1f} ms target (samples: "
        f"{[round((s - duration_s) * 1e3, 3) for s in samples]} ms)"
    )


def test_sleep_until_waits_for_absolute_deadline() -> None:
    """``sleep_until`` honours a ``time.perf_counter``-relative deadline."""
    target = time.perf_counter() + 5e-3
    sleep_until(target)
    assert time.perf_counter() >= target


def test_sleep_until_returns_immediately_for_past_deadline() -> None:
    """A deadline already in the past must be a no-op."""
    target = time.perf_counter() - 5e-3
    t0 = time.perf_counter()
    sleep_until(target)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2e-3


def test_precise_sleep_cadence_at_30_hz() -> None:
    """Simulate the runner's 30 Hz tick — observed wall-time matches target.

    Runs 10 ticks at 30 Hz (~333.3 ms total), asserts the mean tick interval
    is within ±1 ms of 1/30 s. This is the actual contract the inference
    runner depends on.
    """
    rate_hz = 30.0
    period = 1.0 / rate_hz
    deadline = time.perf_counter()
    timestamps: list[float] = []
    for _ in range(10):
        timestamps.append(time.perf_counter())
        deadline += period
        sleep_until(deadline)
    intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    mean_interval = sum(intervals) / len(intervals)
    assert abs(mean_interval - period) < 1e-3, (
        f"30 Hz cadence drift: mean interval {mean_interval * 1e3:.3f} ms, "
        f"expected {period * 1e3:.3f} ms (intervals: "
        f"{[round(i * 1e3, 3) for i in intervals]} ms)"
    )
