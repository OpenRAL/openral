"""``/scan`` from an observation-fed backend (Isaac) leaves the HAL executor thread.

The HAL's proprio publisher thread publishes the scan the step's observation carries
(``ProprioFrame.scan``), stamped with that step's sim time and throttled in sim time by
``scan_due``. On the executor thread, which env.step and render hold, a 10 Hz timer
delivered 1.5-3.3 Hz, past Nav2's collision-monitor source timeout (Spark, 2026-11-15).
"""

from __future__ import annotations

import pytest
from openral_core import JointState
from openral_hal.proprio_snapshot import ProprioFrame
from openral_hal.sim_sensor_bridge import scan_due

_PERIOD_10HZ = 100_000_000


def test_the_first_observation_is_published() -> None:
    assert scan_due(None, 0, 10.0)


@pytest.mark.parametrize(
    ("last", "now", "due"),
    [
        (0, _PERIOD_10HZ // 2, False),  # inside one period
        (0, _PERIOD_10HZ, True),  # a full period of sim time later
        (_PERIOD_10HZ, _PERIOD_10HZ, False),  # the same observation again (no new step)
        (2 * _PERIOD_10HZ, _PERIOD_10HZ, False),  # sim time went back (a reset): wait
    ],
)
def test_scans_follow_sim_time_never_republishing_an_observation(
    last: int, now: int, due: bool
) -> None:
    assert scan_due(last, now, 10.0) is due


def test_a_slow_sim_publishes_every_step_that_is_a_period_apart() -> None:
    """At 30 Hz control a 10 Hz scan goes out every third step, whatever the wall rate."""
    step = 1_000_000_000 // 30
    last: int | None = None
    sent = []
    for k in range(10):
        if scan_due(last, k * step, 10.0):
            sent.append(k)
            last = k * step
    assert sent == [0, 3, 6, 9]


def test_a_frame_carries_no_scan_unless_the_backend_returned_one() -> None:
    state = JointState(name=["j"], position=[0.0], stamp_ns=1)
    frame = ProprioFrame(
        state=state, base_pose=(0.0, 0.0, 0.0), base_pose_6dof=None, base_twist=(0.0,) * 6
    )
    assert frame.scan is None
    with_scan = ProprioFrame(
        state=state,
        base_pose=(0.0, 0.0, 0.0),
        base_pose_6dof=None,
        base_twist=(0.0,) * 6,
        sim_time_ns=5,
        scan=(1.0, 2.0),
    )
    assert with_scan.scan == (1.0, 2.0)
