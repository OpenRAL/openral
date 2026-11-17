"""``/scan`` from an observation-fed backend (Isaac) leaves the HAL executor thread.

The HAL's proprio publisher thread publishes the scan the step's observation carries
(``ProprioFrame.scan``), stamped like ``/odom`` and scheduled in sim time by
``next_scan_due``. On the executor thread, which env.step and render hold, a 10 Hz timer
delivered 1.5-3.3 Hz, past Nav2's collision-monitor source timeout (Spark, 2026-11-15).
The bridge end of it (gate, publish, warnings) is in ``test_isaac_sim_sidecar_wire.py``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from openral_core import JointState, RobotDescription
from openral_hal.proprio_snapshot import ProprioFrame
from openral_hal.sim_sensor_bridge import next_scan_due

_PERIOD_10HZ = 100_000_000


def test_the_first_observation_is_published() -> None:
    assert next_scan_due(None, 0, 10.0) == _PERIOD_10HZ


@pytest.mark.parametrize(
    ("next_due", "now", "expect"),
    [
        (_PERIOD_10HZ, _PERIOD_10HZ // 2, None),  # inside one period
        (_PERIOD_10HZ, _PERIOD_10HZ, 2 * _PERIOD_10HZ),  # on time
        (_PERIOD_10HZ, 0, None),  # the same observation again (no new step)
        (_PERIOD_10HZ, 5 * _PERIOD_10HZ, 6 * _PERIOD_10HZ),  # after a pause: resync
        (10 * _PERIOD_10HZ, _PERIOD_10HZ, 2 * _PERIOD_10HZ),  # sim time went back: resync
    ],
)
def test_scans_follow_sim_time(next_due: int, now: int, expect: int | None) -> None:
    assert next_scan_due(next_due, now, 10.0) == expect


def _published(step_ns: int, steps: int, rate_hz: float) -> list[int]:
    next_due: int | None = None
    sent = []
    for k in range(steps):
        due = next_scan_due(next_due, k * step_ns, rate_hz)
        if due is not None:
            sent.append(k)
            next_due = due
    return sent


def test_a_slow_sim_publishes_every_step_that_is_a_period_apart() -> None:
    """At 30 Hz control a 10 Hz scan goes out every third step, whatever the wall rate."""
    assert _published(1_000_000_000 // 30, 10, 10.0) == [0, 3, 6, 9]


def test_steps_that_do_not_divide_the_period_still_average_the_rate() -> None:
    """12 Hz steps, 10 Hz scans: resetting the phase at each scan gave every other step
    (6 Hz). Keeping the schedule gives 10 scans a second."""
    sent = _published(1_000_000_000 // 12, 12 * 10, 10.0)
    assert len(sent) in (100, 101)
    assert all(b - a in (1, 2) for a, b in itertools.pairwise(sent))


def test_a_frame_carries_no_scan_unless_the_backend_returned_one() -> None:
    robot = RobotDescription.from_yaml(
        str(Path(__file__).resolve().parents[2] / "robots/openamrobot_v2/robot.yaml")
    )
    names = [j.name for j in robot.joints]
    state = JointState(name=names, position=[0.0] * len(names), stamp_ns=1)
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
        scan=np.asarray([1.0, 2.0], dtype=np.float32),
    )
    assert with_scan.scan is not None
    assert with_scan.scan.tolist() == [1.0, 2.0]
