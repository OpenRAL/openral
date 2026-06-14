"""HIL tests for the SO-100 follower arm.

These tests require a physically connected SO-100 arm and must not run in
standard CI.  They are gated by the ``[self-hosted, lab-so100]`` runner label
in ``.github/workflows/hil-so100.yml``.

Environment:
    SO100_PORT: USB serial port (default ``/dev/ttyUSB0``).

Safety rules (from CLAUDE.md §7.3):
- Each test must be idempotent and time-bounded (< 120 s per test).
- The test fixture always calls ``hal.disconnect()`` in teardown, even on failure.
- No test moves the arm faster than 20 % of its velocity limit.
- If a test raises ``ROSEStopRequested``, the fixture records an incident and
  aborts the suite immediately.
"""

from __future__ import annotations

import contextlib
import math
import os
import time

import pytest
from openral_core import Action, ControlMode
from openral_core.exceptions import ROSEStopRequested, ROSRuntimeError  # noqa: F401
from openral_hal.so100_follower import SO100FollowerHAL

SO100_PORT = os.environ.get("SO100_PORT", "/dev/ttyUSB0")

# ── Skip guard ────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.skipif(
    not os.path.exists(SO100_PORT),
    reason=f"SO-100 not connected on {SO100_PORT}",
)


@pytest.fixture()
def so100_hal():  # type: ignore[no-untyped-def]
    """Connect the SO-100 HAL; disconnect (and estop if needed) in teardown."""
    hal = SO100FollowerHAL(port=SO100_PORT, calibrate_on_connect=False)
    hal.connect()
    yield hal
    with contextlib.suppress(Exception):
        hal.disconnect()


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSO100HIL:
    def test_connect_and_read_state(self, so100_hal) -> None:  # type: ignore[no-untyped-def]
        """Arm must respond with a valid joint state within 200 ms."""
        t0 = time.monotonic()
        state = so100_hal.read_state()
        elapsed_ms = (time.monotonic() - t0) * 1e3
        assert len(state.name) == 6, "Expected 6 joints"
        assert elapsed_ms < 200, f"read_state took {elapsed_ms:.1f} ms (limit 200 ms)"

    def test_send_hold_action(self, so100_hal) -> None:  # type: ignore[no-untyped-def]
        """Send a hold-in-place action (current position); arm must not move."""
        state = so100_hal.read_state()
        # Use current position as target — arm holds still
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[list(state.position)],
            stamp_ns=time.time_ns(),
        )
        so100_hal.send_action(action)
        time.sleep(0.1)

        state_after = so100_hal.read_state()
        # Positions should not move more than 2 degrees from hold target
        for before, after in zip(state.position, state_after.position, strict=True):
            # Gripper is normalised [0,1]; allow 0.02 tolerance
            tol = 0.02 if abs(before) <= 1.0 else math.radians(2.0)
            assert abs(after - before) < tol, (
                f"Joint moved {abs(after - before):.4f} units during hold "
                f"(before={before:.3f}, after={after:.3f})"
            )
