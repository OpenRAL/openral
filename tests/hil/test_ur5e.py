"""HIL tests for the Universal Robots UR5e via ``ur_robot_driver``.

These tests require a physically connected UR5e arm running the URCap
``external_control`` program plus the ``ur_robot_driver`` ROS 2 node
(``ros2_control`` controller manager).  They must not run in standard CI
and are gated by the ``[self-hosted, lab-ur5e]`` runner label.

Environment:
    UR5E_HOST: Static IP of the UR5e controller (e.g. ``"192.168.1.42"``).
        The test only checks the variable is set; bringing up the driver
        with this IP is the lab runner's responsibility.
    UR5E_COMMAND_TOPIC: Override for the trajectory command topic
        (defaults to ``/scaled_joint_trajectory_controller/joint_trajectory``).
    UR5E_STATE_TOPIC: Override for the joint-state topic
        (defaults to ``/joint_states``).

Safety rules (CLAUDE.md §7.3, §7.7):
- Each test is idempotent and time-bounded (< 30 s per test).
- The fixture always calls ``hal.disconnect()`` in teardown, even on failure.
- Tests never command motion — only ``read_state``, hold-in-place, and
  E-stop are exercised.
- ``ROSEStopRequested`` raised by ``estop()`` is allowed to bubble up; per
  CLAUDE.md §10 it is **never** silently caught.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import time
from typing import TYPE_CHECKING

import pytest
from openral_core import Action, ControlMode
from openral_core.exceptions import ROSEStopRequested, ROSPerceptionStale, ROSRuntimeError
from openral_hal.ur_real import UR5e_REAL_DESCRIPTION, UR5eRealHAL

if TYPE_CHECKING:
    from collections.abc import Iterator

UR5E_HOST = os.environ.get("UR5E_HOST", "")
UR5E_COMMAND_TOPIC = os.environ.get(
    "UR5E_COMMAND_TOPIC", "/scaled_joint_trajectory_controller/joint_trajectory"
)
UR5E_STATE_TOPIC = os.environ.get("UR5E_STATE_TOPIC", "/joint_states")

# ── Skip guards ──────────────────────────────────────────────────────────────

pytestmark = [
    pytest.mark.skipif(
        not UR5E_HOST,
        reason="UR5E_HOST not set — no live UR5e controller available.",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("rclpy") is None,
        reason="rclpy not installed — UR HIL needs a live ROS 2 stack.",
    ),
]


@pytest.fixture()
def ur5e_hal() -> Iterator[UR5eRealHAL]:
    """Connect a UR5eRealHAL bridged to the live ros2_control driver.

    The fixture brings up a minimal ``rclpy`` node, wires its
    publish/subscribe pair into the HAL via the injected transport, and
    tears everything down in the ``finally`` branch — no exceptions are
    swallowed except a benign double-shutdown of ``rclpy``.
    """
    from tests.hil._ros_control_transport import make_hil_transport  # reason: HIL-only import

    joint_names = [j.name for j in UR5e_REAL_DESCRIPTION.joints]
    _node, transport, cleanup = make_hil_transport(
        node_name="openral_hil_ur5e",
        joint_names=joint_names,
        command_topic=UR5E_COMMAND_TOPIC,
        joint_state_topic=UR5E_STATE_TOPIC,
    )
    hal = UR5eRealHAL(
        robot_ip=UR5E_HOST,
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    try:
        hal.connect()
        # Wait up to 2 s for the driver's first /joint_states message.
        assert transport.wait_for_first_state(deadline_s=2.0), (
            "driver published no joint state within 2 s — is ur_robot_driver running?"
        )
        yield hal
    finally:
        with contextlib.suppress(Exception):
            hal.disconnect()
        with contextlib.suppress(Exception):
            cleanup()


# ── Tests ────────────────────────────────────────────────────────────────────


class TestUR5eHIL:
    def test_hal_advertises_real_hal(self, ur5e_hal: UR5eRealHAL) -> None:
        """Sanity check — the HAL exposes the real-HW manifest, not the sim one."""
        assert ur5e_hal.description.sdk_kind == "closed"
        assert ur5e_hal.description.hal.real == "openral_hal.ur_real:UR5eRealHAL"
        assert ur5e_hal.description.name == "ur5e"
        assert len(ur5e_hal.description.joints) == 6

    def test_read_state_under_200ms(self, ur5e_hal: UR5eRealHAL) -> None:
        """``read_state`` must return within the 200 ms control-cycle budget."""
        t0 = time.monotonic()
        state = ur5e_hal.read_state()
        elapsed_ms = (time.monotonic() - t0) * 1e3
        assert state.name == [j.name for j in ur5e_hal.description.joints]
        assert len(state.position) == 6
        assert elapsed_ms < 200, f"read_state took {elapsed_ms:.1f} ms (limit 200 ms)"

    def test_hold_in_place_no_motion(self, ur5e_hal: UR5eRealHAL) -> None:
        """Sending a hold-in-place trajectory must not move the arm > 1°/joint."""
        before = ur5e_hal.read_state()
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[list(before.position)],
            stamp_ns=time.time_ns(),
        )
        ur5e_hal.send_action(action)
        time.sleep(0.2)
        after = ur5e_hal.read_state()
        for b, a in zip(before.position, after.position, strict=True):
            assert abs(a - b) < 0.0175, (  # 1° in radians
                f"joint moved {abs(a - b):.4f} rad during hold (before={b:.4f}, after={a:.4f})"
            )

    def test_disconnect_is_idempotent(self, ur5e_hal: UR5eRealHAL) -> None:
        """Per HAL Protocol — calling ``disconnect()`` twice must not raise."""
        ur5e_hal.disconnect()
        ur5e_hal.disconnect()
        # After disconnect, read_state must fail-fast (per the Protocol).
        with pytest.raises(ROSRuntimeError):
            ur5e_hal.read_state()

    def test_deadman_topic_advertised(self, ur5e_hal: UR5eRealHAL) -> None:
        """The HAL records the deadman / safety-mode topic the supervisor watches.

        Per CLAUDE.md §7.7 the safety supervisor (a separate process) is what
        actually subscribes; the HAL is the source of truth for the topic
        name so the supervisor's launch file can pick it up.
        """
        assert ur5e_hal.deadman_topic == "/io_and_status_controller/safety_mode"

    def test_estop_raises_estoprequested(self, ur5e_hal: UR5eRealHAL) -> None:
        """``estop`` must always raise ``ROSEStopRequested``.

        Per CLAUDE.md §10 / HAL Protocol docstring — this exception is
        **never** silently caught.  The fixture's teardown calls
        ``disconnect()`` after the test, which is idempotent.
        """
        with pytest.raises(ROSEStopRequested):
            ur5e_hal.estop()

    def test_perception_stale_after_long_silence(self, ur5e_hal: UR5eRealHAL) -> None:
        """If no joint-state message arrives for > staleness_limit_s, raise.

        The test artificially ages the cached timestamp (driver is still
        running) to confirm the staleness latch fires; we do **not** kill the
        driver because the HIL runner cannot recover from a driver crash
        between tests without manual intervention.
        """
        ur5e_hal._last_state_time -= ur5e_hal._staleness_limit_s + 0.1
        with pytest.raises(ROSPerceptionStale):
            ur5e_hal.read_state()
