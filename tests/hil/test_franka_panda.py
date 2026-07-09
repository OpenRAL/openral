"""HIL tests for the Franka Emika Panda over the FCI.

These tests require a physically connected Panda + a running ``franka_ros2``
launch file with a joint trajectory controller spun up; they must not run
in standard CI.  Gated by the ``[self-hosted, lab-franka]`` runner label.

Environment:
    FRANKA_FCI_IP: hostname or IP of the FCI (default ``172.16.0.2``).
    FRANKA_REQUIRE_HW: when set to ``"1"``, missing FCI fails instead of
        skipping (used on the lab-franka runner so a misconfigured Lab
        machine surfaces in CI).
    FRANKA_COMMAND_TOPIC: Override for the trajectory command topic
        (defaults to ``/franka_arm_controller/joint_trajectory``).
    FRANKA_STATE_TOPIC: Override for the joint-state topic
        (defaults to ``/joint_states``).

Safety rules (from CLAUDE.md §7.3):
- Each test is idempotent and time-bounded (< 120 s).
- The fixture always disconnects on teardown, even on failure.
- No test moves the arm faster than 50 % of its velocity limit.
- ``ROSEStopRequested`` is allowed to propagate so the safety supervisor
  can record the incident.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import socket
import time
from collections.abc import Iterator

import pytest
from openral_core import Action, ControlMode
from openral_core.exceptions import (
    ROSEStopRequested,  # noqa: F401  # reason: re-exported for safety
)
from openral_hal.franka_panda import FRANKA_PANDA_DESCRIPTION
from openral_hal.franka_panda_real import FrankaPandaRealHAL

FRANKA_FCI_IP = os.environ.get("FRANKA_FCI_IP", "172.16.0.2")
FRANKA_REQUIRE_HW = os.environ.get("FRANKA_REQUIRE_HW", "0") == "1"
FRANKA_COMMAND_TOPIC = os.environ.get(
    "FRANKA_COMMAND_TOPIC", "/franka_arm_controller/joint_trajectory"
)
FRANKA_STATE_TOPIC = os.environ.get("FRANKA_STATE_TOPIC", "/joint_states")


def _fci_reachable(host: str, port: int = 80, timeout: float = 0.5) -> bool:
    """TCP-probe the FCI port; the lab Panda exposes a web UI on :80."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.skipif(
        not (FRANKA_REQUIRE_HW or _fci_reachable(FRANKA_FCI_IP)),
        reason=(
            f"Franka FCI not reachable at {FRANKA_FCI_IP} (set FRANKA_REQUIRE_HW=1 to fail instead)"
        ),
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("rclpy") is None,
        reason="rclpy not installed — Franka HIL needs a live ROS 2 stack.",
    ),
]


@pytest.fixture()
def franka_hal() -> Iterator[FrankaPandaRealHAL]:
    """Connect a FrankaPandaRealHAL bridged to the live franka_ros2 driver.

    The fixture brings up a minimal ``rclpy`` node, wires its
    publish/subscribe pair into the HAL via the injected transport, and
    tears everything down in the ``finally`` branch.
    """
    from tests.hil._ros_control_transport import make_hil_transport  # reason: HIL-only import

    joint_names = [j.name for j in FRANKA_PANDA_DESCRIPTION.joints]
    _node, transport, cleanup = make_hil_transport(
        node_name="openral_hil_franka",
        joint_names=joint_names,
        command_topic=FRANKA_COMMAND_TOPIC,
        joint_state_topic=FRANKA_STATE_TOPIC,
    )
    hal = FrankaPandaRealHAL(
        fci_ip=FRANKA_FCI_IP,
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    try:
        hal.connect()
        assert transport.wait_for_first_state(deadline_s=2.0), (
            "driver published no joint state within 2 s — is franka_ros2 running?"
        )
        yield hal
    finally:
        with contextlib.suppress(Exception):
            hal.disconnect()
        with contextlib.suppress(Exception):
            cleanup()


class TestFrankaPandaHIL:
    def test_connect_and_read_state_under_200ms(self, franka_hal: FrankaPandaRealHAL) -> None:
        t0 = time.monotonic()
        state = franka_hal.read_state()
        elapsed_ms = (time.monotonic() - t0) * 1e3
        assert state.name == [j.name for j in FRANKA_PANDA_DESCRIPTION.joints]
        assert elapsed_ms < 200, f"read_state took {elapsed_ms:.1f} ms (limit 200 ms)"

    def test_disconnect_is_idempotent(self, franka_hal: FrankaPandaRealHAL) -> None:
        franka_hal.disconnect()
        # Second call must not raise.
        franka_hal.disconnect()

    def test_send_hold_action(self, franka_hal: FrankaPandaRealHAL) -> None:
        """Send a hold-in-place action; arm must not drift more than 1 deg / 1 mm."""
        state = franka_hal.read_state()
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[list(state.position)],
            stamp_ns=time.time_ns(),
        )
        franka_hal.send_action(action)
        time.sleep(0.2)
        state_after = franka_hal.read_state()
        for before, after in zip(state.position, state_after.position, strict=True):
            tol = 0.02  # 0.02 rad ≈ 1.1° (gripper is normalised so 0.02 is also fine)
            assert abs(after - before) < tol, (
                f"Joint moved {abs(after - before):.4f} during hold "
                f"(before={before:.3f}, after={after:.3f})"
            )
