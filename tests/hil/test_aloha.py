"""HIL tests for the Trossen ALOHA bimanual setup.

These tests require two physical Trossen ViperX 300 arms wired through the
Interbotix XS SDK + a running ``interbotix_xsarm_control`` ROS 2 launch
(bringing up four ros2_control controllers per CLAUDE.md / ADR-0007).
They must not run in standard CI.  Gated by the ``[self-hosted, lab-aloha]``
runner label.

Environment:
    ALOHA_LEFT_PORT: USB serial port of the left Interbotix arm (default
        ``/dev/ttyDXL_left``).
    ALOHA_RIGHT_PORT: USB serial port of the right arm (default
        ``/dev/ttyDXL_right``).
    ALOHA_REQUIRE_HW: when set to ``"1"``, missing hardware fails instead
        of skipping.
    ALOHA_LEFT_ARM_TOPIC / ALOHA_RIGHT_ARM_TOPIC: Override for the
        per-arm ``JointTrajectory`` command topic
        (defaults to ``/left_arm/arm_controller/joint_trajectory`` and
        ``/right_arm/arm_controller/joint_trajectory``).
    ALOHA_LEFT_GRIPPER_TOPIC / ALOHA_RIGHT_GRIPPER_TOPIC: Override for the
        per-arm gripper command topic (defaults to
        ``/left_arm/gripper_controller/command`` and
        ``/right_arm/gripper_controller/command``).
    ALOHA_STATE_TOPIC: Override for the aggregated joint-state topic
        (defaults to ``/joint_states``).

Safety rules (from CLAUDE.md §7.3) mirror tests/hil/test_so100.py.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import time
from collections.abc import Iterator

import pytest
from openral_core import Action, ControlMode
from openral_core.exceptions import (
    ROSEStopRequested,  # noqa: F401  # reason: re-exported for safety
)
from openral_hal.aloha import ALOHA_DESCRIPTION, AlohaHAL

ALOHA_LEFT_PORT = os.environ.get("ALOHA_LEFT_PORT", "/dev/ttyDXL_left")
ALOHA_RIGHT_PORT = os.environ.get("ALOHA_RIGHT_PORT", "/dev/ttyDXL_right")
ALOHA_REQUIRE_HW = os.environ.get("ALOHA_REQUIRE_HW", "0") == "1"
ALOHA_LEFT_ARM_TOPIC = os.environ.get(
    "ALOHA_LEFT_ARM_TOPIC", "/left_arm/arm_controller/joint_trajectory"
)
ALOHA_RIGHT_ARM_TOPIC = os.environ.get(
    "ALOHA_RIGHT_ARM_TOPIC", "/right_arm/arm_controller/joint_trajectory"
)
ALOHA_LEFT_GRIPPER_TOPIC = os.environ.get(
    "ALOHA_LEFT_GRIPPER_TOPIC", "/left_arm/gripper_controller/command"
)
ALOHA_RIGHT_GRIPPER_TOPIC = os.environ.get(
    "ALOHA_RIGHT_GRIPPER_TOPIC", "/right_arm/gripper_controller/command"
)
ALOHA_STATE_TOPIC = os.environ.get("ALOHA_STATE_TOPIC", "/joint_states")


def _both_ports_present() -> bool:
    return os.path.exists(ALOHA_LEFT_PORT) and os.path.exists(ALOHA_RIGHT_PORT)


pytestmark = [
    pytest.mark.skipif(
        not (ALOHA_REQUIRE_HW or _both_ports_present()),
        reason=(
            f"ALOHA arms not connected (left={ALOHA_LEFT_PORT}, right={ALOHA_RIGHT_PORT}); "
            "set ALOHA_REQUIRE_HW=1 to fail instead"
        ),
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("rclpy") is None,
        reason="rclpy not installed — ALOHA HIL needs a live ROS 2 stack.",
    ),
]


@pytest.fixture()
def aloha_hal() -> Iterator[AlohaHAL]:
    """Connect an AlohaHAL bridged to the live Interbotix XS controllers."""
    from tests.hil._aloha_ros_transport import make_aloha_hil_transport  # reason: HIL-only

    _node, transport, cleanup = make_aloha_hil_transport(
        node_name="openral_hil_aloha",
        left_arm_command_topic=ALOHA_LEFT_ARM_TOPIC,
        right_arm_command_topic=ALOHA_RIGHT_ARM_TOPIC,
        left_gripper_command_topic=ALOHA_LEFT_GRIPPER_TOPIC,
        right_gripper_command_topic=ALOHA_RIGHT_GRIPPER_TOPIC,
        joint_state_topic=ALOHA_STATE_TOPIC,
    )
    hal = AlohaHAL(
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    try:
        hal.connect()
        assert transport.wait_for_first_state(deadline_s=2.0), (
            "Interbotix XS published no joint state within 2 s — "
            "is interbotix_xsarm_control running?"
        )
        yield hal
    finally:
        with contextlib.suppress(Exception):
            hal.disconnect()
        with contextlib.suppress(Exception):
            cleanup()


class TestAlohaHIL:
    def test_connect_and_read_state_under_200ms(self, aloha_hal: AlohaHAL) -> None:
        t0 = time.monotonic()
        state = aloha_hal.read_state()
        elapsed_ms = (time.monotonic() - t0) * 1e3
        assert state.name == [j.name for j in ALOHA_DESCRIPTION.joints]
        assert len(state.position) == 14
        assert elapsed_ms < 200, f"read_state took {elapsed_ms:.1f} ms (limit 200 ms)"

    def test_disconnect_is_idempotent(self, aloha_hal: AlohaHAL) -> None:
        aloha_hal.disconnect()
        aloha_hal.disconnect()

    def test_send_hold_action(self, aloha_hal: AlohaHAL) -> None:
        state = aloha_hal.read_state()
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[list(state.position)],
            stamp_ns=time.time_ns(),
        )
        aloha_hal.send_action(action)
        time.sleep(0.2)
        state_after = aloha_hal.read_state()
        for before, after in zip(state.position, state_after.position, strict=True):
            assert abs(after - before) < 0.02, (
                f"ALOHA joint moved {abs(after - before):.4f} during hold"
            )
