"""HIL tests for the Rethink Sawyer over the sawyer_robot ROS 2 driver.

These tests require a physically connected Sawyer + a running
``sawyer_robot`` bring-up launch (with a joint trajectory controller); they
must not run in standard CI.  Gated by the ``[self-hosted, lab-sawyer]``
runner label.

Environment:
    SAWYER_HOSTNAME: hostname or IP of the Sawyer's onboard PC (default
        ``sawyer.local``).
    SAWYER_REQUIRE_HW: when set to ``"1"``, missing hardware fails instead
        of skipping.
    SAWYER_COMMAND_TOPIC: Override for the trajectory command topic
        (defaults to ``/sawyer_arm_controller/joint_trajectory``).
    SAWYER_STATE_TOPIC: Override for the joint-state topic
        (defaults to ``/robot/joint_states`` — Rethink intera_sdk lineage).

Safety rules (from CLAUDE.md §7.3) mirror tests/hil/test_franka_panda.py.
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
    ROSEStopRequested,  # reason: re-exported for safety
)
from openral_hal.sawyer_real import SAWYER_DESCRIPTION, SawyerRealHAL

SAWYER_HOSTNAME = os.environ.get("SAWYER_HOSTNAME", "sawyer.local")
SAWYER_REQUIRE_HW = os.environ.get("SAWYER_REQUIRE_HW", "0") == "1"
SAWYER_COMMAND_TOPIC = os.environ.get(
    "SAWYER_COMMAND_TOPIC", "/sawyer_arm_controller/joint_trajectory"
)
SAWYER_STATE_TOPIC = os.environ.get("SAWYER_STATE_TOPIC", "/robot/joint_states")


def _hostname_reachable(host: str, port: int = 22, timeout: float = 0.5) -> bool:
    """TCP-probe Sawyer's SSH port; the onboard PC exposes :22."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.skipif(
        not (SAWYER_REQUIRE_HW or _hostname_reachable(SAWYER_HOSTNAME)),
        reason=(
            f"Sawyer not reachable at {SAWYER_HOSTNAME} (set SAWYER_REQUIRE_HW=1 to fail instead)"
        ),
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("rclpy") is None,
        reason="rclpy not installed — Sawyer HIL needs a live ROS 2 stack.",
    ),
]


@pytest.fixture()
def sawyer_hal() -> Iterator[SawyerRealHAL]:
    """Connect a SawyerRealHAL bridged to the live sawyer_robot driver."""
    from tests.hil._ros_control_transport import make_hil_transport  # reason: HIL-only import

    joint_names = [j.name for j in SAWYER_DESCRIPTION.joints]
    _node, transport, cleanup = make_hil_transport(
        node_name="openral_hil_sawyer",
        joint_names=joint_names,
        command_topic=SAWYER_COMMAND_TOPIC,
        joint_state_topic=SAWYER_STATE_TOPIC,
    )
    hal = SawyerRealHAL(
        hostname=SAWYER_HOSTNAME,
        publish_fn=transport.publish,
        state_fn=transport.state,
    )
    try:
        hal.connect()
        assert transport.wait_for_first_state(deadline_s=2.0), (
            "driver published no joint state within 2 s — is sawyer_robot running?"
        )
        yield hal
    finally:
        with contextlib.suppress(Exception):
            hal.disconnect()
        with contextlib.suppress(Exception):
            cleanup()


class TestSawyerHIL:
    def test_connect_and_read_state_under_200ms(self, sawyer_hal: SawyerRealHAL) -> None:
        t0 = time.monotonic()
        state = sawyer_hal.read_state()
        elapsed_ms = (time.monotonic() - t0) * 1e3
        assert state.name == [j.name for j in SAWYER_DESCRIPTION.joints]
        assert elapsed_ms < 200, f"read_state took {elapsed_ms:.1f} ms (limit 200 ms)"

    def test_disconnect_is_idempotent(self, sawyer_hal: SawyerRealHAL) -> None:
        sawyer_hal.disconnect()
        sawyer_hal.disconnect()

    def test_send_hold_action(self, sawyer_hal: SawyerRealHAL) -> None:
        state = sawyer_hal.read_state()
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[list(state.position)],
            stamp_ns=time.time_ns(),
        )
        sawyer_hal.send_action(action)
        time.sleep(0.2)
        state_after = sawyer_hal.read_state()
        for before, after in zip(state.position, state_after.position, strict=True):
            assert abs(after - before) < 0.02, (
                f"Sawyer joint moved {abs(after - before):.4f} rad during hold"
            )


# ── Downstream e-stop (issue #295) ───────────────────────────────────────────
# Last in file on purpose: it stops the vendor controller and leaves the robot
# needing an operator restart. Run attended, with the physical E-stop in reach.


def _production_stop_seam(node_name: str, hal: object) -> tuple[object, object]:
    """Attach the production ``RosControlTransport`` as the HAL's stop seam.

    The HIL bridge above drives commands; the stop goes through the same
    transport a real deploy uses, so what is exercised here is the production
    ``controller_manager`` switch + confirmation, not a test-side re-creation.
    """
    import rclpy
    from openral_hal.ros_control_transport import RosControlTransport

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node(node_name)
    transport = RosControlTransport(
        node,
        command_topics=list(hal.command_bindings()),  # type: ignore[attr-defined]  # reason: RosControlDrivable
        joint_names=hal.ros2_control_joint_names(),  # type: ignore[attr-defined]
        joint_state_topic=hal.joint_state_topic,  # type: ignore[attr-defined]
        command_kinds=hal.command_bindings(),  # type: ignore[attr-defined]
        controller_names=hal.controller_names(),  # type: ignore[attr-defined]  # reason: ControllerStoppable
        trigger_services=hal.vendor_stop_services(),  # type: ignore[attr-defined]
        empty_topics=hal.vendor_stop_topics(),  # type: ignore[attr-defined]
    )
    hal.attach_controller_stop(transport)  # type: ignore[attr-defined]
    return node, transport


class TestSawyerDownstreamEStop:
    def test_zz_estop_deactivates_the_controller_and_super_stops(
        self, sawyer_hal: SawyerRealHAL
    ) -> None:
        """``/openral/estop`` must deactivate ``sawyer_arm_controller`` and super-stop the robot.

        The controller half is acknowledged by ``controller_manager``; the
        intera half is observed on ``/robot/state`` (``stopped == true``) when
        ``intera_core_msgs`` is installed. Recovery is ``RESTART_REQUIRED``:
        publish ``/robot/set_super_reset``, re-enable, relaunch the HAL node.
        """
        import importlib.util
        import time

        from openral_hal.protocol import EStopRecovery

        node, transport = _production_stop_seam("openral_hil_sawyer_stop", sawyer_hal)
        try:
            with pytest.raises(ROSEStopRequested, match="downstream stop acknowledged"):
                sawyer_hal.estop()
            report = sawyer_hal.last_stop_report
            assert report is not None and report.stopped, report
            assert report.vendor_stop == "/robot/set_super_stop"
            states = transport.controller_states(timeout_s=5.0)  # type: ignore[attr-defined]
            assert states.get(sawyer_hal.controller_name) == "inactive", states
            assert sawyer_hal.estop_recovery is EStopRecovery.RESTART_REQUIRED
            if importlib.util.find_spec("intera_core_msgs") is not None:
                import rclpy
                from intera_core_msgs.msg import RobotAssemblyState

                seen: list[bool] = []
                node.create_subscription(  # type: ignore[attr-defined]
                    RobotAssemblyState, "/robot/state", lambda m: seen.append(bool(m.stopped)), 10
                )
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and not (seen and seen[-1]):
                    rclpy.spin_once(node, timeout_sec=0.1)
                assert seen and seen[-1], "intera never reported stopped=true on /robot/state"
        finally:
            transport.close()  # type: ignore[attr-defined]
            node.destroy_node()  # type: ignore[attr-defined]
