"""HIL tests for the Trossen ALOHA bimanual setup.

These tests require two physical Trossen ViperX 300 arms wired through the
Interbotix XS SDK + a running ``interbotix_xsarm_control`` ROS 2 launch.
They must not run in standard CI.  Gated by the ``[self-hosted, lab-aloha]``
runner label — no such runner is registered and there is no physical ALOHA
rig, so this file has never run against hardware.

Everything goes through the production ``InterbotixXSTransport``, the same
one the lifecycle node attaches under ``hal_mode:=real``; it speaks the
``xs_sdk`` contract ``AlohaHAL`` emits (issue #250):
``/<ns>/commands/joint_group`` (arm), ``/<ns>/commands/joint_single``
(gripper) and per-namespace ``/<ns>/joint_states``. The namespaces, motor
names and gripper range are stock-Interbotix defaults, not yet read off a
cell — confirm them before the first powered run.

Environment:
    ALOHA_LEFT_PORT: USB serial port of the left Interbotix arm (default
        ``/dev/ttyDXL_left``).
    ALOHA_RIGHT_PORT: USB serial port of the right arm (default
        ``/dev/ttyDXL_right``).
    ALOHA_REQUIRE_HW: when set to ``"1"``, missing hardware fails instead
        of skipping.
    ALOHA_LEFT_NS / ALOHA_RIGHT_NS: ``xs_sdk`` follower namespaces
        (defaults ``follower_left`` / ``follower_right``).

Safety rules mirror the retained real-arm HIL fixtures: every action is
bounded, idempotent teardown is mandatory, and E-stop remains reachable.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import time
from collections.abc import Iterator
from typing import Any

import pytest
from openral_core import Action, ControlMode
from openral_core.exceptions import (
    ROSEStopRequested,  # reason: re-exported for safety
)
from openral_hal.aloha import ALOHA_DESCRIPTION, AlohaHAL

ALOHA_LEFT_PORT = os.environ.get("ALOHA_LEFT_PORT", "/dev/ttyDXL_left")
ALOHA_RIGHT_PORT = os.environ.get("ALOHA_RIGHT_PORT", "/dev/ttyDXL_right")
ALOHA_REQUIRE_HW = os.environ.get("ALOHA_REQUIRE_HW", "0") == "1"
ALOHA_NAMESPACES = (
    os.environ.get("ALOHA_LEFT_NS", "follower_left"),
    os.environ.get("ALOHA_RIGHT_NS", "follower_right"),
)


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
    pytest.mark.skipif(
        importlib.util.find_spec("interbotix_xs_msgs") is None,
        reason="interbotix_xs_msgs not installed — ALOHA HIL needs interbotix_ros_xseries.",
    ),
]


class _Rig:
    """The connected HAL, its production transport, and the node that spins it."""

    def __init__(
        self, hal: AlohaHAL, transport: Any, node: Any
    ) -> None:  # reason: rclpy types are HIL-only imports
        self.hal, self.transport, self.node = hal, transport, node

    def spin_for(self, seconds: float) -> None:
        """Pump callbacks for a wall-clock window (a plain sleep would freeze the cache)."""
        import rclpy

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.01)


@pytest.fixture()
def aloha_rig(request: pytest.FixtureRequest) -> Iterator[_Rig]:
    """Connect an AlohaHAL over the production ``InterbotixXSTransport``.

    Wired exactly as the lifecycle node does (``attach_transport`` +
    ``attach_torque_stop`` before ``connect``), so the first ``send_action``
    reads both arms' operating modes back; and the fixture refuses to yield
    until all 14 joints have reported — the cache zero-fills a silent arm,
    and a hold built from those zeros would drive it to the zero pose at the
    stock ``profile_velocity: 0`` (unlimited).

    Teardown e-stops (torque off on both arms) when the test failed, so a
    test that dies mid-motion does not leave the arms driving to its last
    target; a passing test ends in a hold and keeps torque, since torque off
    drops the unbraked arms. Run attended, over a clear bench.
    """
    import rclpy
    from openral_hal.interbotix_transport import InterbotixXSTransport

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("openral_hil_aloha")
    hal = AlohaHAL(arm_namespaces=ALOHA_NAMESPACES)
    transport = InterbotixXSTransport(
        node,
        arm_namespaces=hal.arm_namespaces(),
        joint_names=hal.joint_names(),
        gripper_name=hal.gripper_name,
        gripper_position_range=hal.gripper_position_range,
    )
    hal.attach_transport(transport.publish, transport.state, transport.last_arrival)
    hal.attach_torque_stop(transport)
    rig = _Rig(hal, transport, node)
    failed_before = request.session.testsfailed
    try:
        hal.connect()
        deadline = time.monotonic() + 5.0
        while transport.missing_joints() and time.monotonic() < deadline:
            rig.spin_for(0.05)
        assert not transport.missing_joints(), (
            f"Interbotix XS never reported {transport.missing_joints()} within 5 s — "
            "is interbotix_xsarm_control running for both follower arms?"
        )
        yield rig
    finally:
        if request.session.testsfailed > failed_before:
            with contextlib.suppress(ROSEStopRequested):
                hal.estop()
        with contextlib.suppress(Exception):
            hal.disconnect()
        with contextlib.suppress(Exception):
            transport.close()
        with contextlib.suppress(Exception):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


@pytest.fixture()
def aloha_hal(aloha_rig: _Rig) -> AlohaHAL:
    return aloha_rig.hal


class TestAlohaHIL:
    def test_connect_and_read_state_under_200ms(self, aloha_rig: _Rig) -> None:
        aloha_hal = aloha_rig.hal
        aloha_rig.spin_for(0.05)  # freshness is the stream's, so keep it flowing
        t0 = time.monotonic()
        state = aloha_hal.read_state()
        elapsed_ms = (time.monotonic() - t0) * 1e3
        assert state.name == [j.name for j in ALOHA_DESCRIPTION.joints]
        assert len(state.position) == 14
        assert elapsed_ms < 200, f"read_state took {elapsed_ms:.1f} ms (limit 200 ms)"

    def test_disconnect_is_idempotent(self, aloha_hal: AlohaHAL) -> None:
        aloha_hal.disconnect()
        aloha_hal.disconnect()

    def test_send_hold_action(self, aloha_rig: _Rig) -> None:
        aloha_hal = aloha_rig.hal
        aloha_rig.spin_for(0.05)
        state = aloha_hal.read_state()
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[list(state.position)],
            stamp_ns=time.time_ns(),
        )
        aloha_hal.send_action(action)
        # Spin, not sleep: without callbacks the cache never changes and "after"
        # would equal "before" whatever the arm did.
        aloha_rig.spin_for(0.2)
        state_after = aloha_hal.read_state()
        # Arms only: the gripper is read as finger position but commanded as
        # motor angle (upstream's convention), so a "hold" from its read-back
        # is not exactly where it is — the linkage between them is nonlinear.
        for i, (before, after) in enumerate(zip(state.position, state_after.position, strict=True)):
            if i in (6, 13):
                continue
            assert abs(after - before) < 0.02, (
                f"ALOHA joint {state.name[i]} moved {abs(after - before):.4f} during hold"
            )


# ── Downstream e-stop (issue #295) ───────────────────────────────────────────
# Last in file on purpose: torque off leaves both ViperX arms limp, so they will
# settle under gravity. Run attended, with the arms over a clear bench.


class TestAlohaDownstreamEStop:
    def test_zz_estop_cuts_torque_on_both_arms(self, aloha_hal: AlohaHAL) -> None:
        """``/openral/estop`` must call ``torque_enable(false)`` on every arm and be acknowledged.

        Goes through the production ``InterbotixXSTransport`` the fixture
        attached (real ``interbotix_xs_msgs/srv/TorqueEnable`` clients).
        Recovery is ``RESTART_REQUIRED``: re-torque and re-home with the
        Interbotix tooling, then relaunch the HAL node.
        """
        from openral_hal.protocol import EStopRecovery

        with pytest.raises(ROSEStopRequested, match="downstream stop acknowledged"):
            aloha_hal.estop()
        report = aloha_hal.last_stop_report
        assert report is not None and report.stopped, report
        assert set(report.controller_states.values()) == {"torque_off"}
        assert aloha_hal.estop_recovery is EStopRecovery.RESTART_REQUIRED
