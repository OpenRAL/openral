"""The generic lifecycle node's e-stop branching on a real ros2_control adapter (issue #295).

Hardware-free, but not stack-free of what matters: a real ``UR5eRealHAL`` / ``OpenArmRealHAL``
on a real ``rclpy`` node, with ``SimTransport`` — the in-repo controller simulator that
implements the same ``ControllerStopSeam`` the production transport does — attached as the
stop seam. Covers what ``packages/openral_hal_galaxea_a1/test/test_lifecycle_safety.py``
proves with stand-in HALs, on the adapters this issue is about: the node reads the HAL's
``DownstreamStopReport``, surfaces it on the heartbeat, and clears its latch only when the
policy and the downstream re-arm allow. The live counterpart against a real
``controller_manager`` is ``tests/integration/test_real_hal_estop_ros2_control_live.py``.
"""

from __future__ import annotations

from typing import cast

import pytest

rclpy = pytest.importorskip("rclpy")

from openral_hal.lifecycle import ManifestHALLifecycleNode
from openral_hal.protocol import HAL
from openral_hal.sim_transport import SimTransport
from openral_hal.ur_real import UR5eRealHAL
from openral_observability import Level

_UR_CONTROLLER = "scaled_joint_trajectory_controller"


def _ur5e(transport: SimTransport) -> UR5eRealHAL:
    """A connected ``UR5eRealHAL`` with ``transport`` as both drive and stop seam."""
    hal = UR5eRealHAL(publish_fn=transport.publish, state_fn=transport.state)
    hal.attach_controller_stop(transport)
    hal.connect()
    return hal


def test_ur5e_estop_is_acknowledged_on_the_heartbeat_and_clear_is_rejected() -> None:
    """Ur5e estop is acknowledged on the heartbeat and clear is rejected."""
    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_ur5e_estop")
        transport = SimTransport(n_joints=6, controllers=[_UR_CONTROLLER])
        hal = _ur5e(transport)
        node._hal = cast(HAL, hal)
        try:
            node._on_estop(object())
            assert node._estopped
            assert transport.controller_state(_UR_CONTROLLER) == "inactive"
            assert transport.trigger_calls == ["/dashboard_client/stop"]

            level, message, fields = node._heartbeat_status("ur5e")
            assert level == Level.ERROR and message == "estop latched"
            assert fields["downstream_stop"] == "acknowledged"
            assert fields["downstream_controllers"] == _UR_CONTROLLER
            assert fields["downstream_vendor_stop"] == "/dashboard_client/stop"

            node._on_estop_cleared(object())
            assert node._estopped, "RESTART_REQUIRED must reject an in-process clear"
            assert transport.controller_state(_UR_CONTROLLER) == "inactive"
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_a_refused_dashboard_stop_reads_unacknowledged_but_stays_latched() -> None:
    """A refused dashboard stop reads unacknowledged but stays latched."""
    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_ur5e_estop_refused")
        transport = SimTransport(
            n_joints=6,
            controllers=[_UR_CONTROLLER],
            trigger_responses={"/dashboard_client/stop": (False, "not connected")},
        )
        node._hal = cast(HAL, _ur5e(transport))
        try:
            node._on_estop(object())
            assert node._estopped
            # The controller half still happened; the report says the vendor half did not.
            assert transport.controller_state(_UR_CONTROLLER) == "inactive"
            _level, message, fields = node._heartbeat_status("ur5e")
            assert message == "estop latched"
            assert fields["downstream_stop"] == "unacknowledged"
            assert "not connected" in fields["downstream_stop_detail"]
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_openarm_clears_only_after_all_four_controllers_are_reactivated() -> None:
    """Openarm clears only after all four controllers are reactivated."""
    from openral_hal.openarm_real import OpenArmRealHAL

    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_openarm_estop")
        hal = OpenArmRealHAL(require_can_links=False)
        names = hal.controller_names()
        transport = SimTransport(n_joints=16, controllers=names)
        hal.attach_transport(transport.publish, transport.state)
        hal.attach_controller_stop(transport)
        hal.connect()
        node._hal = cast(HAL, hal)
        try:
            node._on_estop(object())
            assert node._estopped
            assert all(transport.controller_state(n) == "inactive" for n in names)

            node._on_estop_cleared(object())
            assert not node._estopped, "RESETTABLE clears once the re-arm is acknowledged"
            assert all(transport.controller_state(n) == "active" for n in names)
            assert node._heartbeat_status("openarm")[2]["estopped"] == "false"
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_openarm_stays_latched_when_reactivation_is_refused() -> None:
    """Openarm stays latched when reactivation is refused."""
    from openral_hal.openarm_real import OpenArmRealHAL

    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_openarm_estop_refused")
        hal = OpenArmRealHAL(require_can_links=False)
        names = hal.controller_names()
        transport = SimTransport(n_joints=16, controllers=names)
        hal.attach_transport(transport.publish, transport.state)
        hal.attach_controller_stop(transport)
        hal.connect()
        node._hal = cast(HAL, hal)
        try:
            node._on_estop(object())
            # The manager unloads one gripper controller between stop and clear.
            transport._controller_states.pop(names[1])
            node._on_estop_cleared(object())
            assert node._estopped, "a refused re-activation must keep the latch"
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()
