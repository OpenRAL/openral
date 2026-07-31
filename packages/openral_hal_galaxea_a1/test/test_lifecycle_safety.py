"""Hardware-free lifecycle safety wiring for the Galaxea A1 package."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

rclpy = pytest.importorskip("rclpy")

from openral_core.exceptions import ROSEStopRequested
from openral_hal.lifecycle import ManifestHALLifecycleNode
from openral_hal.protocol import HAL, EStopRecovery, HALHealthReport
from openral_observability import Level


class _FakeRealHAL:
    description = SimpleNamespace(name="galaxea_a1")
    estop_recovery = EStopRecovery.RESTART_REQUIRED

    def __init__(self) -> None:
        self.estop_calls = 0

    def estop(self) -> None:
        self.estop_calls += 1
        raise ROSEStopRequested("fake A1 sidecar stopped")

    def health(self) -> HALHealthReport:
        return HALHealthReport(
            message="A1 healthy",
            fields={"motor_status_codes": "0,0,0,0,0,0,0"},
        )


class _FakeLatchOnlyHAL:
    description = SimpleNamespace(name="legacy_hal")

    def __init__(self) -> None:
        self.estop_calls = 0

    def estop(self) -> None:
        self.estop_calls += 1
        raise ROSEStopRequested("must not be forwarded without explicit opt-in")


class _FakeResettableHAL(_FakeRealHAL):
    estop_recovery = EStopRecovery.RESETTABLE

    def __init__(self) -> None:
        super().__init__()
        self.reset_calls = 0

    def reset_estop(self) -> None:
        self.reset_calls += 1


class _FakeFailingEStopHAL(_FakeRealHAL):
    def estop(self) -> None:
        self.estop_calls += 1
        raise RuntimeError("fake downstream stop failed")


def test_estop_reaches_real_hal_and_reset_requires_restart() -> None:
    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_galaxea_a1")
        hal = _FakeRealHAL()
        node._hal = cast(HAL, hal)
        try:
            node._on_estop(object())
            assert node._estopped
            assert hal.estop_calls == 1

            node._on_estop_cleared(object())
            assert node._estopped
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_estopped_node_does_not_poll_disconnected_real_hal() -> None:
    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_galaxea_a1")
        node._hal = cast(HAL, _FakeRealHAL())
        node._publisher = object()
        node._estopped = True
        try:
            node._publish_joint_state()
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_estopped_snapshot_backed_node_keeps_publishing_joint_state() -> None:
    """Snapshot-backed (sim-attached) HALs keep /joint_states alive through the latch.

    Only real HALs skip the publish while ``_estopped`` — their transport may
    already be torn down. A proprio snapshot is plain data, so operators keep
    joint visibility during an e-stop instead of the topic going dark.
    """
    import time

    from openral_core.schemas import JointState
    from openral_hal.galaxea_a1 import GALAXEA_A1_DESCRIPTION, GalaxeaA1HAL
    from openral_hal.proprio_snapshot import ProprioFrame, ProprioSnapshot
    from sensor_msgs.msg import JointState as RosJointState

    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_galaxea_a1_snapshot_estop")
        listener = rclpy.create_node("t_estop_joint_state_listener")
        received: list[RosJointState] = []
        listener.create_subscription(RosJointState, "/t_estop_joint_states", received.append, 5)
        try:
            node._publisher = node.create_publisher(RosJointState, "/t_estop_joint_states", 5)
            node._hal = cast(HAL, GalaxeaA1HAL())
            names = [joint.name for joint in GALAXEA_A1_DESCRIPTION.joints]
            positions = [0.25] * len(names)
            snapshot = ProprioSnapshot()
            snapshot.set(
                ProprioFrame(
                    state=JointState(name=names, position=positions, stamp_ns=1),
                    base_pose=(0.0, 0.0, 0.0),
                    base_pose_6dof=None,
                    base_twist=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                )
            )
            node._proprio = snapshot
            node._estopped = True
            node._publish_joint_state()
            deadline = time.monotonic() + 2.0
            while not received and time.monotonic() < deadline:
                rclpy.spin_once(listener, timeout_sec=0.1)
            assert received, "estopped snapshot-backed node stopped publishing joint state"
            assert list(received[0].position) == positions
        finally:
            node.destroy_node()
            listener.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_downstream_estop_failure_stays_latched_without_crashing_callback() -> None:
    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_galaxea_a1")
        hal = _FakeFailingEStopHAL()
        node._hal = cast(HAL, hal)
        try:
            node._on_estop(object())
            assert node._estopped
            assert hal.estop_calls == 1
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_diagnostics_include_cached_a1_health() -> None:
    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_galaxea_a1")
        node._hal = cast(HAL, _FakeRealHAL())
        try:
            level, message, fields = node._heartbeat_status("galaxea_a1")
            assert level == Level.OK
            assert message == "A1 healthy"
            assert fields["motor_status_codes"] == "0,0,0,0,0,0,0"

            node._estopped = True
            level, message, fields = node._heartbeat_status("galaxea_a1")
            assert level == Level.ERROR
            assert message == "estop latched"
            assert fields["estopped"] == "true"
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_hal_without_lifecycle_estop_extension_keeps_latch_only_behavior() -> None:
    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_legacy")
        hal = _FakeLatchOnlyHAL()
        node._hal = cast(HAL, hal)
        try:
            node._on_estop(object())
            assert node._estopped
            assert hal.estop_calls == 0

            node._on_estop_cleared(object())
            assert not node._estopped
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_resettable_lifecycle_estop_clears_only_after_hal_reset() -> None:
    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_resettable")
        hal = _FakeResettableHAL()
        node._hal = cast(HAL, hal)
        try:
            node._on_estop(object())
            node._on_estop_cleared(object())

            assert hal.estop_calls == 1
            assert hal.reset_calls == 1
            assert not node._estopped
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()
