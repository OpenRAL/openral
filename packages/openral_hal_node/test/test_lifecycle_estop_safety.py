"""Hardware-free e-stop / diagnostics wiring of the generic HAL node (Galaxea A1 real HAL)."""

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


class _RecordingLogger:
    """Captures the node's log calls by severity (the node's own rcutils logger is the sink)."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def __getattr__(self, level: str) -> object:
        return lambda msg, *_a, **_k: self.records.append((level, str(msg)))


def _so101_hal_on_fake_bus(**bus_kwargs: object) -> object:
    """A connected real ``SO100FollowerHAL`` whose servos sit behind the fake Feetech bus."""
    import sys
    from pathlib import Path

    # colcon runs this from the package; the serial-boundary fake lives in the repo's tests tree.
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from openral_hal.so100_follower import SO100FollowerHAL
    from openral_hal.so100_sim import SO100DigitalTwin, SO100DigitalTwinConfig

    from tests.unit.fakes.fake_feetech_bus import FakeFeetechBus

    twin = SO100DigitalTwin(SO100DigitalTwinConfig())
    twin.bus = FakeFeetechBus(**bus_kwargs)  # type: ignore[arg-type]
    hal = SO100FollowerHAL(robot=twin)
    hal._robot = twin  # bypass the connect-time ping so a dark servo reaches the e-stop
    hal._connected = True
    return hal


@pytest.mark.parametrize(
    ("bus_kwargs", "level", "text"),
    [
        ({}, "error", "hardware estop completed"),
        (
            {"dark": frozenset({"elbow_flex"})},
            "fatal",
            "hardware estop NOT acknowledged downstream",
        ),
    ],
    ids=["acknowledged", "failed_write"],
)
def test_so101_estop_logs_error_when_acknowledged_and_fatal_when_a_write_fails(
    bus_kwargs: dict[str, object], level: str, text: str
) -> None:
    """Issue #526: an acknowledged SO-101 stop is ERROR "completed", a failed one FATAL."""
    rclpy.init()
    try:
        node = ManifestHALLifecycleNode("openral_hal_so101_follower")
        logger = _RecordingLogger()
        node.get_logger = lambda: logger  # type: ignore[method-assign]
        node._hal = cast(HAL, _so101_hal_on_fake_bus(**bus_kwargs))
        try:
            node._on_estop(object())
            assert node._estopped
            hits = [msg for lvl, msg in logger.records if text in msg]
            assert hits, logger.records
            assert any(lvl == level for lvl, msg in logger.records if text in msg)
            fields = node._downstream_stop_fields()
            expected = "acknowledged" if level == "error" else "unacknowledged"
            assert fields["downstream_stop"] == expected
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()
