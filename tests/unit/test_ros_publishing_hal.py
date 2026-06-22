"""Unit tests for :class:`openral_runner.ROSPublishingHAL`.

The adapter is the single change to the in-process hot path mandated by
ADR-0018 §F1 — it replaces a motor-driving HAL with a publisher of
``openral_msgs/ActionChunk`` on ``/openral/candidate_action`` while
keeping the existing `DeployRunner._tick_impl` contract intact.

Two test tiers (mirrors ``tests/unit/test_diagnostics_heartbeat.py``):

* **Construction / validation** — no rclpy required. Asserts read /
  send before connect raise typed errors and the chunk-row → flat
  serialisation respects the row-major contract.
* **Live publish/subscribe** — gated on rclpy. Drives a real adapter
  attached to a real ``LifecycleNode``, publishes one ``Action``, opens
  an rclpy subscriber in the same process, and asserts the typed
  ``ActionChunk`` arrives with the right ``flat`` / ``n_dof`` /
  ``rskill_id`` / ``trace_id`` fields.

Per CLAUDE.md §1.11 — no mocks. All Pydantic schemas are real,
``RobotDescription`` comes from a real fixture.
"""

from __future__ import annotations

import time

import pytest
from openral_core import (
    Action,
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_core.exceptions import ROSConfigError, ROSEStopRequested, ROSRuntimeError
from openral_runner.ros_publishing_hal import ROSPublishingHAL, _row_major_flatten


def _so100_like_description() -> RobotDescription:
    """Real :class:`RobotDescription` with six revolute joints (SO-100-shaped)."""
    joints = [
        JointSpec(
            name=f"j{i}",
            joint_type=JointType.REVOLUTE,
            parent_link=f"link_{i}",
            child_link=f"link_{i + 1}",
        )
        for i in range(6)
    ]
    return RobotDescription(
        name="so100_test",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=joints,
        capabilities=RobotCapabilities(
            supported_control_modes=[ControlMode.JOINT_POSITION],
        ),
        safety=SafetyEnvelope(),
    )


# ── Construction / validation (no rclpy) ─────────────────────────────────────


def test_row_major_flatten_handles_none_and_empty() -> None:
    """``_row_major_flatten`` returns [] for None / empty inputs."""
    assert _row_major_flatten(None) == []
    assert _row_major_flatten([]) == []


def test_row_major_flatten_row_major_order() -> None:
    """The flat array preserves row-major chunk ordering."""
    rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert _row_major_flatten(rows) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_read_state_before_connect_raises() -> None:
    """The HAL Protocol mandates a connect/read sequence."""

    class _StubNode:
        """Constructor never touches the node — safe to use a stub."""

    hal = ROSPublishingHAL(
        node=_StubNode(),  # type: ignore[arg-type]
        description=_so100_like_description(),
    )
    with pytest.raises(ROSRuntimeError, match=r"before connect"):
        hal.read_state()


def test_send_action_before_connect_raises() -> None:
    """Same as ``read_state``: connect must precede send."""

    class _StubNode:
        pass

    hal = ROSPublishingHAL(
        node=_StubNode(),  # type: ignore[arg-type]
        description=_so100_like_description(),
    )
    action = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.0] * 6],
    )
    with pytest.raises(ROSRuntimeError, match=r"before connect"):
        hal.send_action(action)


def test_estop_raises_typed() -> None:
    """``estop()`` raises the canonical ``ROSEStopRequested``."""

    class _StubNode:
        pass

    hal = ROSPublishingHAL(
        node=_StubNode(),  # type: ignore[arg-type]
        description=_so100_like_description(),
    )
    with pytest.raises(ROSEStopRequested):
        hal.estop()


# ── Live publish/subscribe (rclpy-gated) ─────────────────────────────────────


def _rclpy_available() -> bool:
    """True iff rclpy + openral_msgs + sensor_msgs are importable."""
    try:
        import openral_msgs.msg  # noqa: F401
        import rclpy
        import rclpy.lifecycle  # noqa: F401
        import sensor_msgs.msg  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _rclpy_available(),
    reason="rclpy / openral_msgs / sensor_msgs not on PYTHONPATH; "
    "source a ROS 2 install + colcon-build openral_msgs to run live tests",
)
def test_publish_action_chunk_round_trip() -> None:
    """Real Action → ROSPublishingHAL → /openral/candidate_action subscriber."""
    import rclpy
    from openral_msgs.msg import ActionChunk
    from rclpy.lifecycle import LifecycleNode

    rclpy.init()
    received: list[ActionChunk] = []
    host: LifecycleNode | None = None
    sub_node = None
    try:
        host = LifecycleNode("openral_ros_publishing_hal_test")
        sub_node = rclpy.create_node("openral_ros_publishing_hal_test_sub")
        sub_node.create_subscription(
            ActionChunk,
            "/openral/candidate_action",
            received.append,
            1,
        )

        hal = ROSPublishingHAL(
            node=host,
            description=_so100_like_description(),
            skill_id_getter=lambda: "openral/test-skill",
            skill_revision_getter=lambda: "rev-1",
        )
        hal.connect()

        deadline = time.monotonic() + 1.0
        action = Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=2,
            joint_targets=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.2] * 6],
            ee_name="tool0",
            frame_id="base_link",
            confidence=0.9,
        )
        # Spin briefly to let discovery complete.
        while time.monotonic() < deadline and not received:
            hal.send_action(action)
            rclpy.spin_once(host, timeout_sec=0.02)
            rclpy.spin_once(sub_node, timeout_sec=0.02)

        assert received, "no ActionChunk received within 1 s"
        chunk = received[0]
        assert int(chunk.n_dof) == 6
        assert int(chunk.horizon) == 2
        assert list(chunk.flat) == pytest.approx(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        )
        assert chunk.rskill_id == "openral/test-skill"
        assert chunk.rskill_revision == "rev-1"
        assert chunk.ee_name == "tool0"
        assert chunk.frame_id == "base_link"
        # control_mode == 0 (JOINT_POSITION) per the _CONTROL_MODE_TO_UINT8 map.
        assert int(chunk.control_mode) == 0
        # No active OTel context → empty trace_id (no fabrication).
        assert chunk.trace_id == ""

        hal.disconnect()
    finally:
        if sub_node is not None:
            sub_node.destroy_node()
        if host is not None:
            host.destroy_node()
        rclpy.shutdown()


@pytest.mark.skipif(
    not _rclpy_available(),
    reason="rclpy / sensor_msgs not on PYTHONPATH",
)
def test_read_state_caches_joint_states() -> None:
    """A publisher on /joint_states fills the adapter's cached read_state."""
    import rclpy
    from rclpy.lifecycle import LifecycleNode
    from sensor_msgs.msg import JointState as RosJointState

    rclpy.init()
    host: LifecycleNode | None = None
    pub_node = None
    try:
        host = LifecycleNode("openral_ros_publishing_hal_test_state")
        hal = ROSPublishingHAL(
            node=host,
            description=_so100_like_description(),
        )
        hal.connect()
        # Before any /joint_states arrives, read_state must raise.
        with pytest.raises(ROSRuntimeError, match=r"no /joint_states"):
            hal.read_state()

        pub_node = rclpy.create_node("openral_ros_publishing_hal_test_state_pub")
        pub = pub_node.create_publisher(RosJointState, "/joint_states", 5)

        msg = RosJointState()
        msg.name = [f"j{i}" for i in range(6)]
        msg.position = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        msg.velocity = [0.0] * 6
        msg.effort = [0.0] * 6

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            pub.publish(msg)
            rclpy.spin_once(host, timeout_sec=0.02)
            rclpy.spin_once(pub_node, timeout_sec=0.02)
            try:
                state = hal.read_state()
                # First successful read — assert and break.
                assert list(state.position) == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
                break
            except ROSRuntimeError:
                continue
        else:
            pytest.fail("read_state never returned a cached state within 1 s")

        hal.disconnect()
    finally:
        if pub_node is not None:
            pub_node.destroy_node()
        if host is not None:
            host.destroy_node()
        rclpy.shutdown()


@pytest.mark.skipif(
    not _rclpy_available(),
    reason="rclpy / openral_msgs not on PYTHONPATH",
)
def test_send_action_rejects_unsupported_control_mode() -> None:
    """cartesian_pose / foot_placement / dex_hand_joint are still out of the F1 wire.

    ADR-0028c/0028d wired cartesian_delta / cartesian_twist / body_twist /
    gripper / composite_mode onto the typed ActionChunk, so those are now
    accepted; only cartesian_pose, foot_placement and dex_hand_joint remain
    unserialised. Assert one of the still-unsupported modes is rejected.
    """
    import rclpy
    from rclpy.lifecycle import LifecycleNode

    rclpy.init()
    host: LifecycleNode | None = None
    try:
        host = LifecycleNode("openral_ros_publishing_hal_test_modes")
        hal = ROSPublishingHAL(
            node=host,
            description=_so100_like_description(),
        )
        hal.connect()

        action = Action(control_mode=ControlMode.CARTESIAN_POSE, horizon=1)
        with pytest.raises(ROSConfigError, match=r"does not serialise"):
            hal.send_action(action)

        hal.disconnect()
    finally:
        if host is not None:
            host.destroy_node()
        rclpy.shutdown()
