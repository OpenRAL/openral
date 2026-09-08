"""Lab-runner-only ``rclpy`` bridge for the ALOHA bimanual real-HW HAL.

``openral_hal.aloha.AlohaHAL`` splits a single 14-D ``openral_core.Action`` across four
``ros2_control`` controllers — left arm, right arm, left gripper, right gripper. This bridge
is the HIL counterpart of ``tests.hil._ros_control_transport`` for the bimanual fan-out: it
owns four ``trajectory_msgs/JointTrajectory`` publishers + one aggregated
``sensor_msgs/JointState`` subscriber and dispatches by topic match.

All four publishers use ``trajectory_msgs/JointTrajectory`` because every ALOHA controller
(arms and grippers alike) is a ``joint_trajectory_controller/JointTrajectoryController``
instance — grippers are 1-DOF JointTrajectoryControllers. The standalone
``parallel_gripper_action_controller/GripperActionController`` (action interface,
``control_msgs/action/GripperCommand``) and Trossen's native
``interbotix_xs_msgs/JointSingleCommand`` are deliberately not used: neither matches
AlohaHAL's ``publish_fn(topic, msg)`` contract, which fans out via ``self._publish_fn(...)``
four times per ``send_action()`` (see ``python/hal/src/openral_hal/aloha.py``).

HIL-only; shares the import-time ``rclpy`` guard from ``tests.hil._ros_control_transport``
(CLAUDE.md §5.4: real component or ``pytest.skip`` — nothing in between).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

if importlib.util.find_spec("rclpy") is None:  # pragma: no cover
    raise RuntimeError(
        "rclpy is not installed; this transport may only be imported by HIL "
        "tests after they've confirmed rclpy is available."
    )

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState as RosJointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tests.hil._ros_control_transport import _make_trajectory_publisher, _PolledJointStateMixin

__all__ = ["AlohaHILTransport", "make_aloha_hil_transport"]


_CONTROL_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    depth=10,
)

# AlohaHAL exposes 14 joints in this canonical interleaved order
# (left arm 6, left gripper 1, right arm 6, right gripper 1).  See
# ``python/hal/src/openral_hal/aloha.py`` and
# ``ALOHA_REAL_DESCRIPTION.joints`` — re-derived from the public manifest
# by ``make_aloha_hil_transport``.
_LEFT_ARM_SLICE = slice(0, 6)
_LEFT_GRIPPER_INDEX = 6
_RIGHT_ARM_SLICE = slice(7, 13)
_RIGHT_GRIPPER_INDEX = 13


class AlohaHILTransport(_PolledJointStateMixin):
    """4-way ``rclpy`` bridge for the bimanual ALOHA HIL test.

    Owns one ``JointTrajectory`` publisher per controller (two arms + two
    grippers) plus one ``JointState`` subscriber on the aggregated
    ``joint_state_topic``.  Dispatch is by topic-string match against the
    four constructor topics — unknown topics raise ``ValueError`` so that
    a HAL-side contract drift fails loudly (CLAUDE.md §10).

    Args:
        node: A live ``rclpy`` node owned by the test (cleanup is the test's
            responsibility).
        joint_names: All 14 ALOHA joint names in the canonical interleaved
            order from ``ALOHA_REAL_DESCRIPTION.joints``.
        left_arm_command_topic: Topic for the left-arm
            ``JointTrajectoryController`` (e.g.
            ``/left_arm/arm_controller/joint_trajectory``).
        right_arm_command_topic: Same for the right arm.
        left_gripper_command_topic: Topic for the 1-DOF left-gripper
            ``JointTrajectoryController`` (e.g.
            ``/left_arm/gripper_controller/command``).
        right_gripper_command_topic: Same for the right gripper.
        joint_state_topic: Aggregated ``sensor_msgs/JointState`` topic the
            Interbotix XS launch publishes on (default ``"/joint_states"``).
    """

    def __init__(
        self,
        node: Node,
        joint_names: list[str],
        *,
        left_arm_command_topic: str,
        right_arm_command_topic: str,
        left_gripper_command_topic: str,
        right_gripper_command_topic: str,
        joint_state_topic: str = "/joint_states",
    ) -> None:
        if len(joint_names) != 14:
            raise ValueError(f"AlohaHILTransport expects 14 joint names, got {len(joint_names)}.")
        self._node = node
        self._joint_names = list(joint_names)

        # Pre-compute the per-publisher joint-name slices once.
        self._left_arm_joints = self._joint_names[_LEFT_ARM_SLICE]
        self._right_arm_joints = self._joint_names[_RIGHT_ARM_SLICE]
        self._left_gripper_joints = [self._joint_names[_LEFT_GRIPPER_INDEX]]
        self._right_gripper_joints = [self._joint_names[_RIGHT_GRIPPER_INDEX]]

        # Four JointTrajectory publishers, dispatched by topic match.
        self._left_arm_topic = left_arm_command_topic
        self._right_arm_topic = right_arm_command_topic
        self._left_gripper_topic = left_gripper_command_topic
        self._right_gripper_topic = right_gripper_command_topic
        self._left_arm_pub = _make_trajectory_publisher(node, left_arm_command_topic)
        self._right_arm_pub = _make_trajectory_publisher(node, right_arm_command_topic)
        self._left_gripper_pub = _make_trajectory_publisher(node, left_gripper_command_topic)
        self._right_gripper_pub = _make_trajectory_publisher(node, right_gripper_command_topic)

        self._latest: dict[str, tuple[float, float, float]] = {}
        self._last_stamp = 0.0
        node.create_subscription(
            RosJointState,
            joint_state_topic,
            self._on_joint_state,
            _CONTROL_QOS,
        )

    # -- Transport callables (injected into the HAL) --------------------------

    def publish(self, topic: str, msg: dict[str, object]) -> None:
        """Dispatch a HAL publish call to the matching JointTrajectory publisher."""
        if topic == self._left_arm_topic:
            self._publish_arm(self._left_arm_pub, self._left_arm_joints, msg)
        elif topic == self._right_arm_topic:
            self._publish_arm(self._right_arm_pub, self._right_arm_joints, msg)
        elif topic == self._left_gripper_topic:
            self._publish_gripper(self._left_gripper_pub, self._left_gripper_joints, msg)
        elif topic == self._right_gripper_topic:
            self._publish_gripper(self._right_gripper_pub, self._right_gripper_joints, msg)
        else:
            raise ValueError(f"AlohaHILTransport: unknown topic {topic!r}")

    # state/spin_once/last_stamp/wait_for_first_state come from
    # _PolledJointStateMixin.

    # -- Internal publish helpers --------------------------------------------

    @staticmethod
    def _publish_arm(publisher: Any, joint_names: list[str], msg: dict[str, object]) -> None:
        """Emit a 6-DOF ``JointTrajectory`` for an arm chunk (last step only)."""
        targets = msg.get("joint_targets")
        if not isinstance(targets, list) or not targets:
            return
        last_step = targets[-1]
        if not isinstance(last_step, list):
            return
        traj = JointTrajectory()
        traj.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in last_step]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100_000_000
        traj.points.append(point)
        publisher.publish(traj)

    @staticmethod
    def _publish_gripper(publisher: Any, joint_names: list[str], msg: dict[str, object]) -> None:
        """Emit a 1-DOF ``JointTrajectory`` for a single-position gripper command."""
        position = msg.get("position")
        if position is None:
            return
        traj = JointTrajectory()
        traj.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100_000_000
        traj.points.append(point)
        publisher.publish(traj)

    # _on_joint_state comes from _PolledJointStateMixin.


def make_aloha_hil_transport(
    node_name: str,
    *,
    left_arm_command_topic: str = "/left_arm/arm_controller/joint_trajectory",
    right_arm_command_topic: str = "/right_arm/arm_controller/joint_trajectory",
    left_gripper_command_topic: str = "/left_arm/gripper_controller/command",
    right_gripper_command_topic: str = "/right_arm/gripper_controller/command",
    joint_state_topic: str = "/joint_states",
) -> tuple[Node, AlohaHILTransport, Callable[[], None]]:
    """Initialise rclpy and return ``(node, transport, cleanup)`` for the ALOHA HIL test.

    Joint names are derived from the public ``ALOHA_REAL_DESCRIPTION``
    manifest so the bridge stays in lock-step with the HAL — no private
    name imports across the hal-package boundary.
    """
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node(node_name)

    # Imported lazily so ``importlib.util.find_spec("rclpy") is None``
    # users (the off-lab skip path) never load the HAL package's heavy
    # dependencies.
    from openral_hal.aloha import ALOHA_REAL_DESCRIPTION

    joint_names = [j.name for j in ALOHA_REAL_DESCRIPTION.joints]
    transport = AlohaHILTransport(
        node,
        joint_names,
        left_arm_command_topic=left_arm_command_topic,
        right_arm_command_topic=right_arm_command_topic,
        left_gripper_command_topic=left_gripper_command_topic,
        right_gripper_command_topic=right_gripper_command_topic,
        joint_state_topic=joint_state_topic,
    )

    def cleanup() -> None:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return node, transport, cleanup
