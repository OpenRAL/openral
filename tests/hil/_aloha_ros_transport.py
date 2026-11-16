"""Lab-runner-only ``rclpy`` bridge for the ALOHA bimanual real-HW HAL.

``openral_hal.aloha.AlohaHAL`` emits the ``interbotix_xs_sdk`` ``xs_sdk`` contract
(issue #250): per follower namespace, a ``JointGroupCommand`` payload on
``/<ns>/commands/joint_group`` (the 6-joint ``arm`` group) and a ``JointSingleCommand``
payload on ``/<ns>/commands/joint_single`` (the ``gripper`` motor). This bridge turns those
payloads into real ``interbotix_xs_msgs`` messages, and merges the two per-namespace
``/<ns>/joint_states`` streams (unprefixed ``xs_sdk`` joint names) into the HAL's 14-DoF
manifest order by prefixing each with its side (``left_`` / ``right_``).

The gripper joint is read back in ``xs_sdk``'s units and normalised onto ``[0, 1]`` with the
manifest's ``gripper_cmd_range`` — the inverse of what ``AlohaHAL.send_action`` applies — so
a hold action built from ``read_state()`` commands the gripper where it already is. Its
velocity is scaled by the same span (normalised units per second). ``effort`` stays in
``xs_sdk``'s unit for every joint — Dynamixel present current in mA, not a torque.

``last_stamp`` is the *older* of the two arms' newest stamps (``0.0`` until both have
reported), so one silent ``xs_sdk`` reads stale in ``AlohaHAL.read_state`` instead of
hiding behind the other arm's stream.

.. warning::
   Test-only, and not yet run against a rig (no ``lab-aloha`` runner, no physical ALOHA).
   It is not the production transport: none exists yet, which is why ``AlohaHAL.connect()``
   refuses without an injected ``publish_fn`` / ``state_fn`` (#250).

HIL-only; needs ``rclpy`` and ``interbotix_xs_msgs`` (CLAUDE.md §1.11: real component or
``pytest.skip`` — nothing in between).
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable, Sequence
from typing import Any

if importlib.util.find_spec("rclpy") is None:  # pragma: no cover
    raise RuntimeError(
        "rclpy is not installed; this transport may only be imported by HIL "
        "tests after they've confirmed rclpy is available."
    )

import rclpy
from interbotix_xs_msgs.msg import (  # ROS-only dep: interbotix_ros_xseries
    JointGroupCommand,
    JointSingleCommand,
)
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState as RosJointState

from tests.hil._ros_control_transport import _PolledJointStateMixin

__all__ = ["AlohaHILTransport", "make_aloha_hil_transport"]


_CONTROL_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, depth=10)
_SIDES = ("left", "right")


class AlohaHILTransport(_PolledJointStateMixin):
    """``xs_sdk`` bridge for the bimanual ALOHA HIL test.

    Owns a ``JointGroupCommand`` and a ``JointSingleCommand`` publisher per follower
    namespace plus one ``JointState`` subscriber per namespace. ``publish`` dispatches by
    exact topic; unknown topics raise ``ValueError`` so HAL-side drift fails loudly.

    Args:
        node: A live ``rclpy`` node owned by the test.
        joint_names: All 14 ALOHA joint names in ``ALOHA_REAL_DESCRIPTION.joints`` order
            (``left_waist`` … ``right_gripper``).
        arm_namespaces: ``(left, right)`` ``xs_sdk`` namespaces, from
            ``AlohaHAL.arm_namespaces()``.
        gripper_cmd_range: ``(closed, open)`` used to normalise the gripper read-back.
    """

    def __init__(
        self,
        node: Node,
        joint_names: list[str],
        *,
        arm_namespaces: Sequence[str],
        gripper_cmd_range: Sequence[float],
    ) -> None:
        if len(joint_names) != 14:
            raise ValueError(f"AlohaHILTransport expects 14 joint names, got {len(joint_names)}.")
        self._node = node
        self._joint_names = list(joint_names)
        self._latest: dict[str, tuple[float, float, float]] = {}
        self._last_stamp = 0.0
        self._side_stamps: dict[str, float] = {}
        self._closed, self._open = (float(v) for v in gripper_cmd_range)
        self._publishers: dict[str, tuple[Any, Any]] = {}
        for side, ns in zip(_SIDES, arm_namespaces, strict=True):
            for suffix, msg_type in (
                ("joint_group", JointGroupCommand),
                ("joint_single", JointSingleCommand),
            ):
                topic = f"/{ns}/commands/{suffix}"
                self._publishers[topic] = (
                    msg_type,
                    node.create_publisher(msg_type, topic, _CONTROL_QOS),
                )
            node.create_subscription(
                RosJointState, f"/{ns}/joint_states", self._side_callback(side), _CONTROL_QOS
            )

    @property
    def last_stamp(self) -> float:
        """Older arm's newest stamp; ``0.0`` until both arms reported (``AlohaHAL.stamp_fn``)."""
        if len(self._side_stamps) < len(_SIDES):
            return 0.0
        return min(self._side_stamps.values())

    def publish(self, topic: str, msg: dict[str, object]) -> None:
        """Turn a HAL ``{name, cmd}`` payload into the matching ``interbotix_xs_msgs`` message."""
        entry = self._publishers.get(topic)
        if entry is None:
            raise ValueError(f"AlohaHILTransport: unknown topic {topic!r}")
        msg_type, publisher = entry
        out = msg_type()
        out.name = str(msg["name"])
        cmd = msg["cmd"]
        out.cmd = [float(v) for v in cmd] if isinstance(cmd, list) else float(cmd)  # type: ignore[arg-type]  # reason: payload values are untyped
        publisher.publish(out)

    def _side_callback(self, side: str) -> Callable[[Any], None]:
        """Prefix one namespace's unprefixed ``xs_sdk`` names, normalise the gripper, merge."""

        def on_state(msg: Any) -> None:
            names = [f"{side}_{n}" for n in msg.name]
            positions = list(msg.position)
            velocities = list(msg.velocity)
            gripper = f"{side}_gripper"
            span = self._open - self._closed
            if gripper in names:
                i = names.index(gripper)
                positions[i] = (positions[i] - self._closed) / span
                if i < len(velocities):
                    velocities[i] /= span
            relabelled = RosJointState()
            relabelled.name = names
            relabelled.position = positions
            relabelled.velocity = velocities
            relabelled.effort = list(msg.effort)
            self._on_joint_state(relabelled)
            self._side_stamps[side] = time.monotonic()

        return on_state


def make_aloha_hil_transport(
    node_name: str, *, arm_namespaces: Sequence[str]
) -> tuple[Node, AlohaHILTransport, Callable[[], None]]:
    """Initialise rclpy and return ``(node, transport, cleanup)`` for the ALOHA HIL test.

    Joint names and the gripper range come from the public ``ALOHA_REAL_DESCRIPTION``
    manifest so the bridge stays in lock-step with the HAL.
    """
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node(node_name)

    from openral_hal.aloha import ALOHA_REAL_DESCRIPTION

    params = ALOHA_REAL_DESCRIPTION.hal.parameters
    transport = AlohaHILTransport(
        node,
        [j.name for j in ALOHA_REAL_DESCRIPTION.joints],
        arm_namespaces=arm_namespaces,
        gripper_cmd_range=params.defaults["gripper_cmd_range"],  # type: ignore[arg-type]  # reason: manifest defaults are untyped
    )

    def cleanup() -> None:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return node, transport, cleanup
