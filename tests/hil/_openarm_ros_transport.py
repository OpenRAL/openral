# SPDX-License-Identifier: Apache-2.0
"""Lab-runner-only ``rclpy`` bridge for the bimanual OpenArm v2 real-HW HAL.

:class:`openral_hal.openarm_real.OpenArmRealHAL` fans one 16-DoF
:class:`openral_core.Action` across **four** ``ros2_control`` controllers —
left arm, left gripper, right arm, right gripper (``openarm_bringup``'s
bimanual configuration). This is the HIL counterpart of
:mod:`tests.hil._aloha_ros_transport` for that fan-out.

Simpler than the ALOHA bridge, because the OpenArm HAL puts ``joint_names``
**in the message it publishes** (ADR-0102). The ALOHA bridge has to carry its
own slice table to know which joints each publisher owns; here the message
says so, and the transport just forwards it. That also means the transport
cannot silently disagree with the HAL about joint order — there is no second
copy of the mapping to drift.

Those names are in the **ros2_control namespace**
(``openarm_left_joint1``), not the manifest's (``left_joint1``); the adapter
translates, and ``/joint_states`` is keyed the same way. Build this transport
from :meth:`OpenArmRealHAL.ros2_control_joint_names`, never from
``description.joints``.

``time_from_start`` is a **constructor argument** here, unlike the 100 ms the
production transports hardcode. A ``JointTrajectoryController`` given an
absolute target and a 100 ms deadline moves at ``(target - current) / 0.1s``,
so on a first powered run the rate is set by how wrong the command is — which
is the quantity under test. A longer window bounds the rate by construction.
Tests that care about production timing must say so and pass 0.1.

This module is HIL-only and shares the import-time ``rclpy`` guard from
:mod:`tests.hil._ros_control_transport` (CLAUDE.md §1.11: real component or
``pytest.skip`` — nothing in between).
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any

if importlib.util.find_spec("rclpy") is None:  # pragma: no cover
    raise RuntimeError(
        "rclpy is not installed; this transport may only be imported by HIL "
        "tests after they've confirmed rclpy is available."
    )

from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState as RosJointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tests.hil._ros_control_transport import _CONTROL_QOS, _make_trajectory_publisher

__all__ = ["OpenArmHILTransport"]


class OpenArmHILTransport:
    """4-way ``rclpy`` bridge for the bimanual OpenArm HIL tests.

    Owns one ``JointTrajectory`` publisher per controller plus one
    ``JointState`` subscriber on the aggregated ``joint_state_topic``.
    Dispatch is by topic-string match; an unknown topic raises so a HAL-side
    contract drift fails loudly rather than dropping a command.

    Args:
        node: A live ``rclpy`` node owned by the caller (teardown is the
            caller's responsibility).
        joint_names: All 16 joint names in **ros2_control** order — i.e.
            :meth:`OpenArmRealHAL.ros2_control_joint_names`, which is what
            ``/joint_states`` is keyed by.
        command_topics: The four controller command topics, from
            :meth:`OpenArmRealHAL.command_topics`.
        joint_state_topic: Aggregated ``sensor_msgs/JointState`` topic.
        time_from_start_s: Trajectory deadline for every published point. See
            the module docstring — this bounds the motion rate.

    Example:
        >>> OpenArmHILTransport.__name__
        'OpenArmHILTransport'
    """

    def __init__(
        self,
        node: Node,
        joint_names: list[str],
        *,
        command_topics: list[str],
        joint_state_topic: str = "/joint_states",
        time_from_start_s: float = 0.8,
    ) -> None:
        """Build the four publishers and the joint-state subscription."""
        if len(joint_names) != 16:
            raise ValueError(f"OpenArmHILTransport expects 16 joint names, got {len(joint_names)}.")
        if len(command_topics) != 4:
            raise ValueError(
                f"OpenArmHILTransport expects 4 command topics, got {len(command_topics)}."
            )
        if time_from_start_s <= 0.0:
            raise ValueError("time_from_start_s must be > 0.")
        self._node = node
        self._joint_names = list(joint_names)
        self._time_from_start_s = float(time_from_start_s)
        self._pubs = {t: _make_trajectory_publisher(node, t) for t in command_topics}

        self._latest: dict[str, tuple[float, float, float]] = {}
        self._last_stamp = 0.0
        node.create_subscription(
            RosJointState, joint_state_topic, self._on_joint_state, _CONTROL_QOS
        )

        # Bound to the NODE's context, not the default one. A HIL test runs on
        # its own `rclpy.Context` so it cannot join the lab's live graph by
        # accident, and bare `rclpy.spin_once(node)` would reach for the
        # uninitialised default context and raise.
        self._executor = SingleThreadedExecutor(context=node.context)
        self._executor.add_node(node)

    # -- Transport callables (injected into the HAL) --------------------------

    def publish(self, topic: str, msg: dict[str, Any]) -> None:
        """Forward one HAL command to its controller as a ``JointTrajectory``.

        The message's own ``joint_names`` are used verbatim, so this cannot
        misroute a value the HAL placed correctly.

        Args:
            topic: One of the four command topics.
            msg: The HAL's command dict.

        Raises:
            ValueError: The topic is not one of the four (contract drift).
        """
        publisher = self._pubs.get(topic)
        if publisher is None:
            raise ValueError(f"OpenArmHILTransport: unknown topic {topic!r}")
        names = msg.get("joint_names")
        targets = msg.get("joint_targets")
        if not isinstance(names, list) or not isinstance(targets, list) or not targets:
            raise ValueError(f"OpenArmHILTransport: malformed command on {topic!r}: {msg!r}")
        last_step = targets[-1]
        if not isinstance(last_step, list) or len(last_step) != len(names):
            raise ValueError(
                f"OpenArmHILTransport: {topic!r} names {len(names)} joints but the "
                f"chunk's last step has {len(last_step) if isinstance(last_step, list) else '?'} "
                "values."
            )
        traj = JointTrajectory()
        traj.joint_names = [str(n) for n in names]
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in last_step]
        whole = int(self._time_from_start_s)
        point.time_from_start.sec = whole
        point.time_from_start.nanosec = int((self._time_from_start_s - whole) * 1e9)
        traj.points.append(point)
        publisher.publish(traj)

    def state(self) -> dict[str, object]:
        """Latest joint state in the transport's joint-name order."""
        positions: list[float] = []
        velocities: list[float] = []
        efforts: list[float] = []
        for name in self._joint_names:
            p, v, e = self._latest.get(name, (0.0, 0.0, 0.0))
            positions.append(p)
            velocities.append(v)
            efforts.append(e)
        return {"position": positions, "velocity": velocities, "effort": efforts}

    # -- Helpers --------------------------------------------------------------

    def spin_once(self, timeout_sec: float = 0.05) -> None:
        """Pump the node's callbacks once, on its own context's executor."""
        self._executor.spin_once(timeout_sec=timeout_sec)

    def spin_for(self, seconds: float) -> None:
        """Pump callbacks for a wall-clock window."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.spin_once(timeout_sec=0.02)

    def seen_joints(self) -> set[str]:
        """Joint names that have actually appeared on ``/joint_states``."""
        return set(self._latest)

    def missing_joints(self) -> list[str]:
        """Expected joints that ``/joint_states`` has never reported."""
        return [n for n in self._joint_names if n not in self._latest]

    def close(self) -> None:
        """Release the executor. The node and context belong to the caller."""
        self._executor.shutdown()

    # -- Internal callbacks ---------------------------------------------------

    def _on_joint_state(self, msg: Any) -> None:
        """Record the latest position/velocity/effort per joint name.

        ``/joint_states`` on the OpenArm is aggregated from four independently
        publishing controllers with no ordering guarantee, so entries are
        merged by name and never by index.
        """
        names = list(getattr(msg, "name", []))
        positions = list(getattr(msg, "position", []))
        velocities = list(getattr(msg, "velocity", []))
        efforts = list(getattr(msg, "effort", []))
        for i, name in enumerate(names):
            self._latest[name] = (
                positions[i] if i < len(positions) else 0.0,
                velocities[i] if i < len(velocities) else 0.0,
                efforts[i] if i < len(efforts) else 0.0,
            )
        self._last_stamp = time.monotonic()

    @property
    def last_stamp(self) -> float:
        """Monotonic timestamp of the most recent ``/joint_states`` message."""
        return self._last_stamp

    def wait_for_every_joint(self, deadline_s: float = 5.0) -> bool:
        """Block until every expected joint has been reported at least once.

        Not "any message": :meth:`state` zero-fills an unreported joint, and
        :class:`OpenArmRealHAL` builds a full 16-DoF vector regardless. A
        partial ``/joint_states`` therefore reads as a *plausible pose* with
        zeros in it, which is the one input a motion test must never act on.

        Args:
            deadline_s: How long to wait.

        Returns:
            True when all 16 have been seen.
        """
        start = time.monotonic()
        while time.monotonic() - start < deadline_s:
            self.spin_once(timeout_sec=0.05)
            if not self.missing_joints():
                return True
        return False
