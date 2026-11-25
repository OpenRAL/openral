"""Production ``rclpy`` transport for the Interbotix XS arms (ALOHA).

A real ALOHA runs ``interbotix_xs_sdk``'s ``xs_sdk`` node per follower arm
(issue #250): no ``controller_manager``, so the ros2_control stop seam in
``RosControlTransport`` has nothing to deactivate. What ``xs_sdk`` does expose
is ``/<robot_name>/torque_enable`` (``interbotix_xs_msgs/srv/TorqueEnable``:
``cmd_type`` ``'group'``/``'single'``, ``name``, ``enable``), and torque off is
the one stop the SDK has — with torque cut, no trajectory the node still holds
can move the arm.

This transport carries the whole ``xs_sdk`` contract ``AlohaHAL`` speaks:

* **Commands** — ``publish(topic, msg)`` turns the HAL's ``{name, cmd}``
  payloads into ``interbotix_xs_msgs/JointGroupCommand`` on
  ``/<ns>/commands/joint_group`` and ``JointSingleCommand`` on
  ``/<ns>/commands/joint_single``; an unknown topic raises.
* **State** — one ``/<ns>/joint_states`` subscription per arm. ``xs_sdk``
  publishes unprefixed names (``waist`` … ``wrist_rotate``, the gripper
  motor, ``left_finger`` / ``right_finger``), so arm ``i``'s arm joints are
  prefixed with ``left_`` / ``right_`` and merged by name into the
  manifest's 14-joint order. The manifest gripper is read the way upstream
  ``real_env.get_qpos`` reads it: the ``left_finger`` position (metres),
  normalised onto ``[0, 1]`` with ``gripper_position_range``, velocity by
  the same span; its effort is the gripper motor's. Every effort stays
  ``xs_sdk`` present current (mA). Names outside the manifest are dropped.
  ``last_arrival`` is the *older* arm's newest arrival, and ``0.0`` until
  both arms reported and every manifest joint has arrived, so a silent arm
  or a missing joint reads stale instead of zero-filled.
* **Stop** — ``openral_hal.aloha.InterbotixStopSeam``: the torque-off stop,
  plus a read-only ``/<robot_name>/get_robot_info`` probe ``AlohaHAL`` uses
  to confirm each arm's operating modes before its first command.

The lifecycle node builds it for any ``InterbotixStoppable`` HAL under
``hal_mode:=real`` and attaches it *before* ``connect()``. Never run against a
rig (#250): namespaces, ``motor_config`` and the gripper range are stock
Interbotix defaults until a cell confirms them.

Publishers and subscriptions live on the lifecycle node (its executor spins
them). Service calls run on a private helper node with its own executor, so they can
be made from inside the lifecycle node's ``/openral/estop`` callback without
deadlocking the single-threaded executor that callback runs on.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import structlog

from openral_hal.ros_control import TriggerReport
from openral_hal.ros_control_transport import _COMMAND_DEPTH, _STATE_DEPTH

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from rclpy.node import Node

__all__ = ["InterbotixXSTransport"]

log = structlog.get_logger(__name__)

# Arm i of `arm_namespaces` -> prefix of its manifest joint names.
_SIDES = ("left", "right")
# The xs_sdk joint upstream real_env.get_qpos reads the gripper from (qpos_raw[7]).
_FINGER_JOINT = "left_finger"


class InterbotixXSTransport:
    """``xs_sdk`` command / state / torque-stop wiring for a bimanual Interbotix HAL.

    Args:
        node: The lifecycle node. Command publishers and joint-state
            subscriptions are created on it; its context is borrowed for the
            helper node the service clients live on.
        arm_namespaces: ``(left, right)`` ``xs_sdk`` robot namespaces, from
            ``hal.arm_namespaces()``. Every client is created at wire-up so
            the stop path never has to create one.
        joint_names: The HAL's 14 joint names in manifest order
            (``left_waist`` … ``right_gripper``).
        gripper_name: The ``xs_sdk`` single motor of each gripper (the HAL's
            ``gripper_name``); its effort becomes the manifest gripper's.
        gripper_position_range: ``(closed, open)`` ``left_finger`` metres the
            gripper read-back is normalised with.

    Raises:
        ROSConfigError: If ``interbotix_xs_msgs`` is not installed, there are
            not exactly two namespaces, or ``gripper_position_range`` is not
            two distinct finite values — the transport fails at wire-up, not
            on the first command or e-stop.

    Example:
        >>> transport = InterbotixXSTransport(  # doctest: +SKIP
        ...     node,
        ...     arm_namespaces=hal.arm_namespaces(),
        ...     joint_names=hal.joint_names(),
        ...     gripper_name=hal.gripper_name,
        ...     gripper_position_range=hal.gripper_position_range,
        ... )
        >>> hal.attach_transport(
        ...     transport.publish, transport.state, transport.last_arrival
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        node: Node,
        *,
        arm_namespaces: Sequence[str],
        joint_names: Sequence[str],
        gripper_name: str,
        gripper_position_range: Sequence[float],
    ) -> None:
        """Create the command publishers, state subscriptions and service clients."""
        from openral_core.exceptions import ROSConfigError  # noqa: PLC0415

        if len(arm_namespaces) != len(_SIDES):
            raise ROSConfigError(
                f"InterbotixXSTransport needs exactly {len(_SIDES)} arm namespaces "
                f"(left, right); the HAL reported {list(arm_namespaces)}."
            )
        bounds = [float(v) for v in gripper_position_range]
        if len(bounds) != 2 or not all(map(math.isfinite, bounds)) or bounds[0] == bounds[1]:  # noqa: PLR2004
            raise ROSConfigError(
                "InterbotixXSTransport gripper_position_range must be two distinct finite "
                f"values (closed, open); got {list(gripper_position_range)}."
            )
        try:
            from interbotix_xs_msgs.msg import (  # noqa: PLC0415  # reason: ROS-only dep
                JointGroupCommand,
                JointSingleCommand,
            )
            from interbotix_xs_msgs.srv import (  # noqa: PLC0415  # reason: ROS-only dep
                RobotInfo,
                TorqueEnable,
            )
        except ImportError as exc:
            raise ROSConfigError(
                "interbotix_xs_msgs is not installed, so the ALOHA xs_sdk transport cannot "
                "be wired. Install interbotix_ros_xseries (interbotix_xs_msgs) on this host."
            ) from exc

        import rclpy  # noqa: PLC0415  # reason: ROS-only dep
        from rclpy.executors import SingleThreadedExecutor  # noqa: PLC0415
        from rclpy.node import Node as _Node  # noqa: PLC0415
        from rclpy.qos import (  # noqa: PLC0415  # reason: ROS-only dep
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import JointState as RosJointState  # noqa: PLC0415

        self._joint_names = list(joint_names)
        self._known = set(self._joint_names)
        self._gripper_name = gripper_name
        self._closed, self._span = bounds[0], bounds[1] - bounds[0]
        self._latest: dict[str, tuple[float, float, float]] = {}
        self._side_stamps: dict[str, float] = {}
        self._node = node
        self._srv_type: Any = TorqueEnable  # reason: interbotix_xs_msgs types are ROS-only
        self._info_type: Any = RobotInfo  # reason: interbotix_xs_msgs types are ROS-only
        self._rclpy = rclpy
        # A private node + executor: the e-stop callback runs on the lifecycle
        # node's executor, and a service future can only complete if something
        # else spins the client — so the client lives elsewhere. Built first,
        # so close() can release whatever a failed construction left behind.
        self._helper = _Node(f"{node.get_name()}_torque_stop", context=node.context)
        self._executor = SingleThreadedExecutor(context=node.context)
        self._executor.add_node(self._helper)
        self._pubs: dict[str, tuple[Any, Any]] = {}  # reason: rclpy publisher/msg types
        self._subs: list[Any] = []  # reason: rclpy subscription type
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=_COMMAND_DEPTH,
        )
        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=_STATE_DEPTH,
        )
        try:
            for side, ns in zip(_SIDES, arm_namespaces, strict=True):
                for suffix, msg_type in (
                    ("joint_group", JointGroupCommand),
                    ("joint_single", JointSingleCommand),
                ):
                    topic = f"/{ns}/commands/{suffix}"
                    self._pubs[topic] = (
                        msg_type,
                        node.create_publisher(msg_type, topic, command_qos),
                    )
                self._subs.append(
                    node.create_subscription(
                        RosJointState,
                        f"/{ns}/joint_states",
                        lambda msg, side=side: self._on_joint_state(side, msg),
                        state_qos,
                    )
                )
            # reason (both): rclpy client type
            self._clients: dict[str, Any] = {
                ns: self._helper.create_client(TorqueEnable, f"/{ns}/torque_enable")
                for ns in arm_namespaces
            }
            self._info_clients: dict[str, Any] = {
                ns: self._helper.create_client(RobotInfo, f"/{ns}/get_robot_info")
                for ns in arm_namespaces
            }
        except Exception:
            self.close()  # leave nothing on the lifecycle node's graph, then re-raise
            raise
        log.info(
            "hal.interbotix_transport.ready",
            arms=list(self._clients),
            command_topics=list(self._pubs),
            joints=len(self._joint_names),
        )

    # ── Injected callables (AlohaHAL.attach_transport) ───────────────────────

    def publish(self, topic: str, msg: dict[str, object]) -> None:
        """Turn one HAL ``{name, cmd}`` payload into its ``interbotix_xs_msgs`` message.

        Raises:
            ROSConfigError: If ``topic`` is not one of the four wired command
                topics — HAL-side drift must fail loudly, not publish nowhere.
        """
        from openral_core.exceptions import ROSConfigError  # noqa: PLC0415

        entry = self._pubs.get(topic)
        if entry is None:
            raise ROSConfigError(
                f"InterbotixXSTransport has no command publisher for {topic!r}; "
                f"wired: {sorted(self._pubs)}."
            )
        msg_type, publisher = entry
        out = msg_type()
        out.name = str(msg["name"])
        cmd = msg["cmd"]
        out.cmd = (
            [float(v) for v in cmd] if isinstance(cmd, list) else float(cmd)  # type: ignore[arg-type]  # reason: payload values are untyped
        )
        publisher.publish(out)

    def state(self) -> dict[str, object]:
        """Newest merged state in manifest order; an unheard joint reads 0.0."""
        rows = [self._latest.get(n, (0.0, 0.0, 0.0)) for n in self._joint_names]
        return {
            "name": list(self._joint_names),
            "position": [r[0] for r in rows],
            "velocity": [r[1] for r in rows],
            "effort": [r[2] for r in rows],
        }

    def last_arrival(self) -> float:
        """Older arm's newest ``time.monotonic()`` arrival.

        ``0.0`` (infinitely stale) until both arms reported *and* every
        manifest joint has arrived: ``state`` zero-fills a missing joint, and
        a fresh-looking state with zeros in it is a command to slew home.
        """
        if len(self._side_stamps) < len(_SIDES) or self.missing_joints():
            return 0.0
        return min(self._side_stamps.values())

    def missing_joints(self) -> list[str]:
        """Manifest joints no ``/<ns>/joint_states`` message has carried yet."""
        return [n for n in self._joint_names if n not in self._latest]

    def _on_joint_state(self, side: str, msg: Any) -> None:  # noqa: ANN401  # reason: duck-typed sensor_msgs/JointState
        """Prefix one arm's ``xs_sdk`` names, read the gripper off the finger, merge."""
        positions = list(msg.position)
        velocities = list(msg.velocity)
        efforts = list(msg.effort)

        def row(i: int) -> tuple[float, float, float]:
            return (
                positions[i] if i < len(positions) else 0.0,
                velocities[i] if i < len(velocities) else 0.0,
                efforts[i] if i < len(efforts) else 0.0,
            )

        raw = [str(n) for n in msg.name]
        for i, name in enumerate(raw):
            joint = f"{side}_{name}"
            if joint in self._known and joint != f"{side}_gripper":  # filled from the finger
                self._latest[joint] = row(i)
        if _FINGER_JOINT in raw:
            p, v, _ = row(raw.index(_FINGER_JOINT))
            effort = row(raw.index(self._gripper_name))[2] if self._gripper_name in raw else 0.0
            self._latest[f"{side}_gripper"] = (
                (p - self._closed) / self._span,
                v / self._span,
                effort,
            )
        self._side_stamps[side] = time.monotonic()

    def torque_enable(
        self, robot_name: str, *, group: str, enable: bool, timeout_s: float
    ) -> TriggerReport:
        """Call ``/<robot_name>/torque_enable`` for one group and report the acknowledgement.

        ``TorqueEnable`` has an empty response, so the acknowledgement is the
        call completing: a service that is absent or does not answer within
        ``timeout_s`` is reported as ``success=False``.

        Raises:
            ROSConfigError: If ``robot_name`` was not declared at wire-up.
        """
        from openral_core.exceptions import ROSConfigError  # noqa: PLC0415

        client = self._clients.get(robot_name)
        if client is None:
            raise ROSConfigError(
                f"InterbotixXSTransport has no torque_enable client for {robot_name!r}; "
                f"declared arms: {sorted(self._clients)}."
            )
        if not client.wait_for_service(timeout_sec=timeout_s):
            return TriggerReport(
                success=False, message=f"/{robot_name}/torque_enable not available"
            )
        request = self._srv_type.Request()
        request.cmd_type = "group"
        request.name = group
        request.enable = enable
        future = client.call_async(request)
        self._rclpy.spin_until_future_complete(
            self._helper, future, executor=self._executor, timeout_sec=timeout_s
        )
        if not future.done():
            return TriggerReport(
                success=False, message=f"/{robot_name}/torque_enable did not answer"
            )
        exc = future.exception()
        if exc is not None:
            return TriggerReport(success=False, message=f"/{robot_name}/torque_enable: {exc}")
        return TriggerReport(
            success=True, message=f"/{robot_name}/torque_enable {group} enable={enable}"
        )

    def operating_mode(self, robot_name: str, *, cmd_type: str, name: str, timeout_s: float) -> str:
        """Read one group's / single motor's operating mode off ``get_robot_info``.

        Read-only: ``RobotInfo.Response.mode`` is the mode ``xs_sdk`` holds
        for that group or motor (set by the mode yaml or a later
        ``set_operating_modes``).

        Raises:
            ROSConfigError: If ``robot_name`` was not declared at wire-up.
            ROSRuntimeError: If the service is absent, does not answer within
                ``timeout_s``, or fails.
        """
        from openral_core.exceptions import ROSConfigError, ROSRuntimeError  # noqa: PLC0415

        client = self._info_clients.get(robot_name)
        if client is None:
            raise ROSConfigError(
                f"InterbotixXSTransport has no get_robot_info client for {robot_name!r}; "
                f"declared arms: {sorted(self._info_clients)}."
            )
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise ROSRuntimeError(f"/{robot_name}/get_robot_info not available")
        request = self._info_type.Request()
        request.cmd_type = cmd_type
        request.name = name
        future = client.call_async(request)
        self._rclpy.spin_until_future_complete(
            self._helper, future, executor=self._executor, timeout_sec=timeout_s
        )
        if not future.done():
            raise ROSRuntimeError(f"/{robot_name}/get_robot_info did not answer")
        exc = future.exception()
        if exc is not None:
            raise ROSRuntimeError(f"/{robot_name}/get_robot_info: {exc}") from exc
        return str(future.result().mode)

    def close(self) -> None:
        """Destroy the command/state endpoints and the helper node (lifecycle cleanup)."""
        for _, publisher in self._pubs.values():
            self._node.destroy_publisher(publisher)
        for sub in self._subs:
            self._node.destroy_subscription(sub)
        self._pubs.clear()
        self._subs.clear()
        self._executor.remove_node(self._helper)
        self._helper.destroy_node()
