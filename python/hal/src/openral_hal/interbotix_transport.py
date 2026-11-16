"""Production ``rclpy`` torque-stop seam for the Interbotix XS arms (ALOHA).

A real ALOHA runs ``interbotix_xs_sdk``'s ``xs_sdk`` node per follower arm
(issue #250): no ``controller_manager``, so the ros2_control stop seam in
``RosControlTransport`` has nothing to deactivate. What ``xs_sdk`` does expose
is ``/<robot_name>/torque_enable`` (``interbotix_xs_msgs/srv/TorqueEnable``:
``cmd_type`` ``'group'``/``'single'``, ``name``, ``enable``), and torque off is
the one stop the SDK has — with torque cut, no trajectory the node still holds
can move the arm.

This transport implements ``openral_hal.aloha.InterbotixStopSeam`` over real
service clients: the torque-off stop, plus a read-only
``/<robot_name>/get_robot_info`` probe ``AlohaHAL`` uses to confirm each arm's
operating modes before its first command. It does not provide the command /
state wiring for a real ALOHA — that is #250's on-rig work and is not
pretended here. The lifecycle node attaches it to any ``InterbotixStoppable`` HAL under
``hal_mode:=real``.

Service calls run on a private helper node with its own executor, so they can
be made from inside the lifecycle node's ``/openral/estop`` callback without
deadlocking the single-threaded executor that callback runs on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from openral_hal.ros_control import TriggerReport

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from rclpy.node import Node

__all__ = ["InterbotixXSTransport"]

log = structlog.get_logger(__name__)


class InterbotixXSTransport:
    """``torque_enable`` clients for every arm namespace an ``AlohaHAL`` declares.

    Args:
        node: The lifecycle node; only its context is borrowed for the helper
            node the clients live on.
        arm_namespaces: ``xs_sdk`` robot namespaces, from
            ``hal.arm_namespaces()``. One client per namespace is created at
            wire-up so the stop path never has to create one.

    Raises:
        ROSConfigError: If ``interbotix_xs_msgs`` is not installed — the stop
            must fail at wire-up, not on the first e-stop.
    """

    def __init__(self, node: Node, *, arm_namespaces: list[str]) -> None:
        """Create the helper node and one ``TorqueEnable`` client per arm."""
        from openral_core.exceptions import ROSConfigError  # noqa: PLC0415

        if not arm_namespaces:
            raise ROSConfigError(
                "InterbotixXSTransport needs at least one arm namespace; the HAL reported "
                "none, so nothing could be torqued off on e-stop."
            )
        try:
            from interbotix_xs_msgs.srv import (  # noqa: PLC0415  # reason: ROS-only dep
                RobotInfo,
                TorqueEnable,
            )
        except ImportError as exc:
            raise ROSConfigError(
                "interbotix_xs_msgs is not installed, so the ALOHA torque-off e-stop cannot "
                "be wired. Install interbotix_ros_xseries (interbotix_xs_msgs) on this host."
            ) from exc

        import rclpy  # noqa: PLC0415  # reason: ROS-only dep
        from rclpy.executors import SingleThreadedExecutor  # noqa: PLC0415
        from rclpy.node import Node as _Node  # noqa: PLC0415

        self._srv_type: Any = TorqueEnable
        self._info_type: Any = RobotInfo
        # A private node + executor: the e-stop callback runs on the lifecycle
        # node's executor, and a service future can only complete if something
        # else spins the client — so the client lives elsewhere.
        self._helper = _Node(f"{node.get_name()}_torque_stop", context=node.context)
        self._executor = SingleThreadedExecutor(context=node.context)
        self._executor.add_node(self._helper)
        self._rclpy = rclpy
        self._clients: dict[str, Any] = {
            ns: self._helper.create_client(TorqueEnable, f"/{ns}/torque_enable")
            for ns in arm_namespaces
        }
        self._info_clients: dict[str, Any] = {
            ns: self._helper.create_client(RobotInfo, f"/{ns}/get_robot_info")
            for ns in arm_namespaces
        }
        log.info("hal.interbotix_transport.ready", arms=list(self._clients))

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
        """Destroy the helper node (lifecycle cleanup)."""
        self._executor.remove_node(self._helper)
        self._helper.destroy_node()
