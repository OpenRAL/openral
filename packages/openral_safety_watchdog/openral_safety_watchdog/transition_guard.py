"""An unavailable lifecycle transition is refused, not fatal: the E-stop sources' copy.

On ROS 2 Jazzy, rclpy raises ``RCLError`` out of the ``change_state`` service callback,
and so out of the executor, when an id-only request names a transition the node's
current state lacks (a second ``activate``): the node dies. For an E-stop source that
takes a brake off the graph. launch_ros ``ChangeState`` events and a hand-typed
``ros2 service call`` send an id alone.

This is a dependency-free copy of ``openral_observability.refuse_unavailable_transitions``
(the deadman, the hardware E-stop bridge and the human E-stop forwarder depend on
``rclpy`` and the IDL only, so a broken observability install cannot keep a brake
source from starting), as ``_param_guard`` is of the kernel's parameter guard.
"""

from __future__ import annotations

from typing import Any

__all__ = ["refuse_unavailable_transitions"]


def refuse_unavailable_transitions(node: Any) -> None:
    """Answer ``success: false`` to an id-only ``change_state`` the current state lacks.

    Call once in the lifecycle node's ``__init__`` (after ``super().__init__``). The
    refusal is logged and the state is left unchanged; labelled requests (rclpy refuses
    an unknown label itself), valid transitions and any other failure inside a
    transition are left to rclpy.

    Args:
        node: The ``rclpy.lifecycle.LifecycleNode`` whose ``change_state`` to guard.

    Example:
        >>> from rclpy.lifecycle import LifecycleNode
        >>> from openral_safety_watchdog.transition_guard import (
        ...     refuse_unavailable_transitions,
        ... )
        >>> class MyNode(LifecycleNode):
        ...     def __init__(self):
        ...         super().__init__("my_node")
        ...         refuse_unavailable_transitions(self)
    """
    # rclpy keeps the service and state machine on private attributes; no public hook.
    service = node._service_change_state
    serve = service.callback

    def _guarded(req: Any, resp: Any) -> Any:
        if not req.transition.label:
            available = {t[0] for t in node._state_machine.available_transitions}
            if req.transition.id not in available:
                node.get_logger().warning(
                    f"change_state refused: transition id {req.transition.id} is not "
                    f"available in state {node._state_machine.current_state[1]!r}"
                )
                resp.success = False
                return resp
        return serve(req, resp)

    service.callback = _guarded
