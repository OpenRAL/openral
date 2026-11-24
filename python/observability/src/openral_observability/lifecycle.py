"""Make ``LifecycleNode`` transitions observable — failures *and* duration.

rclpy's ``LifecycleNodeMixin.__execute_callback`` wraps every transition
callback (``on_configure`` / ``on_activate`` / …) in a bare ``except
Exception`` that returns ``TransitionCallbackReturn.ERROR`` and — per the
literal ``# TODO(ivanpauno): log sth here`` in upstream rclpy — logs
**nothing**. A composing host such as ``openral_rskill_ros``'s
``runtime_node`` then sees only the ``ERROR`` sentinel and reports a bare
``exit code 4``; the real exception/traceback is gone, turning a one-line
``ModuleNotFound`` into an opaque crash (CLAUDE.md §1.4).

``log_lifecycle_errors`` decorates those callbacks: on an uncaught
exception it logs the full traceback via the node's ROS logger
(``get_logger()`` → ``/rosout`` → launch console) and returns a clean
``TransitionCallbackReturn.FAILURE`` instead of letting rclpy's silent
``ERROR`` conversion swallow it. ``FAILURE`` (not ``ERROR``) keeps the node
in the well-defined ``unconfigured``/``inactive`` state rather than
``errorprocessing``.

``refuse_unavailable_transitions`` keeps a node alive when its ``change_state``
service is asked for a transition its current state lacks (a repeated ``deactivate``).
rclpy (Jazzy) refuses an unknown transition *label* cleanly, but an unknown transition
*id* raises ``RCLError`` out of the service callback and the executor, killing the node.
The reasoner's lifecycle dispatch and ``tools/lifecycle_autostart.py`` send the label
too, but launch_ros ``ChangeState`` events and a hand-typed ``ros2 service call`` send an
id alone, so the guard lives on the server side, once per node, whoever the client is.

Imports ``rclpy`` lazily (inside the wrapper) so the module stays
import-safe on pure-Python hosts, matching
``openral_observability.diagnostics``.
"""

from __future__ import annotations

import contextlib
import functools
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

__all__ = ["log_lifecycle_errors", "refuse_unavailable_transitions"]

_Node = TypeVar("_Node", bound="LifecycleNode")


def log_lifecycle_errors(
    callback: Callable[[_Node, LifecycleState], TransitionCallbackReturn],
) -> Callable[[_Node, LifecycleState], TransitionCallbackReturn]:
    """Wrap a lifecycle transition callback so exceptions are logged, not swallowed.

    Decorate ``on_configure`` / ``on_activate`` / ``on_cleanup`` / … methods of
    an ``rclpy.lifecycle.LifecycleNode``. On a clean return the decorator is
    transparent. On an uncaught exception it logs the callback name and the full
    traceback through ``self.get_logger().error(...)`` and returns
    ``TransitionCallbackReturn.FAILURE``, so the failure surfaces on the launch
    console instead of being reduced to rclpy's silent ``ERROR`` sentinel.

    Args:
        callback: A bound-method-shaped transition callback
            ``(self, state) -> TransitionCallbackReturn``.

    Returns:
        The wrapped callback with the same signature.

    Example:
        >>> from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
        >>> from openral_observability import log_lifecycle_errors
        >>> class MyNode(LifecycleNode):
        ...     @log_lifecycle_errors
        ...     def on_configure(self, state):
        ...         raise RuntimeError("missing weights")  # logged, returns FAILURE
    """

    @functools.wraps(callback)
    def _wrapper(self: _Node, state: LifecycleState) -> TransitionCallbackReturn:
        from rclpy.lifecycle import TransitionCallbackReturn

        from openral_observability import semconv

        tracer = trace.get_tracer("openral.lifecycle")
        node_name = "unknown"
        with contextlib.suppress(Exception):
            node_name = str(self.get_name())
        with tracer.start_as_current_span(
            semconv.SPAN_DEPLOY_BRINGUP,
            attributes={
                semconv.BRINGUP_NODE: node_name,
                semconv.BRINGUP_TRANSITION: callback.__name__,
            },
        ) as span:
            try:
                result = callback(self, state)
            except Exception:  # reason: surface rclpy-swallowed transition failures
                logger: Any = self.get_logger()
                logger.error(
                    f"{callback.__name__} raised — transition failed:\n{traceback.format_exc()}"
                )
                span.set_attribute(semconv.BRINGUP_RESULT, "FAILURE")
                span.set_status(Status(StatusCode.ERROR))
                return TransitionCallbackReturn.FAILURE
            span.set_attribute(semconv.BRINGUP_RESULT, str(getattr(result, "name", result)))
            if result != TransitionCallbackReturn.SUCCESS:
                # A clean FAILURE return is still a failed bringup; without
                # this only the raising path would show red on the dashboard.
                span.set_status(Status(StatusCode.ERROR))
            return result

    return _wrapper


def refuse_unavailable_transitions(node: LifecycleNode) -> None:
    """Answer ``success: false`` to a ``change_state`` the current state cannot take.

    Call once in a ``LifecycleNode``'s ``__init__`` (after ``super().__init__``). A
    transition id not registered for the current state — a second ``deactivate``, an
    ``activate`` before ``configure`` — is logged as a warning and refused, the state is
    left unchanged and the node keeps serving, exactly as rclpy already answers an unknown
    transition label. Valid transitions, labelled requests and any other failure inside a
    transition are left to rclpy unchanged.

    Args:
        node: The lifecycle node whose ``change_state`` service to guard.

    Example:
        >>> from rclpy.lifecycle import LifecycleNode
        >>> from openral_observability import refuse_unavailable_transitions
        >>> class MyNode(LifecycleNode):
        ...     def __init__(self):
        ...         super().__init__("my_node")
        ...         refuse_unavailable_transitions(self)
    """
    # reason: rclpy keeps the service and state machine on private attributes; there is
    # no public hook.
    service: Any = node._service_change_state
    serve = service.callback

    def _guarded(req: Any, resp: Any) -> Any:
        # Only an id-only request needs the check: rclpy resolves a label itself and refuses
        # an unknown one. Any other failure inside the transition still raises.
        if not req.transition.label:
            available = {t[0] for t in node._state_machine.available_transitions}
            if req.transition.id not in available:
                logger: Any = node.get_logger()
                logger.warning(
                    f"change_state refused: transition id {req.transition.id} is not "
                    f"available in state {node._state_machine.current_state[1]!r}"
                )
                resp.success = False
                return resp
        return serve(req, resp)

    service.callback = _guarded
