"""Make ``LifecycleNode`` transition-callback failures observable.

rclpy's ``LifecycleNodeMixin.__execute_callback`` wraps every transition
callback (``on_configure`` / ``on_activate`` / …) in a bare ``except
Exception`` that returns ``TransitionCallbackReturn.ERROR`` and — per the
literal ``# TODO(ivanpauno): log sth here`` in upstream rclpy — logs
**nothing**. A composing host such as ``openral_rskill_ros``'s ``runtime_node``
then sees only the ``ERROR`` sentinel and reports a bare ``exit code 4``; the
real exception and its traceback are gone, turning a one-line ``ModuleNotFound``
into an opaque crash (CLAUDE.md §1.4 — explicit beats implicit).

:func:`log_lifecycle_errors` is a decorator for those callbacks. It runs the
wrapped callback and, on any uncaught exception, logs the full traceback via the
node's ROS logger (``get_logger()`` → ``/rosout`` → the launch console) and
returns a clean ``TransitionCallbackReturn.FAILURE`` instead of letting the
exception escape into rclpy's silent ``ERROR`` conversion. ``FAILURE`` (not
``ERROR``) keeps the managed node in the well-defined ``unconfigured`` /
``inactive`` state rather than ``errorprocessing``.

The module imports ``rclpy`` lazily (inside the wrapper) so it stays import-safe
on pure-Python hosts, matching :mod:`openral_observability.diagnostics`.
"""

from __future__ import annotations

import functools
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

__all__ = ["log_lifecycle_errors"]

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

        try:
            return callback(self, state)
        except Exception:  # reason: surface rclpy-swallowed transition failures
            logger: Any = self.get_logger()
            logger.error(
                f"{callback.__name__} raised — transition failed:\n{traceback.format_exc()}"
            )
            return TransitionCallbackReturn.FAILURE

    return _wrapper
