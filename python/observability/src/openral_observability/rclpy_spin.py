"""Spin an rclpy executor on a worker thread without a teardown traceback.

rclpy's SIGINT handler shuts the context down first. An executor thread already past
its shutdown check then builds its next wait set on the dead context and raises
``ExternalShutdownException`` (or ``RCLError: ... context is not valid``) out of
``executor.spin`` — on a ``threading.Thread`` that prints a traceback on every Ctrl-C
(the Isaac ``deploy sim`` teardown, 2026-10-04). A subscription take racing the shutdown
raises a bare ``RuntimeError`` ("Unable to convert call argument '0' to Python object",
the segmenter, 2026-10-04); ``RCLError`` and ``InvalidHandle`` are ``RuntimeError``
subclasses. That is teardown, not a fault.
"""

from __future__ import annotations

from typing import Any

__all__ = ["spin_executor_until_shutdown", "spin_node_until_shutdown"]


def spin_executor_until_shutdown(executor: Any) -> None:  # reason: rclpy executor is untyped
    """``executor.spin()`` until a signal-driven shutdown, which ends it quietly.

    ``ExternalShutdownException``, ``KeyboardInterrupt`` and a ``RuntimeError`` (which
    covers ``RCLError`` / ``InvalidHandle``) raised after the context went down all
    return normally. A ``RuntimeError`` with the context still up is a real error and
    propagates; so does everything else.

    Meant as ``threading.Thread(target=spin_executor_until_shutdown, args=(executor,))``.
    Same contract as ``openral_hal.lifecycle.spin_until_shutdown`` (a node) and
    ``openral_rskill_ros.compose.spin_until_shutdown`` (an executor); kept per layer,
    this one in the lowest layer so the runner and the dashboard can share it.
    """
    import rclpy
    from rclpy.executors import ExternalShutdownException

    try:
        executor.spin()
    except RuntimeError:
        if rclpy.ok():
            raise
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


def spin_node_until_shutdown(node: Any) -> None:  # reason: rclpy Node is untyped
    """``rclpy.spin(node)`` with the same quiet-teardown contract as the executor form.

    Spins ``node`` on a private ``SingleThreadedExecutor`` through
    ``spin_executor_until_shutdown``; the caller still destroys the node and calls
    ``rclpy.try_shutdown()``.
    """
    from rclpy.executors import SingleThreadedExecutor

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        spin_executor_until_shutdown(executor)
    finally:
        executor.shutdown()
