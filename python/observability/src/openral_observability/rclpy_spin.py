"""Spin an rclpy executor on a worker thread without a teardown traceback.

rclpy's SIGINT handler shuts the context down first. An executor thread already past
its shutdown check then builds its next wait set on the dead context and raises
``ExternalShutdownException`` (or ``RCLError: ... context is not valid``) out of
``executor.spin`` — on a ``threading.Thread`` that prints a traceback on every Ctrl-C
(the Isaac ``deploy sim`` teardown, 2026-10-04). That is teardown, not a fault.
"""

from __future__ import annotations

from typing import Any

__all__ = ["spin_executor_until_shutdown"]


def spin_executor_until_shutdown(executor: Any) -> None:  # reason: rclpy executor is untyped
    """``executor.spin()`` until a signal-driven shutdown, which ends it quietly.

    ``ExternalShutdownException``, ``KeyboardInterrupt`` and an ``RCLError`` raised after
    the context went down all return normally. An ``RCLError`` with the context still up
    is a real error and propagates; so does everything else.

    Meant as ``threading.Thread(target=spin_executor_until_shutdown, args=(executor,))``.
    Same contract as ``openral_hal.lifecycle.spin_until_shutdown`` (a node) and
    ``openral_rskill_ros.compose.spin_until_shutdown`` (an executor); kept per layer,
    this one in the lowest layer so the runner and the dashboard can share it.
    """
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.executors import ExternalShutdownException

    try:
        executor.spin()
    except RCLError:
        if rclpy.ok():
            raise
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
