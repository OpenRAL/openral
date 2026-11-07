"""Stand-in action servers at the ROS graph boundary.

Real ``rclpy.action.ActionServer`` instances of the real IDL types, answering
the way the wrapped servers do (accept, succeed with an empty result) without
the processes behind them: the OpenRAL runner's ``/openral/execute_rskill`` and
Nav2's ``/navigate_to_pose``. Tests hold the returned server and ``destroy()``
it. Lives under ``tests/integration/fakes/`` per CLAUDE.md §1.11 — production
code never imports it.
"""

from __future__ import annotations

from typing import Any


def execute_rskill_server(node: Any, received: list[Any] | None = None) -> Any:
    """Serve ``/openral/execute_rskill``; record each accepted goal in ``received``."""
    from openral_msgs.action import ExecuteRskill
    from rclpy.action import ActionServer
    from rclpy.action.server import GoalResponse

    def _goal(goal: Any) -> Any:
        if received is not None:
            received.append(goal)
        return GoalResponse.ACCEPT

    def _execute(handle: Any) -> Any:
        handle.succeed()
        return ExecuteRskill.Result()

    return ActionServer(
        node, ExecuteRskill, "/openral/execute_rskill", _execute, goal_callback=_goal
    )


def navigate_to_pose_server(node: Any) -> Any:
    """Serve Nav2's ``/navigate_to_pose``; every goal succeeds at once."""
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionServer

    def _execute(handle: Any) -> Any:
        handle.succeed()
        return NavigateToPose.Result()

    return ActionServer(node, NavigateToPose, "/navigate_to_pose", _execute)


def held_navigate_to_pose_server(
    node: Any, outcomes: list[str], *, accept_cancel: bool = True, hold_s: float = 30.0
) -> Any:
    """Serve ``/navigate_to_pose`` holding each goal open, as Nav2 does while driving.

    A goal ends ``canceled`` when a cancel arrives and ``accept_cancel`` is set, else
    ``succeeded`` after ``hold_s``. Each ending is appended to ``outcomes``, so a test sees
    whether the robot would still have been driving.
    """
    import time

    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionServer
    from rclpy.action.server import CancelResponse
    from rclpy.callback_groups import ReentrantCallbackGroup

    def _cancel(_request: Any) -> Any:
        return CancelResponse.ACCEPT if accept_cancel else CancelResponse.REJECT

    def _execute(handle: Any) -> Any:
        end = time.monotonic() + hold_s
        while time.monotonic() < end:
            if handle.is_cancel_requested:
                handle.canceled()
                outcomes.append("canceled")
                return NavigateToPose.Result()
            time.sleep(0.02)
        handle.succeed()
        outcomes.append("succeeded")
        return NavigateToPose.Result()

    return ActionServer(
        node,
        NavigateToPose,
        "/navigate_to_pose",
        _execute,
        cancel_callback=_cancel,
        callback_group=ReentrantCallbackGroup(),
    )
