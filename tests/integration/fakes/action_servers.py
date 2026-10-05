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
