"""Dispatch one rSkill goal at a live ``deploy sim`` graph and report the result.

Private helper of ``tools/validation_matrix.py``. Recovered from
``run_xr1_full.py`` in ``spark:~/openral-runs/<round>/scripts/`` and
parameterised over the rSkill and prompt, which the original hardcoded per
round — the drift that made "which skill did round N dispatch?" unanswerable
from the artifacts.

The validation matrix runs with the reasoner **off**, so nothing else would
issue a goal; this is the direct-dispatch surface standing in for it. The
process prints exactly one JSON line, which the harness stores as the run's
``goal.log`` and reads back for ``wall_s`` and ``failure_reason``.

Requires a sourced ROS 2 environment and the built overlay.

Run::

    python tools/_validation_matrix_dispatch.py --deadline-s 420 \\
        --rskill-id OpenRAL/rskill-... --prompt "Pick the baguette ..."
"""

from __future__ import annotations

import argparse
import json
import time


def main(argv: list[str] | None = None) -> int:
    """Send one ``ExecuteRskill`` goal and print its terminal status as JSON.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        ``0`` when the action returned a result, ``1`` on a hard deadline
        overrun with no result.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-s", type=float, default=420.0)
    parser.add_argument("--rskill-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--server-timeout-s",
        type=float,
        default=120.0,
        help="How long to wait for /openral/execute_rskill to appear.",
    )
    args = parser.parse_args(argv)

    import rclpy  # reason: ROS import, deferred so --help works without it
    from openral_msgs.action import ExecuteRskill  # reason: as above
    from rclpy.action import ActionClient  # reason: as above

    rclpy.init()
    node = rclpy.create_node("validation_matrix_dispatch")
    client = ActionClient(node, ExecuteRskill, "/openral/execute_rskill")
    if not client.wait_for_server(timeout_sec=args.server_timeout_s):
        raise RuntimeError("/openral/execute_rskill unavailable")

    latest_chunk = 0
    first_chunk_t: float | None = None
    started = time.monotonic()

    def on_feedback(message: object) -> None:
        nonlocal latest_chunk, first_chunk_t
        latest_chunk = int(message.feedback.chunk_index)  # type: ignore[attr-defined]  # reason: ROS msg
        if first_chunk_t is None:
            first_chunk_t = time.monotonic() - started

    goal = ExecuteRskill.Goal()
    goal.rskill_id = args.rskill_id
    goal.prompt = args.prompt
    goal.deadline_s = args.deadline_s
    send_future = client.send_goal_async(goal, feedback_callback=on_feedback)
    rclpy.spin_until_future_complete(node, send_future)
    goal_handle = send_future.result()
    if goal_handle is None or not goal_handle.accepted:
        raise RuntimeError(f"goal rejected for rskill {args.rskill_id!r}")

    result_future = goal_handle.get_result_async()
    # The action's own deadline plus a teardown margin: a policy that overruns
    # its deadline still has to unwind the graph before a result appears.
    hard_deadline = time.monotonic() + args.deadline_s + 180.0
    while not result_future.done() and time.monotonic() < hard_deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    if not result_future.done():
        print(json.dumps({"latest_chunk": latest_chunk, "status": -1, "success": False}))
        return 1

    response = result_future.result()
    result = response.result
    print(
        json.dumps(
            {
                "latest_chunk": latest_chunk,
                "first_chunk_s": round(first_chunk_t, 2) if first_chunk_t else None,
                "wall_s": round(time.monotonic() - started, 1),
                "status": int(response.status),
                "success": bool(result.success),
                "failure_reason": result.failure_reason,
                "rskill_id": args.rskill_id,
            },
            sort_keys=True,
        )
    )
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
