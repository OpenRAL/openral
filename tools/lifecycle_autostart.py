#!/usr/bin/env python3
"""Drive a ROS 2 LifecycleNode through CONFIGURE → ACTIVATE with retries.

Used by ``packages/openral_rskill_ros/launch/sim_e2e.launch.py`` to
auto-activate ``/openral_slam_toolbox`` once it appears on the graph.
Two reasons we don't use ``ros2 lifecycle set`` directly here:

1. **Discovery race on slow boots.** On a robocasa-kitchen first-boot
   the HAL lifecycle node spends ~30 s spinning up an editable
   robosuite + robocasa install before yielding the rclpy executor
   long enough for ``ros2 lifecycle set`` to find the node. A
   fixed-delay ``TimerAction`` either fires while the node is still
   "Node not found" (exit 1 → launch_ros logs ``[ERROR] [ros2-9]:
   process has died, exit code 1``) or waits too long for fast
   boots.

2. **launch_ros lifecycle_event_manager false-positive on Jazzy.**
   The ``EmitEvent(ChangeState)`` path logs ``[ERROR]
   [launch_ros.utilities.lifecycle_event_manager]: Failed to make
   transition 'TRANSITION_CONFIGURE'`` whenever
   ``response.success=false`` — but slam_toolbox 2.8.4's
   ``on_configure`` returns SUCCESS at
   ``src/slam_toolbox_common.cpp:139`` and the FSM does transition
   to INACTIVE. The bogus response.success is a Jazzy race that we
   can't patch from our tree.

This script uses rclpy directly: it waits ``--service-timeout-s``
seconds for the ``<node>/change_state`` service to appear, then
calls CONFIGURE and ACTIVATE in sequence with the same service
client. Each transition is bounded by ``--transition-timeout-s``,
which must cover the node's slowest ``on_configure`` — a robocasa-
kitchen HAL first-boot blocks its executor for over a minute (MuJoCo +
robosuite import + env.reset, plus a possible ``uv`` build of the
robocasa editable package), and a too-short bound times out
mid-configure and false-fails a transition that was about to succeed.
Exits 0 on success, non-zero only on real failure (service never
appeared, or the FSM rejected the transition with a clear
``success=false`` AND the post-call state didn't actually advance).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import rclpy
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState, GetState

_STATE_TO_TRANSITION = {
    "inactive": [Transition.TRANSITION_CONFIGURE],
    "active": [Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE],
}


def _service_path(node: str, suffix: str) -> str:
    return f"{node.rstrip('/')}/{suffix}"


def _wait_for_service(
    node: Any,
    service_name: str,
    timeout_s: float,
    srv_type: type,
) -> Any:
    deadline = time.monotonic() + timeout_s
    client = node.create_client(srv_type, service_name)
    while time.monotonic() < deadline:
        if client.wait_for_service(timeout_sec=1.0):
            return client
        rclpy.spin_once(node, timeout_sec=0.0)
    msg = f"service {service_name!r} never appeared within {timeout_s:.1f}s"
    raise TimeoutError(msg)


def _read_state(node: Any, target_node: str, get_state_client: Any) -> str:
    del target_node  # used by callers for log context; not needed here
    req = GetState.Request()
    future = get_state_client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    resp = future.result()
    if resp is None:
        return ""
    label: str = resp.current_state.label
    return label


def _drive_transition(
    node: Any,
    target_node: str,
    change_state_client: Any,
    get_state_client: Any,
    transition_id: int,
    transition_label: str,
    transition_timeout_s: float,
) -> None:
    req = ChangeState.Request()
    req.transition.id = transition_id
    future = change_state_client.call_async(req)
    # A lifecycle node services ``change_state`` by running its
    # ``on_<transition>`` callback synchronously on the same (single-
    # threaded) executor, so this future only resolves once that
    # callback returns. A robocasa-kitchen first-boot ``configure`` can
    # block the HAL's executor for well over a minute — MuJoCo +
    # robosuite import, ``env.reset`` building the kitchen, and on a
    # cold/rebuilt env a ``uv`` resolve+build of robocasa that alone
    # logged ~27 s in the wild. The previous hardcoded 30 s spin timed
    # out mid-``configure``: ``future.result()`` returned ``None`` and
    # the immediately-following ``get_state`` read also returned ``''``
    # (the executor was still inside the callback), so a transition that
    # was about to succeed was reported as ``did not advance the FSM``
    # and the process exited 1. Wait the caller-supplied budget instead.
    rclpy.spin_until_future_complete(node, future, timeout_sec=transition_timeout_s)
    resp = future.result()
    # The post-call state is the source of truth, not ``resp.success``:
    # on Jazzy slam_toolbox's first CONFIGURE returns ``success=false``
    # even though the FSM transitions, and a spin that times out exactly
    # at the deadline yields ``resp=None`` even when ``on_configure``
    # finished microseconds later. Poll the state for a short grace
    # window so an in-flight settle isn't misread as a failure.
    grace_deadline = time.monotonic() + 5.0
    while True:
        post_state = _read_state(node, target_node, get_state_client)
        if post_state in {"inactive", "active"}:
            return
        if resp is not None and resp.success:
            return
        if time.monotonic() >= grace_deadline:
            break
        rclpy.spin_once(node, timeout_sec=0.2)
    msg = (
        f"transition {transition_label!r} on {target_node!r} did not advance the "
        f"FSM within {transition_timeout_s:.1f}s (post-call state={post_state!r}, "
        f"response.success={getattr(resp, 'success', None)!r})"
    )
    raise RuntimeError(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node", required=True, help="Target lifecycle node name (e.g. /openral_slam_toolbox)."
    )
    parser.add_argument(
        "--target",
        choices=("inactive", "active"),
        default="active",
        help="Goal state: drive CONFIGURE → INACTIVE, or +ACTIVATE → ACTIVE.",
    )
    parser.add_argument(
        "--service-timeout-s",
        type=float,
        default=30.0,
        help="Seconds to wait for the change_state service to appear.",
    )
    parser.add_argument(
        "--transition-timeout-s",
        type=float,
        default=300.0,
        help=(
            "Seconds to wait for each CONFIGURE / ACTIVATE transition to "
            "complete. Must cover the node's slowest on_configure — a "
            "robocasa-kitchen HAL first-boot (MuJoCo + robosuite import + "
            "env.reset, plus a possible uv rebuild) can exceed a minute."
        ),
    )
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("openral_lifecycle_autostart")
    try:
        change_state_name = _service_path(args.node, "change_state")
        get_state_name = _service_path(args.node, "get_state")
        try:
            change_state_client = _wait_for_service(
                node, change_state_name, args.service_timeout_s, ChangeState
            )
            get_state_client = _wait_for_service(
                node, get_state_name, args.service_timeout_s, GetState
            )
        except TimeoutError as exc:
            print(f"lifecycle-autostart: {exc}", file=sys.stderr)
            return 0  # don't log an [ERROR] from the process; absent server is informational

        current = _read_state(node, args.node, get_state_client)
        transitions = _STATE_TO_TRANSITION[args.target]
        labels = {
            Transition.TRANSITION_CONFIGURE: "configure",
            Transition.TRANSITION_ACTIVATE: "activate",
        }
        for tid in transitions:
            label = labels[tid]
            if current == "active":
                # Already at goal.
                break
            if current == "inactive" and label == "configure":
                continue  # already configured; only need activate
            _drive_transition(
                node,
                args.node,
                change_state_client,
                get_state_client,
                tid,
                label,
                args.transition_timeout_s,
            )
            current = _read_state(node, args.node, get_state_client)
        print(
            f"lifecycle-autostart: {args.node} reached state={current!r} (target={args.target!r})"
        )
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
