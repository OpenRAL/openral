#!/usr/bin/env python3
"""Wait for a ROS 2 action server to appear, then fire skill_registry_changed.

The reasoner seeds its rSkill palette at ``on_configure`` (early launch
autostart); wrapped-ROS rSkills (``kind: ros_action``/``ros_service``) drop
from the palette when their ``interface_name`` isn't yet advertised — but
Nav2's lifecycle dance takes 15-30s, so ``/navigate_to_pose`` appears long
after the palette is frozen. The reasoner re-seeds on an ``Empty`` message
on ``/openral/skill_registry_changed``
(``_maybe_seed_palette_from_search_paths``); this script polls
``get_action_names_and_types()`` until the action appears, then publishes
the trigger.

Usage::

    python3 wait_for_action_and_signal_palette.py \\
        --action /navigate_to_pose \\
        --timeout-s 60.0

Exits 0 once the trigger is published; exits 1 if the action never appears
within ``--timeout-s``.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import rclpy
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Empty


def _action_on_graph(node: Any, action_name: str) -> bool:
    """Return True when ``action_name`` is advertised on the ROS graph.

    ``get_action_names_and_types`` is a free function in ``rclpy.action``,
    not a ``Node`` method (which only knows topics + services). Matches the
    exact name to mirror the rSkill manifest's ``interface_name`` field.
    """
    from rclpy.action import get_action_names_and_types

    seen = get_action_names_and_types(node)
    return any(name == action_name for name, _types in seen)


def _lifecycle_active(node: Any, lifecycle_node_name: str, timeout_s: float = 30.0) -> bool:
    """Block until ``<lifecycle_node_name>/get_state`` returns ACTIVE (=3).

    A managed action server (Nav2 bt_navigator, MoveIt move_group, etc.)
    only accepts goals once ACTIVE — graph discoverability alone isn't
    enough. ``lifecycle_msgs/msg/State.PRIMARY_STATE_ACTIVE`` is 3. Returns
    False on timeout.
    """
    from lifecycle_msgs.srv import GetState

    service_name = f"{lifecycle_node_name}/get_state"
    client = node.create_client(GetState, service_name)
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if client.wait_for_service(timeout_sec=0.5):
                break
            rclpy.spin_once(node, timeout_sec=0.0)
        else:
            return False
        while time.monotonic() < deadline:
            future = client.call_async(GetState.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=1.0)
            if future.done():
                response = future.result()
                if response is not None and response.current_state.id == 3:
                    return True
            time.sleep(0.5)
        return False
    finally:
        node.destroy_client(client)


def _publish_signal_qos() -> QoSProfile:
    """QoS that matches the reasoner's ``/openral/skill_registry_changed`` sub.

    Per ``reasoner_node.py``: RELIABLE + TRANSIENT_LOCAL + KEEP_LAST=1, so a
    late subscriber still gets the latest value via the durability handshake.
    """
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        required=True,
        help="Full action name to wait for (e.g. /navigate_to_pose).",
    )
    parser.add_argument(
        "--signal-topic",
        default="/openral/skill_registry_changed",
        help="Topic to publish std_msgs/Empty on once the action is on the graph.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="Seconds to wait for the action server before giving up.",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=1.0,
        help="Seconds between graph-discovery polls.",
    )
    parser.add_argument(
        "--lifecycle-node",
        default="",
        help=(
            "Optional fully-qualified lifecycle node name "
            "(e.g. /bt_navigator). When set, the helper additionally waits "
            "for this node's get_state service to return ACTIVE before "
            "publishing — necessary for Nav2 / MoveIt managed action "
            "servers, which are discoverable on the graph during "
            "Configure but only accept goals once Active."
        ),
    )
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("openral_palette_reseed_signal")
    publisher = node.create_publisher(Empty, args.signal_topic, _publish_signal_qos())
    try:
        deadline = time.monotonic() + args.timeout_s
        while time.monotonic() < deadline:
            if _action_on_graph(node, args.action):
                if args.lifecycle_node and not _lifecycle_active(node, args.lifecycle_node):
                    print(
                        f"palette-reseed-signal: {args.action!r} on graph but "
                        f"{args.lifecycle_node!r} did not reach ACTIVE within "
                        "30s; publishing anyway.",
                        file=sys.stderr,
                    )
                # Tiny grace period: action server's lifecycle bond may
                # be a few hundred ms behind discovery. Sleep once so
                # the next ``send_goal_async`` doesn't race the bond.
                time.sleep(0.5)
                # Wait for a subscriber: publish-then-exit before one attaches
                # means the reasoner (subscribed at on_configure) never sees it.
                sub_deadline = time.monotonic() + 5.0
                while publisher.get_subscription_count() == 0 and time.monotonic() < sub_deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
                if publisher.get_subscription_count() == 0:
                    print(
                        f"palette-reseed-signal: no subscriber on "
                        f"{args.signal_topic!r} after 5 s; publishing "
                        "anyway (subscriber may catch via TRANSIENT_LOCAL).",
                        file=sys.stderr,
                    )
                publisher.publish(Empty())
                # Drain repeatedly so the wire actually carries the
                # message before the process exits. 1 second is
                # generous but cheap on this one-shot path.
                drain_deadline = time.monotonic() + 1.0
                while time.monotonic() < drain_deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
                print(
                    f"palette-reseed-signal: {args.action!r} discovered; "
                    f"published Empty on {args.signal_topic!r} "
                    f"({publisher.get_subscription_count()} subscriber(s)).",
                )
                return 0
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(args.poll_interval_s)
        print(
            f"palette-reseed-signal: {args.action!r} did not appear within "
            f"{args.timeout_s:.1f}s; reasoner palette will not be re-seeded.",
            file=sys.stderr,
        )
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
