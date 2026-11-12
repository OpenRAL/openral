"""``openral prompt`` CLI adapter.

Publishes a one-shot ``openral_msgs/PromptStamped`` onto
``/openral/prompt_in/cli`` and exits. ``prompt_router_node``
(``packages/openral_prompt_router``) fans it out onto ``/openral/prompt``
after stamping ``{"source": "cli", "priority": 100}`` onto
``metadata_json``; the F4 reasoner consumes ``/openral/prompt``.

Wire shape:

* QoS: ``RELIABLE + VOLATILE + KEEP_LAST=10`` (matches the router's
  subscription so a one-shot publish survives even a late-subscribing router).
* ``metadata_json``: ``{"source_cli": true}``, plus ``"new_goal": true`` under
  ``--new-goal`` (see ``openral_reasoner.node_policy.should_rebuild_mission``);
  the router appends ``source`` / ``priority``.

``rclpy`` is imported lazily inside the command so ``openral --help`` stays
sub-second when ROS is not sourced.
"""

from __future__ import annotations

import json
import os
import time

import typer

from openral_cli._dds_scope import (
    ROBOT_OPTION_HELP,
    join_robot_scope_or_exit,
    require_robot_on_graph,
)

__all__ = ["prompt_command"]


def prompt_command(
    text: str = typer.Argument(..., help='Prompt text, e.g. "pick the red cube"'),
    topic: str = typer.Option(
        "/openral/prompt_in/cli",
        help="ROS topic to publish on; defaults to the prompt-router's CLI input.",
    ),
    wait_s: float = typer.Option(
        1.0,
        help=(
            "Seconds to wait after publish before exiting. Gives the prompt-router "
            "time to receive on a fresh DDS discovery; 0 exits immediately."
        ),
    ),
    discovery_wait_s: float = typer.Option(
        5.0,
        help=(
            "Seconds to spin waiting for the prompt-router to discover the publisher "
            "before giving up and publishing anyway. Cold-boot deploys with stale "
            "Fast-DDS shared-memory remnants need 10-15 s; the default 5 s matches "
            "the interactive use case where the launch has been up for a while."
        ),
    ),
    new_goal: bool = typer.Option(
        False,
        "--new-goal",
        help=(
            "Mark this prompt as a NEW goal that replaces the reasoner's current "
            "mission. Without it, a prompt sent while a mission is in progress is "
            "treated as conversational context (an answer to a reasoner question, "
            "a hint) and does NOT rebuild the task queue."
        ),
    ),
    robot: str | None = typer.Option(None, "--robot", help=ROBOT_OPTION_HELP),
) -> None:
    """Publish a one-shot PromptStamped on the prompt-router CLI input topic.

    Requires a sourced ROS 2 install. Exits 0 on success, 2 when
    ``rclpy`` / ``openral_msgs`` are not importable (after pointing
    the user at ``just ros2-build`` + ``source install/setup.bash``) or the
    ROS scope is networked without the opt-in, 3 when ``--robot``'s HAL node
    is not on the graph (nothing published). Prints the ROS domain and range
    it used first.

    Example::

        openral prompt --robot openarm "pick the red cube"
    """
    join_robot_scope_or_exit("openral prompt", robot_id=robot)  # before rclpy.init() (#557)
    try:
        import rclpy  # noqa: PLC0415  # reason: heavy ROS dep deferred
        from openral_msgs.msg import (  # noqa: PLC0415  # reason: heavy ROS dep deferred
            PromptStamped,
        )
        from rclpy.qos import (  # noqa: PLC0415
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
    except ImportError as exc:
        typer.echo(
            f"openral prompt: cannot import rclpy / openral_msgs ({exc!s}). "
            "Run `just ros2-build` then `source install/setup.bash` first.",
            err=True,
        )
        raise typer.Exit(code=2) from exc

    from openral_core.ros_security import (  # noqa: PLC0415  # reason: lazy CLI import
        default_operator_enclave,
    )

    # Under SROS2 only the operator enclave may make this call (ADR-0130).
    default_operator_enclave(os.environ)
    rclpy.init()
    try:
        node = rclpy.create_node("openral_cli_prompt")
        require_robot_on_graph("openral prompt", node, robot_id=robot, timeout_s=discovery_wait_s)
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        pub = node.create_publisher(PromptStamped, topic, qos)

        msg = PromptStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "openral_cli_prompt"
        msg.text = text
        metadata = {"source_cli": True} | ({"new_goal": True} if new_goal else {})
        msg.metadata_json = json.dumps(metadata, sort_keys=True)

        # Spin until a subscriber matches so the one-shot publish isn't lost.
        # SHM can fall back to UDP discovery (`RTPS_TRANSPORT_SHM Error:
        # Failed init_port`, ~1 s added); 5 s default covers normal
        # interactive use, cold-boot deploys can need 10-15 s
        # (`--discovery-wait-s`).
        deadline = time.monotonic() + discovery_wait_s
        while time.monotonic() < deadline and pub.get_subscription_count() == 0:
            rclpy.spin_once(node, timeout_sec=0.05)
        n_subs = pub.get_subscription_count()
        if n_subs == 0:
            typer.echo(
                f"openral prompt: no subscriber on {topic} after {discovery_wait_s:.1f} s; "
                "is the launch up on ROS_DOMAIN_ID="
                f"{os.environ.get('ROS_DOMAIN_ID', '').strip() or '0'}"
                f"{'' if robot else ' (a real deploy: pass --robot <id>)'}?",
                err=True,
            )
        pub.publish(msg)
        typer.echo(
            f"openral prompt: published on {topic} text={text!r} (matched_subs={n_subs})",
        )

        if wait_s > 0:
            time.sleep(wait_s)
        node.destroy_node()
    finally:
        rclpy.shutdown()
