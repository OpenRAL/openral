"""``openral prompt`` CLI adapter.

Publishes a one-shot ``openral_msgs/PromptStamped`` onto
``/openral/prompt_in/cli`` and exits. The ``prompt_router_node``
(``packages/openral_prompt_router``) fans the message out onto
``/openral/prompt`` after stamping ``{"source": "cli", "priority": 100}``
onto ``metadata_json``; the F4 reasoner consumes ``/openral/prompt``.

Wire shape:

* QoS: ``RELIABLE + VOLATILE + KEEP_LAST=10`` (matches the router's
  subscription so the message survives a one-shot publish-and-exit
  even if the router was a hair late to subscribe).
* ``metadata_json``: ``{"source_cli": true}`` (plus ``"new_goal": true``
  under ``--new-goal`` — the reasoner's mission-replacement escape hatch,
  see ``openral_reasoner.node_policy.should_rebuild_mission``); the router
  appends the canonical ``source`` / ``priority`` fields.

``rclpy`` is imported lazily inside the typer command so ``openral --help``
stays sub-second even when ROS is not sourced.
"""

from __future__ import annotations

import json
import time

import typer

__all__ = ["build_prompt_metadata", "prompt_command"]


def build_prompt_metadata(*, new_goal: bool) -> str:
    """Build the CLI prompt's ``metadata_json`` payload.

    ``"new_goal": true`` is the reasoner's mission-replacement escape hatch:
    :func:`openral_reasoner.node_policy.should_rebuild_mission` requires it to
    replace a STARTED mission (without it a mid-mission prompt is
    conversational context, never a queue rebuild). The prompt-router
    preserves inbound keys when stamping ``source`` / ``priority``.

    Example:
        >>> build_prompt_metadata(new_goal=False)
        '{"source_cli": true}'
        >>> build_prompt_metadata(new_goal=True)
        '{"new_goal": true, "source_cli": true}'
    """
    metadata: dict[str, bool] = {"source_cli": True}
    if new_goal:
        metadata["new_goal"] = True
    return json.dumps(metadata, sort_keys=True)


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
) -> None:
    """Publish a one-shot PromptStamped on the prompt-router CLI input topic.

    Requires a sourced ROS 2 install. Exits 0 on success, 2 when
    ``rclpy`` / ``openral_msgs`` are not importable (after pointing
    the user at ``just ros2-build`` + ``source install/setup.bash``).

    Example::

        openral prompt "pick the red cube"
    """
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

    rclpy.init()
    try:
        node = rclpy.create_node("openral_cli_prompt")
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
        msg.metadata_json = build_prompt_metadata(new_goal=new_goal)

        # Spin until a subscriber matches so the one-shot publish is
        # not delivered into the void. 0.5 s was too tight on hosts
        # where the shared-memory transport falls back to UDP
        # discovery (we saw `RTPS_TRANSPORT_SHM Error: Failed init_port`
        # on this box, which adds ~1 s before UDP picks up the
        # prompt_router subscription). 5 s covers normal interactive
        # use; cold-boot deploys (where the prompt is sent within the
        # same shell session as the launch start) can need 10-15 s,
        # exposed via ``--discovery-wait-s``.
        deadline = time.monotonic() + discovery_wait_s
        while time.monotonic() < deadline and pub.get_subscription_count() == 0:
            rclpy.spin_once(node, timeout_sec=0.05)
        n_subs = pub.get_subscription_count()
        if n_subs == 0:
            typer.echo(
                f"openral prompt: no subscriber on {topic} after {discovery_wait_s:.1f} s; "
                "is the launch up?",
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
