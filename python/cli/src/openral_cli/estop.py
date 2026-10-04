"""``openral estop reset`` — clear every e-stop latch, kernel first.

An e-stop latches in three places: the safety kernel (cleared only by its
``/openral/estop_reset`` ``std_srvs/Trigger`` service, which enforces the
post-estop cooldown) and, independently, every HAL and the rSkill runner
(cleared only by an ``std_msgs/Empty`` on ``/openral/estop_cleared``). Calling
the kernel service alone leaves the runner latched, and it then rejects every
goal. This command performs the same sequence as the dashboard's
``POST /api/estop_reset``:

1. call ``/openral/estop_reset``;
2. only if the kernel answered ``success=true``, broadcast
   ``/openral/estop_cleared`` so the HAL + runner un-latch.

A refused or unanswered kernel reset never broadcasts the clear: the HAL and
runner must not resume while the kernel still holds its latch.

``rclpy`` is imported lazily inside the command so ``openral --help`` stays
sub-second when ROS is not sourced.
"""

from __future__ import annotations

import time

import typer

__all__ = ["CLEARED_TOPIC", "RESET_SERVICE", "estop_app"]

RESET_SERVICE = "/openral/estop_reset"
CLEARED_TOPIC = "/openral/estop_cleared"

estop_app = typer.Typer(
    name="estop",
    help="Safety e-stop operator commands. Requires a sourced ROS 2 install.",
    no_args_is_help=True,
)


@estop_app.command("reset")
def estop_reset_command(
    service_timeout_s: float = typer.Option(
        15.0,
        help="Seconds to wait for the kernel's reset service to appear and answer.",
    ),
    discovery_wait_s: float = typer.Option(
        5.0,
        help=(
            "Seconds to wait for /openral/estop_cleared subscribers (HAL, runner) to "
            "match before broadcasting anyway."
        ),
    ),
) -> None:
    """Reset the safety kernel, then broadcast /openral/estop_cleared.

    Exit codes: 0 every latch cleared; 1 the kernel refused the reset (e.g.
    inside its cooldown — retry); 2 ``rclpy`` not importable; 3 the kernel's
    reset service never answered. On 1 and 3 nothing is broadcast.

    Example::

        openral estop reset
    """
    try:
        import rclpy  # noqa: PLC0415  # reason: heavy ROS dep deferred
        from rclpy.qos import (  # noqa: PLC0415  # reason: heavy ROS dep deferred
            QoSDurabilityPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
        from std_msgs.msg import Empty  # noqa: PLC0415  # reason: heavy ROS dep deferred
        from std_srvs.srv import Trigger  # noqa: PLC0415  # reason: heavy ROS dep deferred
    except ImportError as exc:
        typer.echo(
            f"openral estop reset: cannot import rclpy ({exc!s}). "
            "Source /opt/ros/<distro>/setup.bash and the workspace install first.",
            err=True,
        )
        raise typer.Exit(code=2) from exc

    rclpy.init()
    try:
        node = rclpy.create_node("openral_cli_estop_reset")
        client = node.create_client(Trigger, RESET_SERVICE)
        if not client.wait_for_service(timeout_sec=service_timeout_s):
            typer.echo(
                f"openral estop reset: {RESET_SERVICE} not offered after "
                f"{service_timeout_s:.1f} s — is the safety kernel running? Nothing cleared.",
                err=True,
            )
            raise typer.Exit(code=3)
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=service_timeout_s)
        response = future.result() if future.done() else None
        if response is None:
            typer.echo(
                f"openral estop reset: {RESET_SERVICE} did not answer within "
                f"{service_timeout_s:.1f} s. Nothing cleared.",
                err=True,
            )
            raise typer.Exit(code=3)
        if not response.success:
            typer.echo(
                f"openral estop reset: kernel REFUSED the reset ({response.message!r}); "
                f"{CLEARED_TOPIC} NOT broadcast — HAL and runner stay latched. Retry "
                "once the cause is gone / the cooldown has passed.",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"openral estop reset: kernel latch cleared ({response.message!r})")

        # Same QoS as the HAL/runner subscriptions (safety class), so it matches.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        pub = node.create_publisher(Empty, CLEARED_TOPIC, qos)
        deadline = time.monotonic() + discovery_wait_s
        while time.monotonic() < deadline and pub.get_subscription_count() == 0:
            rclpy.spin_once(node, timeout_sec=0.05)
        n_subs = pub.get_subscription_count()
        if n_subs == 0:
            typer.echo(
                f"openral estop reset: no subscriber on {CLEARED_TOPIC} after "
                f"{discovery_wait_s:.1f} s; broadcasting anyway — a HAL or runner that "
                "is not up yet will still be latched when it starts.",
                err=True,
            )
        # A few copies: cheap insurance against one lost datagram; the clear
        # is idempotent on every subscriber.
        for _ in range(3):
            pub.publish(Empty())
        # Give the RELIABLE writer a moment to deliver before the node dies.
        flush_deadline = time.monotonic() + 0.5
        while time.monotonic() < flush_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        typer.echo(
            f"openral estop reset: broadcast {CLEARED_TOPIC} (matched_subs={n_subs}); "
            "HAL and runner accept new goals."
        )
        node.destroy_node()
    finally:
        rclpy.try_shutdown()
