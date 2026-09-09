"""A real ros2_control HAL must not publish onto the global ``/joint_states``.

On real hardware the vendor's ``joint_state_broadcaster`` owns that topic: it is
the only thing with access to ``controller_manager``'s in-process hardware state
handles, and it drives them at the controller rate. ``_attach_ros_control_transport``
says so and tries to enforce it by destroying this node's global publisher —
but it runs in ``on_configure``, and the publishers are created in
``on_activate``. The destroy therefore landed on ``None``, activation recreated
the publisher it had just logged dropping, and every real deploy ran with this
node publishing its own 16-DoF state at 30 Hz onto the topic the broadcaster was
driving at 750 Hz.

Observed on the OpenArm cell: ``/joint_states`` carried two publishers, and the
HAL's contribution was all zeros for all sixteen joints — because its transport
reads that same topic, so it was partly reading back its own output. Everything
downstream of ``/joint_states`` (``robot_state_publisher`` and therefore ``/tf``,
the world_state aggregator, the safety kernel's state deadline) saw those zeros
interleaved with the real arm.

Nothing covered the *post-activation* state, which is why it shipped: asserting
after ``configure`` would have passed. These tests assert after ``activate``.

Per CLAUDE.md §1.11: real ``rclpy``, real manifest, real ``OpenArmRealHAL``.
``connect()`` preflights the CAN links, so this needs the two buses *up* — but
not a motor answering, since the publisher wiring under test does not depend on
a reply. Skipped where those interfaces are absent.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import pytest

_OPENARM_YAML = Path(__file__).resolve().parents[2] / "robots" / "openarm" / "robot.yaml"

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO")) and (
    importlib.util.find_spec("openral_msgs") is not None
)

#: `OpenArmRealHAL.connect` refuses a bus that is missing or down, so these
#: tests need the arm cell's two CAN links present. Names match the manifest's
#: `hal.parameters.defaults` (the udev names OpenArm provisioning assigns).
_CAN_LINKS_UP = all(
    (Path("/sys/class/net") / name / "operstate").exists()
    for name in ("openarm_left", "openarm_right")
)

pytestmark = [
    pytest.mark.skipif(
        not _ROS2_AVAILABLE,
        reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
    ),
    pytest.mark.skipif(
        not _CAN_LINKS_UP,
        reason="openarm_left/openarm_right CAN links absent — real HAL connect() would refuse.",
    ),
]


@contextmanager
def _real_mode_node() -> Iterator[Any]:
    """Configure + activate the generic manifest node in ``hal_mode=real``."""
    import rclpy
    from openral_hal.lifecycle import ManifestHALLifecycleNode
    from rclpy.lifecycle import TransitionCallbackReturn

    # One context per process: rclpy refuses a second `init`, so each test
    # cannot own one.
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    node = ManifestHALLifecycleNode("openral_hal_openarm")
    node.set_parameters(
        [
            rclpy.parameter.Parameter("robot_yaml", value=str(_OPENARM_YAML)),
            rclpy.parameter.Parameter("hal_mode", value="real"),
        ],
    )
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
        assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
        yield node
    finally:
        with suppress(Exception):
            node.trigger_deactivate()
        with suppress(Exception):
            node.trigger_cleanup()
        with suppress(Exception):
            node.destroy_node()
        if owns_context:
            with suppress(Exception):
                rclpy.shutdown()


def test_no_global_joint_states_publisher_after_activation() -> None:
    """The regression itself: assert *after* activate, not after configure.

    `_attach_ros_control_transport` runs at configure time, so a check made
    there passes while the node goes on to create the publisher at activate.
    """
    with _real_mode_node() as node:
        assert node._ros_control_transport is not None, (
            "hal_mode=real must attach a RosControlTransport; without it this "
            "test would pass for the wrong reason"
        )
        assert node._joint_state_pub is None, (
            "the vendor joint_state_broadcaster owns /joint_states on real "
            "hardware; this node must not publish onto it"
        )


def test_the_node_still_publishes_its_namespaced_joint_states() -> None:
    """The namespaced topic collides with nothing and stays.

    Without this, "drop the global publisher" could be satisfied by dropping
    both, silently removing the view CLI consumers read.
    """
    with _real_mode_node() as node:
        assert node._publisher is not None
        topic = node._publisher.topic_name
        assert topic.endswith("/joint_states")
        assert topic != "/joint_states", "must be the namespaced topic, not the global one"


def test_no_publisher_is_advertised_on_the_global_topic() -> None:
    """Check the ROS graph, not just the attribute.

    The attribute is what the fix sets, so asserting only on it would pass for
    a node that advertised the topic by some other path. This is the property
    that actually mattered on the cell: publisher count on `/joint_states`.
    """
    with _real_mode_node() as node:
        advertised = [
            name
            for name, _types in node.get_publisher_names_and_types_by_node(
                node.get_name(), node.get_namespace()
            )
        ]
        assert "/joint_states" not in advertised, (
            f"node advertises /joint_states; publishers seen: {advertised}"
        )
