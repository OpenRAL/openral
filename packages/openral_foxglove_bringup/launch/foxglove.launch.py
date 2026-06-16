#!/usr/bin/env python3
"""PROTOTYPE — stand-alone launch for upstream ``foxglove_bridge``.

Brings up ``foxglove_bridge`` as a **read-only** live visualisation
surface for OpenRAL's "Bucket-1" topics (the data that Foxglove renders
natively with no custom extension): camera images, the ``/map`` occupancy
grid, the octomap point cloud, joint states, TF, and ``/robot_description``.

Two deliberate safety choices distinguish this from the upstream
``foxglove_bridge_launch.xml`` defaults (CLAUDE.md §1.1 / §3 "Safety"):

1. **Loopback by default.** ``address`` defaults to ``127.0.0.1`` (the
   upstream default is ``0.0.0.0``), matching the dashboard's loopback-only
   posture (issue #44). A viewer on another host must opt in explicitly.
2. **Read-only capabilities.** ``capabilities`` is restricted to
   ``[connectionGraph, assets]`` — the upstream default additionally
   advertises ``clientPublish`` and ``services``, which would let a Foxglove
   client *publish topics and call services* (e.g. trigger an E-stop reset
   or inject an action). This surface MUST NOT be able to actuate the robot.
   Re-enabling those capabilities is a safety-WG decision, not a flag flip.

The ``topic_whitelist`` is an explicit allowlist: anything not matched —
including ``/openral/estop``, ``/openral/safe_action``, the failure bus —
is invisible to the bridge. This is the feasibility spike from
``docs/investigations/foxglove-dashboard-port-feasibility.md``; graduating
it past prototype requires the ADR named there.

Run:
    ros2 launch openral_foxglove_bringup foxglove.launch.py

Then open https://app.foxglove.dev (or the desktop app), choose
"Open connection → Foxglove WebSocket → ws://localhost:8765", and import
``config/openral_layout.json``.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from openral_foxglove_bringup.topics import (
    BUCKET1_TOPIC_WHITELIST,
    READ_ONLY_CAPABILITIES,
)


def generate_launch_description() -> LaunchDescription:
    """Read-only ``foxglove_bridge`` bring-up for the Bucket-1 topics."""
    args = [
        DeclareLaunchArgument(
            "address",
            default_value="127.0.0.1",
            description=(
                "Bind address. Defaults to loopback (issue #44). Set to "
                "0.0.0.0 ONLY for a trusted LAN; never on an untrusted "
                "network — the bridge has no auth."
            ),
        ),
        DeclareLaunchArgument(
            "port",
            default_value="8765",
            description="Foxglove WebSocket port (ws://<address>:<port>).",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description=(
                "Set true when a /clock is published (deploy-sim, ADR-0048) "
                "so Foxglove timestamps align with sim time."
            ),
        ),
        DeclareLaunchArgument(
            "expose_all_topics",
            default_value="false",
            description=(
                "ESCAPE HATCH (debug only). When true, drops the Bucket-1 "
                "allowlist and exposes every topic read-only. Still does NOT "
                "re-enable clientPublish/services. Leave false in any shared "
                "or robot-connected run."
            ),
        ),
        # --- /tf + robot-model rendering ----------------------------------
        # deploy-sim publishes /joint_states but NOT dynamic /tf (no
        # robot_state_publisher in its graph), so Foxglove's 3D panel has no
        # frames and can't draw the robot. Opt in to a robot_state_publisher
        # that turns /joint_states + a URDF into /tf + /tf_static +
        # /robot_description — all already Bucket-1-whitelisted, so the model
        # renders read-only with no other change.
        DeclareLaunchArgument(
            "with_robot_state_publisher",
            default_value="false",
            description=(
                "Run a robot_state_publisher that converts /joint_states + the "
                "``robot_description_urdf`` into /tf + /robot_description so the "
                "3D panel can draw the robot. Requires ``robot_description_urdf``."
            ),
        ),
        DeclareLaunchArgument(
            "with_joint_state_publisher",
            default_value="false",
            description=(
                "Also run a joint_state_publisher (zeros) so the model renders "
                "standalone WITHOUT a sim. Leave false under deploy-sim, which "
                "is the real /joint_states source — two publishers would fight."
            ),
        ),
        DeclareLaunchArgument(
            "robot_description_urdf",
            default_value="",
            description=(
                "Filesystem path to a URDF for the state publishers. Resolve a "
                "manifest robot's URDF with: ``python -c 'from "
                "openral_core.urdf_resolve import resolve_urdf_path; "
                "print(resolve_urdf_path(open(\"robots/<id>/robot.yaml\")...))'`` "
                "or directly via robot_descriptions (e.g. panda_description). "
                "NOTE: openarm has no local URDF (ADR-0027)."
            ),
        ),
    ]

    address = LaunchConfiguration("address")
    port = LaunchConfiguration("port")
    use_sim_time = LaunchConfiguration("use_sim_time")
    expose_all = LaunchConfiguration("expose_all_topics")
    with_rsp = LaunchConfiguration("with_robot_state_publisher")
    with_jsp = LaunchConfiguration("with_joint_state_publisher")
    urdf_path = LaunchConfiguration("robot_description_urdf")

    # URDF file content as the ``robot_description`` parameter. ``cat`` only
    # runs when a state-publisher node is actually included (conditioned
    # actions don't visit their substitutions otherwise).
    robot_description = {
        "robot_description": ParameterValue(
            Command(["cat ", urdf_path]), value_type=str
        ),
        "use_sim_time": use_sim_time,
    }

    common_params: dict[str, object] = {
        "address": address,
        "port": port,
        "use_sim_time": use_sim_time,
        "tls": False,
        "capabilities": READ_ONLY_CAPABILITIES,
        # Sensor frames can be large; keep the upstream 10 MB send buffer.
        "send_buffer_limit": 10_000_000,
        "max_qos_depth": 10,
        # Don't surface ROS hidden topics (e.g. action feedback internals).
        "include_hidden": False,
    }

    # `topic_whitelist` can't be branched in Python (it depends on a launch
    # arg resolved at runtime), so we gate two mutually-exclusive Nodes on
    # `expose_all_topics`. Same node name — exactly one ever runs. Neither
    # re-enables clientPublish/services; the escape hatch only widens which
    # topics are *read*.
    bridge_safe = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="openral_foxglove_bridge",
        output="screen",
        condition=UnlessCondition(expose_all),
        parameters=[{**common_params, "topic_whitelist": BUCKET1_TOPIC_WHITELIST}],
    )
    bridge_all = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="openral_foxglove_bridge",
        output="screen",
        condition=IfCondition(expose_all),
        parameters=[{**common_params, "topic_whitelist": [r".*"]}],
    )

    # Optional /tf + robot-model publishers (off by default). Both are pure
    # transforms of /joint_states + URDF → read-only; they publish no command.
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="openral_robot_state_publisher",
        output="screen",
        condition=IfCondition(with_rsp),
        parameters=[robot_description],
    )
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="openral_joint_state_publisher",
        output="screen",
        condition=IfCondition(with_jsp),
        parameters=[robot_description],
    )

    return LaunchDescription(
        [*args, bridge_safe, bridge_all, robot_state_publisher, joint_state_publisher]
    )
