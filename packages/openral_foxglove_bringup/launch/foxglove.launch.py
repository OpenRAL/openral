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
is invisible to the bridge. This is the feasibility spike for ADR-0059
(``docs/adr/0059-foxglove-live-scene-visualization.md``); graduating it
past prototype requires that ADR's sign-offs.

Run:
    ros2 launch openral_foxglove_bringup foxglove.launch.py

Then open https://app.foxglove.dev (or the desktop app), choose
"Open connection → Foxglove WebSocket → ws://localhost:8765", and import
``config/openral_layout.json``.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_context import LaunchContext
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
                'print(resolve_urdf_path(open("robots/<id>/robot.yaml")...))\'`` '
                "or directly via robot_descriptions (e.g. panda_description). "
                "NOTE: openarm has no local URDF (ADR-0027)."
            ),
        ),
        # --- Compressed-image transport (ADR-0059 decision 4) ---------------
        # Raw sensor_msgs/Image is ~9 MB/s per camera; a multi-camera arm can
        # saturate a laptop link and Foxglove's send buffer. The republisher
        # below converts selected raw camera topics to sensor_msgs/CompressedImage
        # via image_transport, bringing bandwidth down ~10×. Default OFF so the
        # raw path stays the default and no extra nodes run in CI.
        DeclareLaunchArgument(
            "republish_compressed",
            default_value="false",
            description=(
                "When true, spawn one image_transport republisher per topic in "
                "``compressed_camera_topics`` to convert raw→compressed. "
                "Compressed topics (/…/image/compressed) are already Bucket-1 "
                "whitelisted (ADR-0059 decision 4). Default false — raw path stays "
                "available for fidelity-sensitive use."
            ),
        ),
        DeclareLaunchArgument(
            "compressed_camera_topics",
            default_value="",
            description=(
                "Comma- or space-separated list of raw sensor_msgs/Image topics to "
                "compress when ``republish_compressed`` is true. Example: "
                '"/openral/cameras/base/image /openral/cameras/left_wrist/image". '
                "Each topic gains a /compressed sibling republished by "
                "image_transport."
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
    republish_compressed = LaunchConfiguration("republish_compressed")
    compressed_camera_topics = LaunchConfiguration("compressed_camera_topics")

    # URDF file content as the ``robot_description`` parameter. ``cat`` only
    # runs when a state-publisher node is actually included (conditioned
    # actions don't visit their substitutions otherwise).
    robot_description = {
        "robot_description": ParameterValue(Command(["cat ", urdf_path]), value_type=str),
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

    # ADR-0059 decision 4: opt-in compressed-image republishers.
    # ``compressed_camera_topics`` is a runtime LaunchConfiguration string —
    # we can't branch on it at module load, so it is resolved inside an
    # OpaqueFunction that runs after all args are substituted.
    def _spawn_compressed_republishers(
        context: LaunchContext,
        republish_compressed_cfg: LaunchConfiguration,
        compressed_camera_topics_cfg: LaunchConfiguration,
        use_sim_time_cfg: LaunchConfiguration,
    ) -> list:
        if republish_compressed_cfg.perform(context).lower() != "true":
            return []
        raw_topics_str = compressed_camera_topics_cfg.perform(context).strip()
        if not raw_topics_str:
            return []
        sim_time = use_sim_time_cfg.perform(context)
        # Accept comma- or whitespace-separated topic lists.
        raw_topics = [t for t in raw_topics_str.replace(",", " ").split() if t]
        nodes = []
        for idx, topic in enumerate(raw_topics):
            # image_transport republish subscribes on remap "in" (raw) and
            # publishes compressed images with remap "out/compressed" →
            # <topic>/compressed, matching the Bucket-1 whitelist pattern
            # r"/openral/cameras/.*/image/compressed".
            nodes.append(
                Node(
                    package="image_transport",
                    executable="republish",
                    name=f"openral_foxglove_compressed_republisher_{idx}",
                    output="screen",
                    arguments=["raw", "compressed"],
                    parameters=[{"use_sim_time": sim_time == "true"}],
                    remappings=[
                        ("in", topic),
                        ("out/compressed", topic + "/compressed"),
                    ],
                )
            )
        return nodes

    compressed_republishers = OpaqueFunction(
        function=_spawn_compressed_republishers,
        args=[republish_compressed, compressed_camera_topics, use_sim_time],
    )

    return LaunchDescription(
        [
            *args,
            bridge_safe,
            bridge_all,
            robot_state_publisher,
            joint_state_publisher,
            compressed_republishers,
        ]
    )
