r"""Run the OctoMap → OccupancyVoxels bridge for the safety kernel (ADR-0030).

Wire it alongside your OctoMap producer (e.g. octomap_server) and the safety
kernel launched with ``world_voxel_enabled:=true``::

    ros2 launch openral_octomap_bridge octomap_voxel_bridge.launch.py \\
        base_frame:=base_link octomap_topic:=/octomap_binary

The bridge needs TF from ``base_frame`` into the octomap's ``header.frame_id``
(usually ``map``) — typically published by your SLAM / localization stack.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Declare the bridge's parameters and spawn the node."""
    args = [
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("octomap_topic", default_value="/octomap_binary"),
        DeclareLaunchArgument("output_topic", default_value="/openral/world_voxels"),
        DeclareLaunchArgument("resolution", default_value="0.05"),
        DeclareLaunchArgument("box_size_x", default_value="2.0"),
        DeclareLaunchArgument("box_size_y", default_value="2.0"),
        DeclareLaunchArgument("box_size_z", default_value="2.0"),
        DeclareLaunchArgument("box_center_z", default_value="0.5"),
        DeclareLaunchArgument("publish_rate_hz", default_value="10.0"),
    ]
    bridge = Node(
        package="openral_octomap_bridge",
        executable="octomap_voxel_bridge",
        name="openral_octomap_voxel_bridge",
        output="screen",
        parameters=[
            {
                "base_frame": LaunchConfiguration("base_frame"),
                "octomap_topic": LaunchConfiguration("octomap_topic"),
                "output_topic": LaunchConfiguration("output_topic"),
                "resolution": LaunchConfiguration("resolution"),
                "box_size_x": LaunchConfiguration("box_size_x"),
                "box_size_y": LaunchConfiguration("box_size_y"),
                "box_size_z": LaunchConfiguration("box_size_z"),
                "box_center_z": LaunchConfiguration("box_center_z"),
                "publish_rate_hz": LaunchConfiguration("publish_rate_hz"),
            }
        ],
    )
    return LaunchDescription([*args, bridge])
