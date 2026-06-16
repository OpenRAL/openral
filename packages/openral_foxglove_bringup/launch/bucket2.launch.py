#!/usr/bin/env python3
"""Launch file for the Bucket-2 converter node (ADR-0059 Phase 3).

Spawns ``bucket2_markers`` — a read-only converter that re-publishes
``openral_msgs/WorldCollision`` and ``openral_msgs/OccupancyVoxels``
as standard ROS visualization types so Foxglove renders them natively
without any TypeScript extension.

Output topics (added to the Bucket-1 allowlist in ``topics.py``):
  /openral/world_collisions_markers  — visualization_msgs/MarkerArray
  /openral/world_voxels_cloud        — sensor_msgs/PointCloud2

This launch pairs with ``foxglove.launch.py`` but does not include it;
compose them with ``IncludeLaunchDescription`` or run them together
with ``ros2 launch`` as separate processes.

Run:
    ros2 launch openral_foxglove_bringup bucket2.launch.py
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Bucket-2 converter node launch."""
    args = [
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description=(
                "Set true when a /clock is published (deploy-sim, ADR-0048) "
                "so marker/cloud timestamps align with sim time."
            ),
        ),
    ]

    use_sim_time = LaunchConfiguration("use_sim_time")

    converter = Node(
        package="openral_foxglove_bringup",
        executable="bucket2_markers",
        name="openral_bucket2_markers",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription([*args, converter])
