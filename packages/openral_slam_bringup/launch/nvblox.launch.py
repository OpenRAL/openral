#!/usr/bin/env python3
"""ADR-0064 (Phase 2) — stand-alone launch for NVIDIA Isaac ROS nvblox.

nvblox turns cuVSLAM's pose (`map→odom` TF) plus a depth image into the 2D
obstacle map a **lidar-less** robot needs for Nav2 — the occupancy half of
SLAM that cuVSLAM alone does not provide. OpenRAL first filters depth pixels to
a robot-measurement-derived body-height band so floor returns do not project
into `/map` as occupied cells, then remaps nvblox's `~/static_occupancy_grid`
(`nav_msgs/OccupancyGrid`) to the backend-agnostic `/map`. nvblox can also
publish a height-matched ESDF slice for deployments that use the `nvblox_nav2`
costmap plugin.

Depth comes either from a real RGB-D sensor or, for mono-only robots, from the
monocular metric-depth provider (DA3-Small by default — measured 0.27 GB /
~27 Hz on an 8 GB Ada; see `openral_perception_ros` + the depth sidecar).

Like cuVSLAM, nvblox's node is a **composable node** (`nvblox::NvbloxNode`),
run inside a `ComposableNodeContainer`. The nvblox engine is a precompiled
NVIDIA binary OpenRAL does not bundle (ADR-0064 license guard); the operator
installs `nvblox_ros` on the target GPU host.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

_NVBLOX_NODE_NAME = "openral_nvblox"
_NVBLOX_CONTAINER_NAME = "openral_nvblox_container"
# Upstream package + composable component for nvblox. OpenRAL does not
# vendor these — they come from the operator's Isaac ROS install.
_NVBLOX_PACKAGE = "nvblox_ros"
_NVBLOX_PLUGIN = "nvblox::NvbloxNode"


def _default_params_path() -> str:
    share = get_package_share_directory("openral_slam_bringup")
    return os.path.join(share, "config", "nvblox.yaml")


def generate_launch_description() -> LaunchDescription:
    """Stand-alone bring-up for the upstream nvblox composable node."""
    args = [
        DeclareLaunchArgument(
            "params_file",
            default_value=_default_params_path(),
            description=(
                "YAML parameter file for nvblox; defaults to "
                "openral_slam_bringup/config/nvblox.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "node_name",
            default_value=_NVBLOX_NODE_NAME,
            description="Composable node name for the nvblox component.",
        ),
        DeclareLaunchArgument(
            "container_name",
            default_value=_NVBLOX_CONTAINER_NAME,
            description="Name of the ComposableNodeContainer hosting nvblox.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Pass-through to nvblox's `use_sim_time`.",
        ),
        DeclareLaunchArgument(
            "robot_yaml",
            default_value="",
            description=(
                "RobotDescription manifest used to derive the depth prefilter's "
                "navigation-height band from footprint/collision/link measurements. "
                "Empty falls back to live base/camera height."
            ),
        ),
        # Depth + camera_info from the metric-depth provider (or a real
        # RGB-D sensor); nvblox subscribes `depth/image` + `depth/camera_info`.
        DeclareLaunchArgument(
            "depth_image_topic",
            default_value="/openral/cameras/front_depth/depth/image",
            description="Metric depth (32FC1, metres) → nvblox depth/image.",
        ),
        DeclareLaunchArgument(
            "depth_camera_info_topic",
            default_value="/openral/cameras/front_depth/depth/camera_info",
            description="Depth intrinsics → nvblox depth/camera_info.",
        ),
        DeclareLaunchArgument(
            "height_filter_floor_clearance_m",
            default_value="0.10",
            description=(
                "Robot-relative clearance above the measured floor/chassis bottom "
                "retained by the depth filter before nvblox."
            ),
        ),
        DeclareLaunchArgument(
            "height_filter_min_body_height_m",
            default_value="0.30",
            description=(
                "Minimum retained body-height slice when a manifest has sparse "
                "geometry; real footprint/collision measurements can raise it."
            ),
        ),
        # ADR-0064 — publish nvblox's 2D occupancy grid on the SAME topic
        # slam_toolbox uses (`/map`) so the visual backend exposes one
        # backend-agnostic `nav_msgs/OccupancyGrid` interface. Nav2's
        # static_layer, the dashboard slam_bridge, and the reasoner's
        # `occupancy_map_topic` all consume `/map` regardless of how the map
        # was built (lidar vs cuVSLAM+nvblox).
        DeclareLaunchArgument(
            "map_topic",
            default_value="/map",
            description="Backend-agnostic OccupancyGrid topic for nvblox's "
            "static occupancy grid (matches slam_toolbox's /map).",
        ),
    ]

    params_file = LaunchConfiguration("params_file")
    node_name = LaunchConfiguration("node_name")
    container_name = LaunchConfiguration("container_name")
    use_sim_time = LaunchConfiguration("use_sim_time")
    filtered_depth_image_topic = "/openral/nvblox/depth_filtered/image"
    filtered_depth_camera_info_topic = "/openral/nvblox/depth_filtered/camera_info"

    depth_height_filter = Node(
        package="openral_slam_bringup",
        executable="depth_height_filter_node.py",
        name="openral_nvblox_depth_height_filter",
        namespace="",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "input_depth_topic": LaunchConfiguration("depth_image_topic"),
                "input_camera_info_topic": LaunchConfiguration("depth_camera_info_topic"),
                "output_depth_topic": filtered_depth_image_topic,
                "output_camera_info_topic": filtered_depth_camera_info_topic,
                "global_frame": "map",
                "robot_yaml": LaunchConfiguration("robot_yaml"),
                "floor_clearance_m": LaunchConfiguration("height_filter_floor_clearance_m"),
                "min_body_height_m": LaunchConfiguration("height_filter_min_body_height_m"),
            }
        ],
        output="screen",
    )

    nvblox_node = ComposableNode(
        package=_NVBLOX_PACKAGE,
        plugin=_NVBLOX_PLUGIN,
        name=node_name,
        namespace="",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
        # nvblox 4.4.0 subscribes camera-namespaced inputs `camera_0/depth/*`
        # and publishes its grids on the node-private `~/...` namespace (both
        # verified live via `ros2 node info`). Remap the depth inputs onto our
        # bus and the static occupancy grid onto the shared `/map`.
        remappings=[
            ("camera_0/depth/image", filtered_depth_image_topic),
            ("camera_0/depth/camera_info", filtered_depth_camera_info_topic),
            ("~/static_occupancy_grid", LaunchConfiguration("map_topic")),
        ],
    )

    container = ComposableNodeContainer(
        name=container_name,
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[nvblox_node],
        output="screen",
    )

    return LaunchDescription([*args, depth_height_filter, container])


# Used by `test/test_nvblox_launch.py` for hermetic argument validation
# without spawning a real ROS 2 graph (and without the nvblox engine).
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "nvblox.yaml"
NODE_NAME = _NVBLOX_NODE_NAME
PACKAGE = _NVBLOX_PACKAGE
PLUGIN = _NVBLOX_PLUGIN
