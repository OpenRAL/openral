#!/usr/bin/env python3
"""Stand-alone launch for cuVSLAM (NVIDIA Isaac ROS Visual SLAM) under ``/openral/visual_slam``.

Camera-based SLAM backend for **lidar-less** robots: fills the same
``map→odom`` TF edge ``slam_toolbox`` fills on lidar robots, from
stereo / mono+IMU / RGB-D cameras instead of ``/scan``. Composed into
``sim_e2e.launch.py`` when ``slam_backend`` is ``visual`` (from
``RobotCapabilities.has_vision_slam``, see ``deploy_sim.py``).

Unlike ``slam_toolbox.launch.py``, cuVSLAM's ``VisualSlamNode`` is a
**composable node**, not a lifecycle node — live once composed, no
Reasoner-driven CONFIGURE/ACTIVATE. Runs inside a ``ComposableNodeContainer``
so NITROS type-negotiated image topics stay intra-process zero-copy.

cuVSLAM ships as a precompiled NVIDIA library under an NVIDIA EULA — **not**
bundled by OpenRAL (license guard); the operator installs
``isaac_ros_visual_slam`` on the target GPU host.

Topic remappings map OpenRAL's camera bus onto cuVSLAM's
``visual_slam/image_{i}`` / ``visual_slam/camera_info_{i}`` /
``visual_slam/imu``; pass them via the matching launch arguments.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

_VSLAM_NODE_NAME = "openral_visual_slam"
_VSLAM_CONTAINER_NAME = "openral_visual_slam_container"
# Upstream package + composable component for cuVSLAM. OpenRAL does not
# vendor these — they come from the operator's Isaac ROS install.
_VSLAM_PACKAGE = "isaac_ros_visual_slam"
_VSLAM_PLUGIN = "nvidia::isaac_ros::visual_slam::VisualSlamNode"


def _default_params_path() -> str:
    share = get_package_share_directory("openral_slam_bringup")
    return os.path.join(share, "config", "cuvslam.yaml")


def generate_launch_description() -> LaunchDescription:
    """Stand-alone bring-up for the upstream cuVSLAM composable node."""
    args = [
        DeclareLaunchArgument(
            "params_file",
            default_value=_default_params_path(),
            description=(
                "YAML parameter file for cuVSLAM; defaults to "
                "openral_slam_bringup/config/cuvslam.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "node_name",
            default_value=_VSLAM_NODE_NAME,
            description="Composable node name for the cuVSLAM component.",
        ),
        DeclareLaunchArgument(
            "container_name",
            default_value=_VSLAM_CONTAINER_NAME,
            description="Name of the ComposableNodeContainer hosting cuVSLAM.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Pass-through to cuVSLAM's `use_sim_time`.",
        ),
        # Camera-bus remappings: OpenRAL publishes RGB on
        # `/openral/cameras/<name>/image`; the operator points cuVSLAM's
        # stereo inputs at the robot's left/right (or RGB-D) streams.
        DeclareLaunchArgument(
            "image_0_topic",
            default_value="/openral/cameras/left/image",
            description="Stereo-left (or mono / RGB) image → visual_slam/image_0.",
        ),
        DeclareLaunchArgument(
            "camera_info_0_topic",
            default_value="/openral/cameras/left/camera_info",
            description="Calibration for image_0 → visual_slam/camera_info_0.",
        ),
        DeclareLaunchArgument(
            "image_1_topic",
            default_value="/openral/cameras/right/image",
            description="Stereo-right image → visual_slam/image_1 (stereo rigs).",
        ),
        DeclareLaunchArgument(
            "camera_info_1_topic",
            default_value="/openral/cameras/right/camera_info",
            description="Calibration for image_1 → visual_slam/camera_info_1.",
        ),
        DeclareLaunchArgument(
            "imu_topic",
            default_value="/openral/imu",
            description="IMU stream → visual_slam/imu (visual-inertial rigs).",
        ),
    ]

    params_file = LaunchConfiguration("params_file")
    node_name = LaunchConfiguration("node_name")
    container_name = LaunchConfiguration("container_name")
    use_sim_time = LaunchConfiguration("use_sim_time")

    vslam_node = ComposableNode(
        package=_VSLAM_PACKAGE,
        plugin=_VSLAM_PLUGIN,
        name=node_name,
        namespace="",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
        remappings=[
            ("visual_slam/image_0", LaunchConfiguration("image_0_topic")),
            ("visual_slam/camera_info_0", LaunchConfiguration("camera_info_0_topic")),
            ("visual_slam/image_1", LaunchConfiguration("image_1_topic")),
            ("visual_slam/camera_info_1", LaunchConfiguration("camera_info_1_topic")),
            ("visual_slam/imu", LaunchConfiguration("imu_topic")),
        ],
    )

    container = ComposableNodeContainer(
        name=container_name,
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[vslam_node],
        output="screen",
    )

    return LaunchDescription([*args, container])


# Used by `test/test_cuvslam_launch.py` for hermetic argument validation
# without spawning a real ROS 2 graph (and without the cuVSLAM engine).
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "cuvslam.yaml"
NODE_NAME = _VSLAM_NODE_NAME
PACKAGE = _VSLAM_PACKAGE
PLUGIN = _VSLAM_PLUGIN
