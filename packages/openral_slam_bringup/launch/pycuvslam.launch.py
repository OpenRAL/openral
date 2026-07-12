#!/usr/bin/env python3
"""Stand-alone launch for the PyCuVSLAM stereo node under ``/openral``.

The in-process alternative to ``cuvslam.launch.py``: same cuVSLAM engine,
but from the NVIDIA **PyCuVSLAM** pip wheel instead of the composable
``isaac_ros_visual_slam`` C++ node — no Isaac ROS apt stack (NITROS /
VPI / ``nvsci``) required. It fills the same ``map → odom`` TF edge; see
``openral_slam_bringup/pycuvslam_node.py`` for the frame contract and
license posture (wheel is operator-installed, never bundled).

Unlike the NITROS path there is no zero-copy requirement, so this is a
plain ``Node``, not a ``ComposableNodeContainer``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_NODE_NAME = "openral_pycuvslam"
_PACKAGE = "openral_slam_bringup"
_EXECUTABLE = "pycuvslam_node.py"


def _default_params_path() -> str:
    share = get_package_share_directory("openral_slam_bringup")
    return os.path.join(share, "config", "pycuvslam.yaml")


def generate_launch_description() -> LaunchDescription:
    """Stand-alone bring-up for the PyCuVSLAM stereo node."""
    args = [
        DeclareLaunchArgument(
            "params_file",
            default_value=_default_params_path(),
            description=(
                "YAML parameter file; defaults to openral_slam_bringup/config/pycuvslam.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Pass-through to the node's `use_sim_time`.",
        ),
        # Stereo camera topics. Like the sibling ``cuvslam.launch.py`` (which
        # owns its ``image_0/1_topic`` remaps), this launch owns the topic
        # defaults so a scene's ``slam_stereo_cameras`` can retarget the rig
        # via ``sim_e2e.launch.py`` without editing a params file. Passed as
        # parameter overrides above ``params_file``.
        DeclareLaunchArgument(
            "left_image_topic",
            default_value="/openral/cameras/left/image",
            description="Rectified stereo-left image topic.",
        ),
        DeclareLaunchArgument(
            "left_camera_info_topic",
            default_value="/openral/cameras/left/camera_info",
            description="Calibration for the left image.",
        ),
        DeclareLaunchArgument(
            "right_image_topic",
            default_value="/openral/cameras/right/image",
            description="Rectified stereo-right image topic.",
        ),
        DeclareLaunchArgument(
            "right_camera_info_topic",
            default_value="/openral/cameras/right/camera_info",
            description="Calibration for the right image.",
        ),
        # Multi-camera rig frame. ``robot_yaml`` lets the node derive the rig
        # frame from the manifest's base_frame (the natural rig for a mobile
        # robot with base-mounted cameras); ``rig_frame`` overrides it directly.
        # Either non-empty selects cuVSLAM's multi-camera mode (per-camera
        # extrinsics from TF) over the rectified-baseline path.
        DeclareLaunchArgument(
            "robot_yaml",
            default_value="",
            description="Robot manifest; the node derives the rig frame from its base_frame.",
        ),
        DeclareLaunchArgument(
            "rig_frame",
            default_value="",
            description="Explicit rig frame (overrides robot_yaml's base_frame).",
        ),
    ]

    node = Node(
        package=_PACKAGE,
        executable=_EXECUTABLE,
        name=_NODE_NAME,
        namespace="",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "left_image_topic": LaunchConfiguration("left_image_topic"),
                "left_camera_info_topic": LaunchConfiguration("left_camera_info_topic"),
                "right_image_topic": LaunchConfiguration("right_image_topic"),
                "right_camera_info_topic": LaunchConfiguration("right_camera_info_topic"),
                "robot_yaml": LaunchConfiguration("robot_yaml"),
                "rig_frame": LaunchConfiguration("rig_frame"),
            },
        ],
        output="screen",
    )

    return LaunchDescription([*args, node])


# Used by `test/test_pycuvslam_launch.py` for hermetic argument validation
# without spawning a real ROS 2 graph (and without the cuVSLAM wheel).
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "pycuvslam.yaml"
NODE_NAME = _NODE_NAME
PACKAGE = _PACKAGE
EXECUTABLE = _EXECUTABLE
