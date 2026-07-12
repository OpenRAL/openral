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
    ]

    node = Node(
        package=_PACKAGE,
        executable=_EXECUTABLE,
        name=_NODE_NAME,
        namespace="",
        parameters=[
            LaunchConfiguration("params_file"),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
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
