#!/usr/bin/env python3
"""ADR-0025 — stand-alone launch for slam_toolbox under ``/openral/slam_toolbox``.

Includes the upstream ``slam_toolbox/async_slam_toolbox_node`` as a
``LifecycleNode`` parameterised from this package's
``config/slam_toolbox_2d.yaml``. The auto-transition stops at
``INACTIVE``; the Reasoner promotes to ``ACTIVE`` via
:class:`~openral_core.LifecycleTransitionTool`, mirroring the
safety_kernel pattern in ``sim_e2e.launch.py:159``.

Composed into ``packages/openral_rskill_ros/launch/sim_e2e.launch.py``
when the ``enable_slam`` launch argument is ``true``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode

_SLAM_NODE_NAME = "openral_slam_toolbox"
# `/openral/slam_toolbox` is the canonical full path the Reasoner
# emits in `LifecycleTransitionTool(node=..., transition=...)`. ROS
# joins `namespace` + `name`, so we use namespace="" and the name
# above; the registered node ends up at `/openral_slam_toolbox` —
# the deploy-sim CLI rewrites this to `/openral/slam_toolbox` if a
# namespace remap is required.


def _default_params_path() -> str:
    share = get_package_share_directory("openral_slam_bringup")
    return os.path.join(share, "config", "slam_toolbox_2d.yaml")


def generate_launch_description() -> LaunchDescription:
    """Stand-alone bring-up for the upstream slam_toolbox node."""
    args = [
        DeclareLaunchArgument(
            "params_file",
            default_value=_default_params_path(),
            description=(
                "YAML parameter file for slam_toolbox; defaults to "
                "openral_slam_bringup/config/slam_toolbox_2d.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "node_name",
            default_value=_SLAM_NODE_NAME,
            description=(
                "Lifecycle node name. The Reasoner's "
                "LifecycleTransitionTool.node field MUST match this "
                "fully-qualified ROS node name."
            ),
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Pass-through to slam_toolbox's `use_sim_time`.",
        ),
    ]

    params_file = LaunchConfiguration("params_file")
    node_name = LaunchConfiguration("node_name")
    use_sim_time = LaunchConfiguration("use_sim_time")

    slam_node = LifecycleNode(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name=node_name,
        namespace="",
        parameters=[params_file, {"use_sim_time": use_sim_time}],
        output="screen",
    )

    # ADR-0025 — leave slam_toolbox in UNCONFIGURED. The Reasoner
    # promotes through CONFIGURE → ACTIVATE via
    # ``LifecycleTransitionTool(node="/openral_slam_toolbox",
    # transition=...)``. We deliberately do NOT auto-configure from
    # the launch: slam_toolbox 2.8.4's ``on_configure`` returns
    # SUCCESS at the end (``src/slam_toolbox_common.cpp:139``) but
    # the change_state service response on Jazzy arrives with
    # ``response.success=false`` even though the FSM does transition
    # to INACTIVE — launch_ros logs a spurious
    # ``Failed to make transition 'TRANSITION_CONFIGURE'`` ERROR.
    # Reasoner-driven lifecycle dodges the upstream race entirely.
    return LaunchDescription([*args, slam_node])


# Used by `tests/test_slam_toolbox_launch.py` for hermetic argument
# validation without spawning a real ROS 2 graph.
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "slam_toolbox_2d.yaml"
NODE_NAME = _SLAM_NODE_NAME
