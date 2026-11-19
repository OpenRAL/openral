r"""Real-hardware ros2_control bringup for the dual Flexiv Rizon 4s.

``RizonDualRealHAL`` (``python/hal/src/openral_hal/flexiv_rizon_real.py``) only publishes
to two already-running ``JointTrajectoryController``s; this launch starts that C++ graph
(CLAUDE.md §1.5): ``robot_state_publisher`` with this package's xacro (upstream
``flexiv_description`` kinematics + one upstream ``FlexivHardwareInterface`` per arm),
``controller_manager`` at 1 kHz, and the three spawners. ``openral deploy run`` includes it
through the manifest's ``hal.real_bringup`` and forwards no arguments, so the defaults below
are the reference rig's and must agree with ``robots/rizon_dual/robot.yaml``.

**Activation energises the arms.** ``FlexivHardwareInterface::on_activate`` clears a fault and
calls ``Robot::Enable()`` (brakes released, servo on); the trajectory controllers then hold the
measured pose. Nothing moves until a trajectory arrives, but run it only with the arms'
E-stop in reach.

Host-confined graph first, as every OpenRAL real bringup (ADR-0130, issue #556).

Usage::

    source ~/flexiv_ws/install/setup.bash   # upstream Flexiv packages (README "Provisioning")
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
        .venv/bin/python "$(command -v ros2)" launch openral_hal_rizon real_bringup.launch.py
"""

from __future__ import annotations

import os
from typing import Any

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# The reference rig's arms (q5090 cell): left on the 180-deg-rolled mount.
_ROBOT_SN_LEFT = "Rizon4s-063642"
_ROBOT_SN_RIGHT = "Rizon4s-063618"
# Must match RizonDualRealHAL's defaults.
_CONTROLLERS = ("joint_state_broadcaster", "left_arm_controller", "right_arm_controller")


def _require_host_confined_graph(context: LaunchContext) -> list[Any]:
    """Refuse a networked ROS graph before anything is started (issue #556)."""
    try:
        from openral_core.ros_graph_scope import require_host_confined_real_bringup
    except ImportError as exc:  # fail closed: the scope cannot be checked
        raise ImportError(
            f"openral_hal_rizon real_bringup.launch.py checks the ROS graph scope with "
            f"openral_core ({exc}): run it under the OpenRAL venv "
            f'(`.venv/bin/python "$(command -v ros2)" launch ...`) or via `openral deploy run`.'
        ) from exc
    require_host_confined_real_bringup(
        "openral_hal_rizon",
        os.environ,
        included_by_deploy="hal_mode" in context.launch_configurations,
    )
    return []


def generate_launch_description() -> LaunchDescription:
    """Start controller_manager + both Flexiv arms + the three controllers."""
    share = FindPackageShare("openral_hal_rizon")
    robot_description = ParameterValue(
        Command(
            [
                PathJoinSubstitution([FindExecutable(name="xacro")]),
                " ",
                PathJoinSubstitution([share, "urdf", "rizon_dual.urdf.xacro"]),
                " ros2_control:=true",
                " robot_sn_left:=",
                LaunchConfiguration("robot_sn_left"),
                " robot_sn_right:=",
                LaunchConfiguration("robot_sn_right"),
                " mounting_file:=",
                LaunchConfiguration("mounting_file"),
                " network_interface_whitelist:=",
                LaunchConfiguration("network_interface_whitelist"),
            ]
        ),
        value_type=str,
    )
    spawners = [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[name, "--controller-manager", "/controller_manager"],
            output="screen",
        )
        for name in _CONTROLLERS
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_sn_left", default_value=_ROBOT_SN_LEFT),
            DeclareLaunchArgument("robot_sn_right", default_value=_ROBOT_SN_RIGHT),
            DeclareLaunchArgument(
                "mounting_file",
                default_value=PathJoinSubstitution([share, "config", "rizon_dual_mounting.yaml"]),
            ),
            DeclareLaunchArgument(
                "network_interface_whitelist",
                default_value="",
                description="Comma-separated host IPs the RDK discovers on (multi-NIC hosts).",
            ),
            OpaqueFunction(function=_require_host_confined_graph),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[PathJoinSubstitution([share, "config", "rizon_dual_controllers.yaml"])],
                output="screen",
            ),
            *spawners,
        ]
    )
