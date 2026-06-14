#!/usr/bin/env python3
"""ADR-0025 — stand-alone launch for the Nav2 stack.

Includes the upstream ``nav2_bringup/launch/navigation_launch.py`` —
brings up ``bt_navigator``, ``planner_server``, ``controller_server``,
``smoother_server``, ``behavior_server``, ``velocity_smoother`` and
the ``lifecycle_manager_navigation`` that drives them all to
``ACTIVE``. Parameters come from this package's
``config/nav2_panda_mobile.yaml`` (the shared base) — a copy of the
upstream ``nav2_params.yaml``. Per-robot geometry/kinematics are NOT
hand-edited here: when a ``robot_yaml`` arg is passed, ``RewrittenYaml``
substitutes ``robot_radius`` (from the robot's ``footprint_radius``),
the costmap ``inflation_radius`` (footprint + a small clearance) and
the MPPI ``motion_model`` (from ``base_kinematics``) via
``RobotDescription.nav2_param_overrides()`` — so one base file serves
any mobile base. The base ships panda_mobile's values
(``robot_radius: 0.35``, ``inflation_radius: 0.40``, ``motion_model:
Omni`` for the holonomic base, symmetric ``vy_min: -0.5``), so the
rewrite is a no-op for panda and differs only for a differently-shaped
robot. Velocity bounds remain Nav2 tuning in the base file (not robot
identity).

Unlike slam_toolbox (which idles until the Reasoner activates it),
Nav2 is always-on: each Nav2 sub-node is a LifecycleNode driven by
the in-stack ``lifecycle_manager_navigation`` (autostart=true). The
Reasoner triggers Nav2 by dispatching the
``OpenRAL/rskill-nav2-navigate-to-pose`` wrapped-action rSkill —
which sends a ``NavigateToPose`` action goal to ``/navigate_to_pose``
— rather than by lifecycle-transitioning the planner.

Composed into ``packages/openral_rskill_ros/launch/sim_e2e.launch.py``
when the ``enable_nav2`` launch argument is ``true``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _default_params_path() -> str:
    share = get_package_share_directory("openral_nav2_bringup")
    return os.path.join(share, "config", "nav2_panda_mobile.yaml")


def _upstream_navigation_launch() -> str:
    share = get_package_share_directory("nav2_bringup")
    return os.path.join(share, "launch", "navigation_launch.py")


def generate_launch_description() -> LaunchDescription:
    """Stand-alone bring-up for the upstream Nav2 navigation stack."""
    args = [
        DeclareLaunchArgument(
            "params_file",
            default_value=_default_params_path(),
            description=(
                "YAML parameter file for Nav2; defaults to "
                "openral_nav2_bringup/config/nav2_panda_mobile.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Pass-through to Nav2's `use_sim_time`.",
        ),
        DeclareLaunchArgument(
            "autostart",
            default_value="true",
            description=(
                "Drive Nav2's lifecycle_manager_navigation to ACTIVE "
                "automatically. Nav2 sits idle until a NavigateToPose "
                "goal arrives, so always-on is the right default — "
                "the Reasoner triggers navigation by dispatching the "
                "wrapped-action rSkill, not by lifecycle-transition."
            ),
        ),
        DeclareLaunchArgument(
            "use_composition",
            default_value="False",
            description=(
                "When True, run all Nav2 components in a single "
                "process. Off by default — composition makes "
                "per-component lifecycle introspection harder."
            ),
        ),
        DeclareLaunchArgument(
            "robot_yaml",
            default_value="",
            description=(
                "Path to the robot's robot.yaml. When set, the base "
                "params_file is rewritten with the robot's "
                "`RobotDescription.nav2_param_overrides()` (robot_radius + "
                "inflation_radius from footprint_radius, motion_model from "
                "base_kinematics) so one shared base file serves any mobile "
                "base. Empty string uses params_file verbatim (ADR-0025)."
            ),
        ),
    ]

    # ``OpaqueFunction`` (upstream ``launch.actions``) defers the callback
    # to launch-execution time, where the ``robot_yaml`` / ``params_file``
    # LaunchConfiguration values are finally resolved — we need them to
    # build the per-robot RewrittenYaml, which can't happen at parse time.
    return LaunchDescription([*args, OpaqueFunction(function=_nav2_include_with_robot_overrides)])


def _nav2_include_with_robot_overrides(context: object) -> list[IncludeLaunchDescription]:
    """Rewrite the base Nav2 params with per-robot overrides, then include.

    Runs at launch time (via ``OpaqueFunction``) so it can read the
    resolved ``robot_yaml`` / ``params_file`` launch args off the
    ``context``.

    ADR-0025 — keeps the bringup generic: ``robot.yaml`` is the single
    source for the per-robot Nav2 geometry/kinematics. ``RewrittenYaml``
    substitutes the matching keys in the shared base param file; an empty
    ``robot_yaml`` (or a fixed-base arm) yields no rewrites and the base
    file is used verbatim.
    """
    from nav2_common.launch import (
        RewrittenYaml,  # reason: nav2 dep, launch-time only
    )

    params_file = LaunchConfiguration("params_file").perform(context)  # type: ignore[attr-defined]
    robot_yaml = LaunchConfiguration("robot_yaml").perform(context)  # type: ignore[attr-defined]

    rewrites: dict[str, str] = {}
    if robot_yaml:
        from openral_core import (
            RobotDescription,  # reason: defer schema import to launch time
        )

        rewrites = RobotDescription.from_yaml(robot_yaml).nav2_param_overrides()

    resolved_params = RewrittenYaml(
        source_file=params_file,
        param_rewrites=rewrites,
        convert_types=True,
    )
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(_upstream_navigation_launch()),
            launch_arguments={
                "params_file": resolved_params,
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "autostart": LaunchConfiguration("autostart"),
                "use_composition": LaunchConfiguration("use_composition"),
            }.items(),
        )
    ]


# Used by ``test/test_nav2_launch.py`` for hermetic argument validation
# without spawning a real ROS 2 graph.
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "nav2_panda_mobile.yaml"
