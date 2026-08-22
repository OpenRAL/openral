#!/usr/bin/env python3
"""Stand-alone launch for the Nav2 stack.

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
``OpenRAL/rskill-nav2-mobile_base-navigate_to_pose-none`` wrapped-action rSkill —
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


def _params_path_for_backend(backend: str) -> str:
    """Pick the Nav2 base params for the SLAM backend.

    ``visual`` (cuVSLAM + nvblox) uses ``nav2_visual.yaml`` — global+local
    costmaps consume the backend-agnostic ``/map`` via ``static_layer``.
    Anything else (``lidar``/``none``) uses the ``/scan``-based base config.
    """
    share = get_package_share_directory("openral_nav2_bringup")
    fname = "nav2_visual.yaml" if backend.strip().lower() == "visual" else "nav2_panda_mobile.yaml"
    return os.path.join(share, "config", fname)


def _default_params_path() -> str:
    return _params_path_for_backend("lidar")


def _upstream_navigation_launch() -> str:
    share = get_package_share_directory("nav2_bringup")
    return os.path.join(share, "launch", "navigation_launch.py")


def generate_launch_description() -> LaunchDescription:
    """Stand-alone bring-up for the upstream Nav2 navigation stack."""
    args = [
        DeclareLaunchArgument(
            "params_file",
            default_value="",
            description=(
                "YAML parameter file for Nav2. Empty (default) selects the "
                "base config by `slam_backend`: nav2_panda_mobile.yaml (lidar) "
                "or nav2_visual.yaml (visual). Set explicitly to override."
            ),
        ),
        DeclareLaunchArgument(
            "slam_backend",
            default_value="lidar",
            description=(
                "Which SLAM backend feeds the costmap: `visual` "
                "(cuVSLAM+nvblox; costmaps consume `/map` via static_layer) or "
                "`lidar`/`none` (costmaps ray-cast `/scan`). Selects the base "
                "params file when `params_file` is empty."
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
                "base. Empty string uses params_file verbatim."
            ),
        ),
        DeclareLaunchArgument(
            "payload_footprint",
            default_value="true",
            description=(
                "Run the payload footprint publisher + payload scan filter "
                "alongside Nav2. The publisher grows the costmaps' footprint "
                "polygon to cover a carried object; the filter takes that same "
                "object's returns out of the scan the costmaps and the "
                "collision monitor read. Needs `robot_yaml` for the publisher "
                "(the nominal outline is the manifest's). The scan filter runs "
                "on the lidar backend only — the base params point every "
                "observation source at its output, so turning this off means "
                "re-pointing them at `/scan`."
            ),
        ),
    ]

    # ``OpaqueFunction`` (upstream ``launch.actions``) defers the callback
    # to launch-execution time, where the ``robot_yaml`` / ``params_file``
    # LaunchConfiguration values are finally resolved — we need them to
    # build the per-robot RewrittenYaml, which can't happen at parse time.
    return LaunchDescription([*args, OpaqueFunction(function=_nav2_include_with_robot_overrides)])


def _payload_footprint_nodes(
    *, robot_yaml: str, slam_backend: str, use_sim_time: object
) -> list[object]:
    """The dynamic-footprint pair that rides with Nav2.

    ``openral_nav2_payload_footprint`` publishes the costmaps' ``footprint``
    polygon (chassis outline unioned with the ground projection of whatever is
    attached); ``openral_nav2_payload_scan_filter`` removes that same object's
    returns — **and the robot's own** — from the scan the costmaps and the
    collision monitor consume. Both read the attachment set off
    ``/openral/world_state_fast`` — the safety kernel's own source — so no
    consumer can act on a stale one, and both take the nominal chassis outline
    from the same ``robot_yaml``.

    The publisher is skipped without a ``robot_yaml`` (there is no manifest to
    take the nominal outline from, and inventing one would put a made-up robot
    shape on Nav2's collision surface). The scan filter still runs without one,
    with its self half disabled. The scan filter is skipped on the ``visual``
    backend, which has no ``/scan`` at all.
    """
    from launch_ros.actions import Node  # reason: launch-time only

    nodes: list[object] = []
    if robot_yaml:
        nodes.append(
            Node(
                package="openral_nav2_bringup",
                executable="payload_footprint_node.py",
                name="openral_nav2_payload_footprint",
                output="screen",
                parameters=[{"robot_yaml": robot_yaml, "use_sim_time": use_sim_time}],
            )
        )
    if slam_backend.strip().lower() != "visual":
        # `robot_yaml` is optional here, unlike for the publisher: without it
        # the payload half still runs and only the *self* half goes dark (the
        # node warns). With it, returns inside the manifest's bare chassis
        # outline are dropped too — the half a real lidar needs and a MuJoCo
        # ray-cast provides for free.
        params: dict[str, object] = {"use_sim_time": use_sim_time}
        if robot_yaml:
            params["robot_yaml"] = robot_yaml
        nodes.append(
            Node(
                package="openral_nav2_bringup",
                executable="payload_scan_filter_node.py",
                name="openral_nav2_payload_scan_filter",
                output="screen",
                parameters=[params],
            )
        )
    return nodes


def _nav2_include_with_robot_overrides(context: object) -> list[object]:
    """Rewrite the base Nav2 params with per-robot overrides, then include.

    Runs at launch time (via ``OpaqueFunction``) so it can read the
    resolved ``robot_yaml`` / ``params_file`` launch args off the
    ``context``.

    Keeps the bringup generic: ``robot.yaml`` is the single
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
    slam_backend = LaunchConfiguration("slam_backend").perform(context)  # type: ignore[attr-defined]
    # An empty params_file selects the base config by SLAM backend
    # (visual → nav2_visual.yaml consuming `/map`; lidar → the /scan base).
    if not params_file:
        params_file = _params_path_for_backend(slam_backend)

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
    actions: list[object] = [
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
    payload_footprint = LaunchConfiguration("payload_footprint").perform(context)  # type: ignore[attr-defined]
    if payload_footprint.strip().lower() in ("true", "1", "yes"):
        use_sim_time = LaunchConfiguration("use_sim_time").perform(context)  # type: ignore[attr-defined]
        actions.extend(
            _payload_footprint_nodes(
                robot_yaml=robot_yaml,
                slam_backend=slam_backend,
                use_sim_time=use_sim_time.strip().lower() in ("true", "1", "yes"),
            )
        )
    return actions


# Used by ``test/test_nav2_launch.py`` for hermetic argument validation
# without spawning a real ROS 2 graph.
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "nav2_panda_mobile.yaml"
