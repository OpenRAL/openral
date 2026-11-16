#!/usr/bin/env python3
"""Stand-alone launch for the Nav2 stack.

Includes upstream ``nav2_bringup/launch/navigation_launch.py`` — brings up ``bt_navigator``,
``planner_server``, ``controller_server``, ``smoother_server``, ``behavior_server``,
``velocity_smoother`` and ``lifecycle_manager_navigation`` (drives them to ``ACTIVE``). Params come
from ``config/nav2_panda_mobile.yaml`` (a copy of upstream ``nav2_params.yaml``). With a
``robot_yaml`` arg, ``RewrittenYaml`` substitutes ``robot_radius`` (from ``footprint_radius``),
costmap ``inflation_radius`` (circumscribed footprint + clearance) and MPPI ``motion_model`` (from
``base_kinematics``) via ``RobotDescription.nav2_param_overrides()``, so one base file serves any
mobile base. Base ships panda_mobile's values (``robot_radius: 0.35``, ``inflation_radius: 0.48``,
``motion_model: Omni``, ``vy_min: -0.5``) — a no-op rewrite for panda. Velocity limits stay Nav2
tuning in the base file, but are clamped to the manifest's kernel ``BODY_TWIST`` bounds
(``_cap_speeds``) so Nav2 never commands a twist the safety kernel would E-stop.

Unlike slam_toolbox (idles until the Reasoner activates it), Nav2 is always-on: each sub-node is a
``LifecycleNode`` driven by ``lifecycle_manager_navigation`` (``autostart=true``). The Reasoner
triggers navigation by dispatching the ``OpenRAL/rskill-nav2-mobile_base-navigate_to_pose``
wrapped-action rSkill (``NavigateToPose`` goal to ``/navigate_to_pose``), not by
lifecycle-transitioning the planner.

Composed into ``packages/openral_rskill_ros/launch/deploy_e2e.launch.py`` when ``enable_nav2=true``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

#: How long ``lifecycle_manager_navigation`` waits for a managed server's bond
#: heartbeat before declaring it dead and tearing down the whole Nav2 stack.
#:
#: Nav2's default is 4.0 s, and on a loaded host that is short enough to be
#: missed by a scheduling hiccup rather than by a real fault. The cascade is
#: silent and total: the manager deactivates every server, the graph stays up
#: doing nothing, and the run burns its deadline with no error printed. In the
#: 2026-09-06 ceiling battery this killed 31 of 89 valid runs, every one of
#: which the harness then scored as a policy failure (`deadline-no-grasp`) —
#: see ``docs/reference/collision-validation-evidence.md`` and issue #256.
#: Twenty-five of the 31 named ``controller_server``.
#:
#: This is a liveness timeout on the navigation stack, NOT a safety check: the
#: E-stop path is ``openral_safety_kernel``, which is unaffected. Raising it
#: trades a slower reaction to a genuinely hung server for not mistaking a
#: descheduled one for a dead one. It is not disabled (``0.0``), so a server
#: that really dies is still caught.
BOND_TIMEOUT_S = 30.0


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
                "inflation_radius from the footprint, motion_model from "
                "base_kinematics) so one shared base file serves any mobile "
                "base. Empty string uses params_file verbatim."
            ),
        ),
        DeclareLaunchArgument(
            "payload_scan_filter",
            default_value="true",
            description=(
                "Run the payload scan filter alongside Nav2: it removes the "
                "robot's OWN returns, and any carried object's, from the scan "
                "the costmaps and the collision monitor read. `robot_yaml` "
                "enables its self half (the manifest's bare chassis outline); "
                "without one only the payload half runs. Lidar backend only — "
                "the base params point every observation source at its output, "
                "so turning this off means re-pointing them at `/scan`. "
                "There is no footprint publisher: Nav2 is base-only and takes "
                "its polygon statically from the manifest (see the README)."
            ),
        ),
        DeclareLaunchArgument(
            "cmd_vel_relay",
            default_value="false",
            description=(
                "Opt-in: run `twist_to_action_relay_node.py`, which republishes Nav2's "
                "`/cmd_vel` as BODY_TWIST chunks on `/openral/candidate_action` so the safety "
                "kernel bounds them. Only for a manifest WITHOUT `base_joints` that supports "
                "`body_twist` (the node refuses otherwise; `MobileBaseBridge` owns `/cmd_vel` "
                "for base_joints robots, ADR-0040/0099). Requires `robot_yaml`."
            ),
        ),
    ]

    # ``OpaqueFunction`` (upstream ``launch.actions``) defers the callback
    # to launch-execution time, where the ``robot_yaml`` / ``params_file``
    # LaunchConfiguration values are finally resolved — we need them to
    # build the per-robot RewrittenYaml, which can't happen at parse time.
    return LaunchDescription([*args, OpaqueFunction(function=_nav2_include_with_robot_overrides)])


def _payload_scan_filter_nodes(
    *, robot_yaml: str, slam_backend: str, use_sim_time: object
) -> list[object]:
    """The scan filter that rides with Nav2.

    ``openral_nav2_payload_scan_filter`` removes the robot's own returns and any carried object's
    from the scan the costmaps and collision monitor consume. Reads the attachment set off
    ``/openral/world_state_fast`` (the safety kernel's own source) and takes the chassis outline
    from ``robot_yaml``.

    No footprint publisher: Nav2 is base-only, and the costmaps' ``footprint`` comes statically
    from the manifest via ``RobotDescription.nav2_param_overrides()`` (see README, "Nav2 is
    base-only").

    ``robot_yaml`` is optional: without it only the payload half runs (the node warns). Skipped on
    the ``visual`` backend, which has no ``/scan``.
    """
    from launch_ros.actions import Node  # reason: launch-time only

    nodes: list[object] = []
    if slam_backend.strip().lower() == "visual":
        return nodes
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


def _cmd_vel_relay_nodes(*, robot_yaml: str) -> list[object]:
    """The opt-in ``/cmd_vel`` -> ``/openral/candidate_action`` relay.

    ``robot_yaml`` is required: the relay reads the base limits and frame from it. Without one
    the launch fails here, rather than starting Nav2 with a relay that exits on its first line
    and leaves ``/cmd_vel`` unconsumed.

    Raises:
        ROSConfigError: ``robot_yaml`` is empty.
    """
    if not robot_yaml:
        from openral_core.exceptions import (
            ROSConfigError,  # reason: defer schema import to launch time
        )

        raise ROSConfigError("cmd_vel_relay:=true requires robot_yaml:=<path to robot.yaml>")
    from launch_ros.actions import Node  # reason: launch-time only

    # Wall clock on purpose: its staleness timer is a transport watchdog.
    return [
        Node(
            package="openral_nav2_bringup",
            executable="twist_to_action_relay_node.py",
            name="openral_nav2_twist_to_action_relay",
            output="screen",
            parameters=[{"robot_yaml": robot_yaml}],
        )
    ]


def _cap_speeds(
    params: dict[str, Any], max_linear_m_s: float | None, max_angular_rad_s: float | None
) -> None:
    """Clamp Nav2's velocity limits in ``params`` (in place) to the kernel ``BODY_TWIST`` bounds.

    The kernel vetoes + E-stops a twist with ``‖(vx, vy)‖`` or ``|wz|`` over the manifest's
    ``max_base_*`` bound (issue #405), so Nav2 must not command one. Only ever lowers a limit.

    ``velocity_smoother`` is the gate: controller_server and behavior_server both publish into it
    (``cmd_vel_nav``) and it is the only producer upstream of ``collision_monitor`` → ``/cmd_vel``.
    It clamps each axis independently, so the reachable ``(vx, vy)`` set is the box spanned by
    ``min_velocity``/``max_velocity``; one common scale puts all four corners of that box (mixed
    signs included) on or inside the bound's disk. MPPI's ``vx_max`` / ``wz_max`` are clamped too,
    so the controller does not plan what the smoother then cuts.

    Raises:
        ROSConfigError: a bound is set but ``params`` lacks the MPPI ``FollowPath`` or
            ``velocity_smoother`` limits it must cap — never launch without a verified cap.
    """
    from openral_core.exceptions import ROSConfigError  # reason: launch-time only

    try:
        follow_path = params["controller_server"]["ros__parameters"]["FollowPath"]
        smoother = params["velocity_smoother"]["ros__parameters"]
        vx_max, wz_max = float(follow_path["vx_max"]), float(follow_path["wz_max"])
        hi = [float(v) for v in smoother["max_velocity"]]
        lo = [float(v) for v in smoother["min_velocity"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ROSConfigError(
            "nav2.launch: cannot cap Nav2 speeds to the manifest's max_base_* bounds; the params "
            "file needs controller_server FollowPath.vx_max / wz_max (MPPI) and velocity_smoother "
            f"max_velocity / min_velocity [vx, vy, wz] (missing or malformed: {exc})"
        ) from exc
    if max_linear_m_s is not None:
        follow_path["vx_max"] = min(vx_max, max_linear_m_s)
        corner = math.hypot(max(abs(hi[0]), abs(lo[0])), max(abs(hi[1]), abs(lo[1])))
        if corner > max_linear_m_s:
            scale = max_linear_m_s / corner
            hi[:2] = [v * scale for v in hi[:2]]
            lo[:2] = [v * scale for v in lo[:2]]
    if max_angular_rad_s is not None:
        follow_path["wz_max"] = min(wz_max, max_angular_rad_s)
        hi[2] = math.copysign(min(abs(hi[2]), max_angular_rad_s), hi[2])
        lo[2] = math.copysign(min(abs(lo[2]), max_angular_rad_s), lo[2])
    smoother["max_velocity"], smoother["min_velocity"] = hi, lo


def _speed_capped_params_file(
    params_file: str, max_linear_m_s: float | None, max_angular_rad_s: float | None
) -> str:
    """Return ``params_file``, or a temp copy with ``_cap_speeds`` applied when a bound is set.

    A copy, not ``RewrittenYaml`` keys: Jazzy's ``RewrittenYaml`` cannot set a list element
    (``velocity_smoother.max_velocity``) and stringifies a whole-list value.
    """
    if max_linear_m_s is None and max_angular_rad_s is None:
        return params_file
    import tempfile  # reason: launch-time only

    import yaml  # reason: launch-time only

    params = yaml.safe_load(Path(params_file).read_text())
    _cap_speeds(params, max_linear_m_s, max_angular_rad_s)
    with tempfile.NamedTemporaryFile(
        "w", prefix="nav2_speed_capped_", suffix=".yaml", delete=False
    ) as out:
        yaml.safe_dump(params, out)
    return out.name


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

    params_file = LaunchConfiguration("params_file").perform(context)
    robot_yaml = LaunchConfiguration("robot_yaml").perform(context)
    slam_backend = LaunchConfiguration("slam_backend").perform(context)
    # An empty params_file selects the base config by SLAM backend
    # (visual → nav2_visual.yaml consuming `/map`; lidar → the /scan base).
    if not params_file:
        params_file = _params_path_for_backend(slam_backend)

    rewrites: dict[str, str] = {}
    if robot_yaml:
        from openral_core import (
            RobotDescription,  # reason: defer schema import to launch time
        )

        description = RobotDescription.from_yaml(robot_yaml)
        rewrites = description.nav2_param_overrides()
        params_file = _speed_capped_params_file(
            params_file,
            description.safety.max_base_linear_speed_m_s,
            description.safety.max_base_angular_speed_rad_s,
        )

    resolved_params = RewrittenYaml(
        source_file=params_file,
        param_rewrites=rewrites,
        convert_types=True,
    )
    from launch_ros.actions import SetParameter  # reason: launch-time only

    actions: list[object] = [
        # `bond_timeout` cannot go in the params file: upstream
        # `navigation_launch.py` hands `lifecycle_manager_navigation` only
        # `{autostart, node_names}` and never the params file, so a
        # `lifecycle_manager_navigation:` block there is silently ignored. Set
        # as a scoped override instead; the servers that do not declare it
        # ignore the override.
        GroupAction(
            scoped=True,
            actions=[
                SetParameter(name="bond_timeout", value=BOND_TIMEOUT_S),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(_upstream_navigation_launch()),
                    launch_arguments={
                        "params_file": resolved_params,
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "autostart": LaunchConfiguration("autostart"),
                        "use_composition": LaunchConfiguration("use_composition"),
                    }.items(),
                ),
            ],
        )
    ]
    payload_scan_filter = LaunchConfiguration("payload_scan_filter").perform(context)
    if payload_scan_filter.strip().lower() in ("true", "1", "yes"):
        use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
        actions.extend(
            _payload_scan_filter_nodes(
                robot_yaml=robot_yaml,
                slam_backend=slam_backend,
                use_sim_time=use_sim_time.strip().lower() in ("true", "1", "yes"),
            )
        )
    cmd_vel_relay = LaunchConfiguration("cmd_vel_relay").perform(context)
    if cmd_vel_relay.strip().lower() in ("true", "1", "yes"):
        actions.extend(_cmd_vel_relay_nodes(robot_yaml=robot_yaml))
    return actions


# Used by ``test/test_nav2_launch.py`` for hermetic argument validation
# without spawning a real ROS 2 graph.
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "nav2_panda_mobile.yaml"
