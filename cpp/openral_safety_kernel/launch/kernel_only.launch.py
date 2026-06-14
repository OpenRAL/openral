#!/usr/bin/env python3
"""Stand-alone launch file for the C++ safety kernel.

Used by sim and HIL integration tests to bring up the kernel against a
real :class:`RobotDescription`. The launch's ``OpaqueFunction`` loads
the manifest via Pydantic, synthesises the envelope via
:func:`openral_safety.envelope_loader.compute_intersection` +
:func:`openral_safety.envelope_loader.kernel_params_from_envelope`, and
forwards each canonical field as a ROS parameter on the kernel node
(ADR-0020 PR-K — there is no envelope-file path).

Production deployments compose this into ``sim_e2e.launch.py`` with
the rest of the graph (rskill_runner_node, world_state_node, HAL,
deadman_watchdog).
"""

from __future__ import annotations

import os
import site

# Bootstrap the workspace venv so ``openral_core`` / ``openral_safety``
# import cleanly when ``ros2 launch`` runs under the system Python.
_VENV_SITE = os.environ.get("OPENRAL_VENV_SITE")
if _VENV_SITE and os.path.isdir(_VENV_SITE):
    site.addsitedir(_VENV_SITE)

from launch import LaunchContext, LaunchDescription  # noqa: E402
from launch.actions import DeclareLaunchArgument, OpaqueFunction  # noqa: E402
from launch.substitutions import LaunchConfiguration  # noqa: E402
from launch_ros.actions import LifecycleNode  # noqa: E402


def _launch_setup(context: LaunchContext, *_args: object, **_kwargs: object) -> list:
    """Resolve args, synthesise envelope params, spawn the kernel node."""
    from openral_core import RobotDescription  # noqa: PLC0415
    from openral_safety.envelope_loader import (  # noqa: PLC0415
        compute_intersection,
        kernel_params_from_envelope,
    )

    robot_yaml = LaunchConfiguration("robot_yaml").perform(context)
    estop_reset_cooldown_s = LaunchConfiguration("estop_reset_cooldown_s").perform(context)
    node_name = LaunchConfiguration("node_name").perform(context)

    description = RobotDescription.from_yaml(robot_yaml)
    description.validate_for_e2e_pipeline()
    envelope = compute_intersection(description, skill=None)
    params = kernel_params_from_envelope(envelope)
    params["estop_reset_cooldown_s"] = float(estop_reset_cooldown_s)

    return [
        LifecycleNode(
            package="openral_safety_kernel",
            executable="safety_kernel_node",
            name=node_name,
            namespace="",
            output="screen",
            parameters=[params],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """Return a launch description bringing up only the safety kernel."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_yaml",
                description=(
                    "Path to a RobotDescription manifest "
                    "(robots/<id>/robot.yaml). The launch synthesises the "
                    "kernel envelope from it via "
                    "openral_safety.envelope_loader."
                ),
            ),
            DeclareLaunchArgument(
                "estop_reset_cooldown_s",
                default_value="0.5",
                description="Cooldown between estop publish and a valid /openral/estop_reset.",
            ),
            DeclareLaunchArgument(
                "node_name",
                default_value="openral_safety_kernel",
                description="Lifecycle node name (override per-robot in production launches).",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
