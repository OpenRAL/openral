r"""Real-hardware ros2_control bringup for the Unitree hg humanoids (G1 now).

Starts NVIDIA's S0 graph — the C++ half of the split ``UnitreeHgRealHAL``
(``python/hal/src/openral_hal/unitree_real.py``) proposes into:

* ``controller_manager`` (200 Hz) on ``unitree_hg_ros2_control/UnitreeHgSystemInterface``
  (isaac_ros_robots), its ``robot_description`` from NVIDIA's
  ``g1_real_hardware.urdf.xacro`` with ``with_hands:=false`` — exactly the 29 joints
  ``robots/g1/robot.yaml`` declares;
* ``joint_state_broadcaster`` + ``imu_sensor_broadcaster``;
* the S0 chain from isaac_ros_deploy: ``safety_controller`` ←
  ``inference_controller`` (the whole-body ONNX policy; legs) +
  ``upper_body_forward_joint_command_controller`` (what OpenRAL commands);
* ``freeze_controller``, loaded inactive: what the safety controller's out-of-domain
  detector and OpenRAL's ``/openral/estop`` both hand every joint to.

It does **not** include upstream's ``unitree_g1_bringup`` launch: that package hard-depends
on cuMotion, teleop and MuJoCo packages a real deploy never uses, loads a 43-DoF
(Dex3-hands) controller config that would not activate against a hands-less G1, and
offers no seam to swap the policy's controller parameters. The spawn sequence mirrors it
(load everything inactive, inference last; then activate the runtime group in one switch).

Which S0 policy runs is ``policy:=<stem>`` → ``config/<robot_model>/policies/<stem>.yaml``
(GR00T Decoupled WBC by default; ``agile_velocity`` selectable), read through
``openral_hal.unitree_s0_policy`` — the same reader ``UnitreeHgRealHAL`` takes its joint
split from. The file's ``controllers`` block is loaded after
``config/<robot_model>/controller_manager.yaml``; a ``status: placeholder`` policy is
refused. The LEAPP bundle comes from ``inference_controller_config_path:=`` if given, else
the policy's ``data_package`` share (AGILE), else its locally built ``bundle_dir`` (GR00T:
``tools/build_gr00t_wbc_leapp.py``). A selected ``policy:=`` other than the manifest's
``s0_policy`` must be matched on the HAL side (``--hal s0_policy=<stem>``).

**The policy starts blended out.** ``safety_controller.blend_ratio`` is 0.0 and nothing
here raises it: with the robot on its gantry and a velocity command flowing, the operator
runs ``ros2 param set /safety_controller blend_ratio 1.0``. Activating the hardware
interface also releases Unitree's built-in locomotion service (``MotionSwitcherClient``
``ReleaseMode``), so the robot must be supported before this launch starts.

``deploy run`` includes this file through ``hal.real_bringup`` with no arguments, so every
default must be right for the robot; the one host fact, the NIC wired to the robot, comes
from ``$OPENRAL_UNITREE_NETWORK_INTERFACE`` (default ``eth0``).

Usage::

    OPENRAL_UNITREE_NETWORK_INTERFACE=enp3s0 \
        ros2 launch openral_hal_unitree real_bringup.launch.py \
        robot_model:=g1 policy:=gr00t_decoupled_wbc
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from openral_core.exceptions import ROSConfigError
from openral_hal.unitree_s0_policy import (
    load_s0_policy,
    policy_parameters,
    resolve_leapp_config,
)

if TYPE_CHECKING:  # pragma: no cover - launch is a ROS-only dep
    from launch import LaunchContext, LaunchDescription

#: Robot models this package ships ``config/<model>/`` for. H2 joins here.
_ROBOT_MODELS = ("g1",)
_DEFAULT_ROBOT_MODEL = "g1"
#: GR00T Decoupled WBC — the G1 S0 bake-off winner. Must equal robots/g1/robot.yaml
#: `hal.parameters.defaults.s0_policy` (test-enforced), so the HAL refuses the joints
#: this launch hands to the policy.
_DEFAULT_POLICY = "gr00t_decoupled_wbc"
#: Host NIC wired to the robot; NVIDIA's xacro default. Overridden by the env var below.
_DEFAULT_NETWORK_INTERFACE = "eth0"
_NETWORK_INTERFACE_ENV = "OPENRAL_UNITREE_NETWORK_INTERFACE"
_PACKAGE = "openral_hal_unitree"
#: (ament package, xacro) per model — NVIDIA's real-hardware description.
_XACRO: dict[str, tuple[str, str]] = {
    "g1": ("unitree_g1_description", "urdf/g1_real_hardware.urdf.xacro"),
}
_BROADCASTERS = ("joint_state_broadcaster", "imu_sensor_broadcaster")
_HOLD_CONTROLLER = "freeze_controller"
#: Must match `openral_hal.unitree_real.UnitreeHgRealHAL.controller_names()`.
_S0_CHAIN = (
    "safety_controller",
    "inference_controller",
    "upper_body_forward_joint_command_controller",
)
#: Upstream's default spawner timeout: the inference controller's first configure
#: compiles the model (TensorRT warmup) and can take tens of seconds.
_SPAWNER_TIMEOUT_S = "60"


def _launch_setup(context: LaunchContext) -> list[Any]:
    from ament_index_python.packages import get_package_share_directory
    from launch.actions import (
        ExecuteProcess,
        LogInfo,
        RegisterEventHandler,
        Shutdown,
    )
    from launch.event_handlers import OnProcessExit
    from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
    from launch_ros.actions import Node
    from launch_ros.descriptions import ParameterValue

    robot_model = context.perform_substitution(_config("robot_model"))
    if robot_model not in _ROBOT_MODELS:
        raise ROSConfigError(
            f"robot_model:={robot_model!r} is not one of {list(_ROBOT_MODELS)} "
            f"({_PACKAGE} ships config/<model>/ only for those)."
        )
    policy_name = context.perform_substitution(_config("policy"))
    network_interface = context.perform_substitution(_config("network_interface"))
    override = context.perform_substitution(_config("inference_controller_config_path"))

    share = Path(get_package_share_directory(_PACKAGE))
    base_params = share / "config" / robot_model / "controller_manager.yaml"
    policy = load_s0_policy(share / "config" / robot_model / "policies" / f"{policy_name}.yaml")
    leapp = resolve_leapp_config(policy, override, get_package_share_directory)
    with tempfile.NamedTemporaryFile(
        "w", prefix=f"openral_{robot_model}_{policy_name}_", suffix=".yaml", delete=False
    ) as fh:
        yaml.safe_dump(policy_parameters(policy, leapp), fh)
        policy_params = fh.name

    description_pkg, xacro_rel = _XACRO[robot_model]
    robot_description = {
        "robot_description": ParameterValue(
            Command(
                [
                    PathJoinSubstitution([FindExecutable(name="xacro")]),
                    " ",
                    str(Path(get_package_share_directory(description_pkg)) / xacro_rel),
                    " network_interface:=",
                    network_interface,
                    " with_hands:=false",
                ]
            ),
            value_type=str,
        )
    }
    param_files = [str(base_params), policy_params]

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, *param_files],
        output="both",
        emulate_tty=True,
    )
    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        # Distinct from the robot_state_publisher `deploy run` derives from the manifest.
        name="unitree_robot_state_publisher",
        parameters=[robot_description],
        output="log",
    )
    # Load every controller inactive (inference last: its configure compiles the model),
    # so freeze_controller exists before safety_controller ever runs.
    load_all = Node(
        package="controller_manager",
        executable="spawner",
        name="openral_unitree_spawner",
        arguments=[
            *_BROADCASTERS,
            _HOLD_CONTROLLER,
            "safety_controller",
            "upper_body_forward_joint_command_controller",
            "inference_controller",
            *(arg for f in param_files for arg in ("--param-file", f)),
            "--inactive",
            "--controller-manager-timeout",
            _SPAWNER_TIMEOUT_S,
            "--service-call-timeout",
            _SPAWNER_TIMEOUT_S,
        ],
        output="screen",
    )
    # Then activate broadcasters + the S0 chain in ONE strict switch; freeze stays inactive.
    activate = ExecuteProcess(
        cmd=[
            "ros2",
            "control",
            "switch_controllers",
            "--activate",
            *_BROADCASTERS,
            *_S0_CHAIN,
            "--strict",
        ],
        output="screen",
    )

    def _after_load(event: Any, _ctx: Any) -> list[Any]:  # reason: launch event types are untyped
        if event.returncode == 0:
            return [activate]
        reason = (
            f"{_PACKAGE}: loading the {robot_model} controllers failed "
            f"(spawner exit {event.returncode}); the S0 graph is not up."
        )
        return [LogInfo(msg=reason), Shutdown(reason=reason)]

    return [
        LogInfo(
            msg=(
                f"{_PACKAGE}: {robot_model} on {network_interface}, S0 policy "
                f"{policy_name!r} ({leapp}); safety_controller.blend_ratio starts at 0.0."
            )
        ),
        controller_manager,
        state_publisher,
        load_all,
        RegisterEventHandler(OnProcessExit(target_action=load_all, on_exit=_after_load)),
    ]


def _config(name: str) -> Any:  # reason: LaunchConfiguration is a ROS-only type
    from launch.substitutions import LaunchConfiguration

    return LaunchConfiguration(name)


def generate_launch_description() -> LaunchDescription:
    """Declare the bringup's arguments and defer graph construction to launch time."""
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, OpaqueFunction
    from launch.substitutions import EnvironmentVariable

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_model",
                default_value=_DEFAULT_ROBOT_MODEL,
                choices=list(_ROBOT_MODELS),
                description="Unitree hg-family model; selects config/<model>/.",
            ),
            DeclareLaunchArgument(
                "policy",
                default_value=_DEFAULT_POLICY,
                description="S0 policy file stem under config/<model>/policies/.",
            ),
            DeclareLaunchArgument(
                "network_interface",
                default_value=EnvironmentVariable(
                    _NETWORK_INTERFACE_ENV, default_value=_DEFAULT_NETWORK_INTERFACE
                ),
                description=(
                    "Host NIC wired to the robot's 192.168.123.0/24 network "
                    f"(default: ${_NETWORK_INTERFACE_ENV}, else {_DEFAULT_NETWORK_INTERFACE})."
                ),
            ),
            DeclareLaunchArgument(
                "inference_controller_config_path",
                default_value="",
                description="Absolute LEAPP yaml for the S0 policy; empty = its data package.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
