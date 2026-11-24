r"""Real-hardware ros2_control bringup for the Unitree hg humanoids (G1, H2).

Starts NVIDIA's S0 graph — the C++ half of the split ``UnitreeHgRealHAL``
(``python/hal/src/openral_hal/unitree_real.py``) proposes into:

* ``controller_manager`` (200 Hz) on ``unitree_hg_ros2_control/UnitreeHgSystemInterface``
  (isaac_ros_robots), its ``robot_description`` per model: G1 — NVIDIA's
  ``g1_real_hardware.urdf.xacro`` with ``with_hands:=false``, exactly the 29 joints
  ``robots/g1/robot.yaml`` declares; H2 — this package's ``urdf/h2_real_hardware.urdf.xacro``
  (NVIDIA ships no H2 description) around the pinned unitree_ros ``H2.urdf``
  (``openral_hal._unitree_h2_assets.ensure_h2_urdf``), ``robot_variant: h2``, the 31 joints
  of ``robots/h2/robot.yaml``;
* ``joint_state_broadcaster`` + ``imu_sensor_broadcaster``;
* the S0 chain from isaac_ros_deploy: ``safety_controller`` ←
  ``inference_controller`` (the whole-body ONNX policy; legs) +
  ``upper_body_forward_joint_command_controller`` (what OpenRAL commands); the first two
  run through OpenRAL's guards (``GuardedSafetyController``, ``GuardedInferenceController``).
  A policy file that sets the stale-input watchdog (``topic_input_timeout_ms``) on any other
  inference controller type is refused: the pinned Jazzy release would silently ignore it;
* ``freeze_controller``, loaded inactive: what the safety controller's out-of-domain
  detector and OpenRAL's ``/openral/estop`` both hand every joint to.

It does **not** include upstream's ``unitree_g1_bringup`` launch: that package hard-depends
on cuMotion, teleop and MuJoCo packages a real deploy never uses, loads a 43-DoF
(Dex3-hands) controller config that would not activate against a hands-less G1, and
offers no seam to swap the policy's controller parameters. The spawn sequence mirrors it
(load everything inactive, inference last; then activate the runtime group in one switch).

Which S0 policy runs is ``policy:=<stem>`` → ``config/<robot_model>/policies/<stem>.yaml``
(G1: GR00T Decoupled WBC by default, ``agile_velocity`` selectable; H2: ``agile_velocity``,
NVIDIA's ``nvidia/agile_wbc_unitree_h2``), read through
``openral_hal.unitree_s0_policy`` — the same reader ``UnitreeHgRealHAL`` takes its joint
split from. The file's ``controllers`` block is loaded after
``config/<robot_model>/controller_manager.yaml``; a ``status: placeholder`` policy is
refused. The LEAPP bundle comes from ``inference_controller_config_path:=`` if given, else
the policy's ``data_package`` share (G1 AGILE), else its ``bundle_dir`` (G1 GR00T: built by
``tools/build_gr00t_wbc_leapp.py``; H2 AGILE: the Hub download), checked against the
policy's ``sha256`` pins. A selected ``policy:=`` other than the manifest's ``s0_policy``
must be matched on the HAL side (``--hal s0_policy=<stem>``).

**Which robot.** ``deploy run`` includes this file through ``hal.real_bringup`` and forwards
no arguments, so under it ``robot_model`` and the default ``policy`` come from the deploying
manifest (its ``hal.parameters.defaults``, via the ``robot_yaml`` launch configuration
``deploy run`` declares) — the same values ``build_hal`` hands ``UnitreeHgRealHAL``. An
explicit ``robot_model:=`` that contradicts that manifest is refused. A hand run has no
manifest and must name ``robot_model:=`` itself: the G1 and H2 graphs are not
interchangeable (different joint tables), so there is no default to guess.

**The policy starts blended out.** ``safety_controller.blend_ratio`` is 0.0 and nothing
here raises it: with the robot on its gantry and a velocity command flowing, the operator
runs ``ros2 param set /safety_controller blend_ratio 1.0``. Activating the hardware
interface also releases Unitree's built-in locomotion service (``MotionSwitcherClient``
``ReleaseMode``), so the robot must be supported before this launch starts.

**Host-confined graph first.** Before composing any node the launch applies
``openral_core.ros_graph_scope.require_host_confined_real_bringup`` to its environment: an unset or
``SUBNET`` ``ROS_AUTOMATIC_DISCOVERY_RANGE`` is refused unless ``OPENRAL_ALLOW_NETWORK_ROS=1``
under SROS2 Enforce (ADR-0130), and an opted-in scope is logged on a hand run; export ``LOCALHOST``
for a hand run.

The one host fact, the NIC wired to the robot, comes from
``$OPENRAL_UNITREE_NETWORK_INTERFACE`` (default ``eth0``).

Usage (under the OpenRAL venv interpreter — this file imports ``openral_hal``, which a
bare ``ros2 launch`` running ``/opt/ros``'s system Python cannot see; ``openral deploy
run`` does this for you)::

    OPENRAL_UNITREE_NETWORK_INTERFACE=enp3s0 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
        .venv/bin/python "$(command -v ros2)" launch openral_hal_unitree \
        real_bringup.launch.py robot_model:=g1 policy:=gr00t_decoupled_wbc
    # H2 (policy defaults to agile_velocity):
    ... launch openral_hal_unitree real_bringup.launch.py robot_model:=h2
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

try:
    from openral_core.exceptions import ROSConfigError
    from openral_core.ros_graph_scope import require_host_confined_real_bringup
    from openral_core.ros_security import ENCLAVE_OVERRIDE_ENV as _ENCLAVE_OVERRIDE
    from openral_core.schemas import RobotDescription
    from openral_hal._unitree_h2_assets import ensure_h2_urdf
    from openral_hal.unitree_s0_policy import (
        load_s0_policy,
        policy_parameters,
        require_watchdog_enforced,
        resolve_leapp_config,
    )
except ImportError as exc:  # pragma: no cover - only under a non-venv interpreter
    # A bare `ros2 launch` parses this file under /opt/ros's system python3, which
    # does not see the OpenRAL venv. `openral deploy run` already runs the launch
    # under the venv interpreter (openral_cli.deploy_sim._ros2_argv_head).
    raise ImportError(
        f"openral_hal_unitree real_bringup.launch.py needs the OpenRAL venv with a current "
        f"openral_core / openral_hal ({exc}): run it via `openral deploy run`, or "
        f'`.venv/bin/python "$(command -v ros2)" launch openral_hal_unitree '
        f"real_bringup.launch.py` from the repo root."
    ) from exc

if TYPE_CHECKING:  # pragma: no cover - launch is a ROS-only dep
    from launch import LaunchContext, LaunchDescription

#: Robot models this package ships ``config/<model>/`` for.
_ROBOT_MODELS = ("g1", "h2")
#: Per-model S0 policy for a hand run that names none: G1 — GR00T Decoupled WBC, the S0
#: bake-off winner; H2 — NVIDIA WBC-AGILE, the only H2 checkpoint. Each must equal
#: robots/<model>/robot.yaml `hal.parameters.defaults.s0_policy` (test-enforced), so the
#: HAL refuses the joints this launch hands to the policy.
_DEFAULT_POLICIES: dict[str, str] = {"g1": "gr00t_decoupled_wbc", "h2": "agile_velocity"}
#: The launch configuration `deploy run` (deploy_e2e.launch.py) declares for the manifest.
_DEPLOY_ROBOT_YAML = "robot_yaml"
#: Host NIC wired to the robot; NVIDIA's xacro default. Overridden by the env var below.
_DEFAULT_NETWORK_INTERFACE = "eth0"
_NETWORK_INTERFACE_ENV = "OPENRAL_UNITREE_NETWORK_INTERFACE"
_PACKAGE = "openral_hal_unitree"
#: (ament package, xacro) per model — the real-hardware description. NVIDIA's for the G1;
#: this package's for the H2 (NVIDIA ships none).
_XACRO: dict[str, tuple[str, str]] = {
    "g1": ("unitree_g1_description", "urdf/g1_real_hardware.urdf.xacro"),
    "h2": (_PACKAGE, "urdf/h2_real_hardware.urdf.xacro"),
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

    # Before any node is composed: a hand-run bringup on SUBNET would put the G1 on the
    # LAN graph (issue #556).
    require_host_confined_real_bringup(
        _PACKAGE, os.environ, included_by_deploy="hal_mode" in context.launch_configurations
    )
    robot_model, policy_name = resolve_model_and_policy(
        context.perform_substitution(_config("robot_model")),
        context.perform_substitution(_config("policy")),
        context.launch_configurations.get(_DEPLOY_ROBOT_YAML),
    )
    network_interface = context.perform_substitution(_config("network_interface"))
    override = context.perform_substitution(_config("inference_controller_config_path"))

    share = Path(get_package_share_directory(_PACKAGE))
    base_params = share / "config" / robot_model / "controller_manager.yaml"
    policy = load_s0_policy(share / "config" / robot_model / "policies" / f"{policy_name}.yaml")
    require_watchdog_enforced(policy, yaml.safe_load(base_params.read_text(encoding="utf-8")))
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
                    *(f" {arg}" for arg in _xacro_mappings(robot_model)),
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
        # It starts after the spawner exits, outside the deploy launch's HAL-enclave
        # scope, so it pins the enclave it was included under (ADR-0130): only the
        # HAL enclave may call /controller_manager/* under SROS2.
        additional_env=(
            {_ENCLAVE_OVERRIDE: enclave} if (enclave := os.environ.get(_ENCLAVE_OVERRIDE)) else {}
        ),
    )

    def _after_load(event: Any, _ctx: Any) -> list[Any]:  # reason: launch event types are untyped
        if event.returncode == 0:
            return [activate]
        reason = (
            f"{_PACKAGE}: loading the {robot_model} controllers failed "
            f"(spawner exit {event.returncode}); the S0 graph is not up."
        )
        return [LogInfo(msg=reason), Shutdown(reason=reason)]

    def _after_activate(
        event: Any, _ctx: Any
    ) -> list[Any]:  # reason: launch event types are untyped
        if event.returncode == 0:
            return [LogInfo(msg=f"{_PACKAGE}: {robot_model} S0 chain active.")]
        reason = (
            f"{_PACKAGE}: activating the {robot_model} S0 chain failed "
            f"(switch_controllers exit {event.returncode}); the S0 graph is not up."
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
        RegisterEventHandler(OnProcessExit(target_action=activate, on_exit=_after_activate)),
    ]


def resolve_model_and_policy(
    robot_model: str, policy: str, robot_yaml: str | None
) -> tuple[str, str]:
    """``(robot_model, policy)`` from the launch arguments and the deploying manifest.

    ``robot_yaml`` is the manifest ``deploy run`` deploys (its ``robot_yaml`` launch
    configuration), ``None`` on a hand run. An empty argument takes the manifest's
    ``hal.parameters.defaults`` value; a hand run's empty ``policy`` takes the model's
    ``_DEFAULT_POLICIES`` entry. See the module docstring ("Which robot").

    Raises:
        ROSConfigError: No model to resolve, an unsupported one, or an explicit
            ``robot_model:=`` that contradicts the deploying manifest.
    """
    defaults: dict[str, Any] = {}
    if robot_yaml:
        defaults = dict(RobotDescription.from_yaml(robot_yaml).hal.parameters.defaults)
    manifest_model = str(defaults.get("robot_model", ""))
    if robot_model and manifest_model and robot_model != manifest_model:
        raise ROSConfigError(
            f"robot_model:={robot_model!r} contradicts the deployed manifest {robot_yaml} "
            f"(hal.parameters.defaults.robot_model: {manifest_model!r})."
        )
    model = robot_model or manifest_model
    if model not in _ROBOT_MODELS:
        raise ROSConfigError(
            f"robot_model:={model!r} is not one of {list(_ROBOT_MODELS)} "
            f"({_PACKAGE} ships config/<model>/ only for those)"
            + ("; a hand run must name it." if not model else ".")
        )
    chosen = policy or str(defaults.get("s0_policy") or _DEFAULT_POLICIES[model])
    return model, chosen


def _xacro_mappings(robot_model: str) -> list[str]:
    """Model-specific xacro arguments beyond ``network_interface``."""
    if robot_model == "h2":
        # The pinned, SHA-256-checked unitree_ros H2.urdf (fetched once into the cache).
        return [f"body_urdf:={ensure_h2_urdf()}"]
    return ["with_hands:=false"]


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
                default_value="",
                choices=["", *_ROBOT_MODELS],
                description=(
                    "Unitree hg-family model; selects config/<model>/. Empty = the deploying "
                    "manifest's (required on a hand run)."
                ),
            ),
            DeclareLaunchArgument(
                "policy",
                default_value="",
                description=(
                    "S0 policy file stem under config/<model>/policies/. Empty = the "
                    "manifest's s0_policy, else the model's default."
                ),
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
