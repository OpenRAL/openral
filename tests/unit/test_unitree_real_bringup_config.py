# SPDX-License-Identifier: Apache-2.0
"""`openral_hal_unitree`'s controller configs and launch must agree with the G1 / H2
manifests and with `UnitreeHgRealHAL`.

The bringup's YAML is a hand-maintained derivative of NVIDIA's `unitree_g1_bringup`
config (isaac_ros_robots @ 520aa94f) — trimmed to the 29-DoF hands-less G1, and for the
H2 (which NVIDIA supports only in the hardware interface) rebuilt around OpenRAL's own
`urdf/h2_real_hardware.urdf.xacro`; nothing else compares it with `robots/<id>/robot.yaml`
or with the HAL's constants. A drift is
silent on hardware — a controller claiming a joint the hardware does not export never
activates, an upper-body joint the HAL thinks it owns but the policy also commands
fights the policy, an e-stop list that misses a controller leaves it running. These
tests read the real files (no ROS install needed); the launch-description test, the
upstream parity tests and the H2 export checks run where those sources exist and skip
with a reason otherwise.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_hal.ros_control import ControllerKind
from openral_hal.unitree_real import UnitreeHgRealHAL
from openral_hal.unitree_s0_policy import (
    load_s0_policy,
    policy_parameters,
    require_watchdog_enforced,
    resolve_leapp_config,
    s0_policy_path,
)

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "packages" / "openral_hal_unitree"
_LAUNCH = _PKG / "launch" / "real_bringup.launch.py"
_G1_CONFIG = _PKG / "config" / "g1"
_CONTROLLER_MANAGER = _G1_CONFIG / "controller_manager.yaml"
_POLICIES = sorted((_G1_CONFIG / "policies").glob("*.yaml"))
_H2_CONFIG = _PKG / "config" / "h2"
_H2_XACRO = _PKG / "urdf" / "h2_real_hardware.urdf.xacro"
#: Every model the bringup ships a config/<model>/ for.
_MODELS = ("g1", "h2")
#: (model, policy file) for every shipped policy of every model.
_ALL_POLICIES = [
    (m, p) for m in _MODELS for p in sorted((_PKG / "config" / m / "policies").glob("*.yaml"))
]

#: The G1 joints of NVIDIA's real-hardware ros2_control xacro
#: (`unitree_g1_description/urdf/g1_real_ros2_control.urdf.xacro` @ 520aa94f, the
#: `UnitreeG1System` block — hands excluded, they are a separate `with_hands` system).
_UPSTREAM_XACRO_ENV = "OPENRAL_ISAAC_ROS_ROBOTS_SRC"


def _yaml(path: Path) -> Any:  # reason: arbitrary YAML document
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _manifest(model: str = "g1") -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO / "robots" / model / "robot.yaml"))


def _manifest_joints(model: str = "g1") -> list[str]:
    return [j.name for j in _manifest(model).joints]


def _cm(model: str) -> Path:
    return _PKG / "config" / model / "controller_manager.yaml"


def _params(doc: dict[str, Any], controller: str) -> dict[str, Any]:
    block = doc[controller]["ros__parameters"]
    assert isinstance(block, dict)
    return block


def _load_launch_module() -> types.ModuleType:
    """Import the launch file by path. Its ROS imports are deferred, so this needs no ROS."""
    spec = importlib.util.spec_from_file_location("openral_unitree_real_bringup", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_tool() -> types.ModuleType:
    """`tools/build_gr00t_wbc_leapp.py` by path (its constants are the bundle's contract)."""
    path = _REPO / "tools" / "build_gr00t_wbc_leapp.py"
    spec = importlib.util.spec_from_file_location("build_gr00t_wbc_leapp_cfg", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ready_policies() -> list[Any]:  # reason: pytest.param's ParameterSet is private
    return [
        pytest.param(m, p, id=f"{m}-{p.stem}")
        for m, p in _ALL_POLICIES
        if _yaml(p)["openral_policy"]["status"] == "ready"
    ]


def _resolve_gains(base: dict[str, Any], policy: dict[str, Any], gain: str) -> list[Any]:
    """Emulates isaac_ros_deploy gain_utils: patterns in param-name order, later match wins."""
    patterns = dict(sorted({**base[gain], **policy[gain]}.items()))  # std::map order
    resolved = []
    for joint in policy["joints"]:
        value = None
        for pattern, v in patterns.items():
            if re.fullmatch(pattern, joint):
                value = v
        resolved.append(value)
    return resolved


# ── controller_manager.yaml ───────────────────────────────────────────────────


@pytest.mark.parametrize("model", _MODELS)
def test_whole_body_controllers_cover_exactly_the_manifest_joints(model: str) -> None:
    """Broadcaster, safety and freeze each span the manifest joints — no hands, no gaps."""
    doc = _yaml(_cm(model))
    manifest = _manifest_joints(model)
    for controller in ("joint_state_broadcaster", "safety_controller", "freeze_controller"):
        joints = _params(doc, controller)["joints"]
        assert len(joints) == len(set(joints)), f"{controller} lists a joint twice"
        assert set(joints) == set(manifest), (
            f"{controller}: extra {sorted(set(joints) - set(manifest))}, "
            f"missing {sorted(set(manifest) - set(joints))}"
        )


@pytest.mark.parametrize("model", _MODELS)
def test_the_safety_controller_clamps_to_the_manifest_joint_limits(model: str) -> None:
    """Hazard-log Entry 105 C1: the guard's limits are the manifest's, joint for joint.

    GuardedSafetyController clamps every position command to ``joint_position_limits`` (the
    gtest proves the clamp); this pins the values it clamps to. A drifted limit here would
    let an S0 target past the manifest's range on hardware, where the kernel never sees it.
    """
    params = _params(_yaml(_cm(model)), "safety_controller")
    limits = params["joint_position_limits"]
    manifest = {j.name: j.position_limits for j in _manifest(model).joints}
    assert len(limits["lower"]) == len(limits["upper"]) == len(params["joints"])
    for joint, lower, upper in zip(params["joints"], limits["lower"], limits["upper"], strict=True):
        expected = manifest[joint]
        assert expected is not None, f"{joint} has no manifest position_limits"
        assert (lower, upper) == tuple(expected), joint


@pytest.mark.parametrize("model", _MODELS)
def test_every_declared_controller_type_is_a_known_plugin(model: str) -> None:
    """Upstream plugins, except safety_controller and inference_controller, which run
    through OpenRAL's guards (#548, #565)."""
    types_ = {
        name: spec["type"]
        for name, spec in _params(_yaml(_cm(model)), "controller_manager").items()
        if isinstance(spec, dict)
    }
    assert types_ == {
        "joint_state_broadcaster": "joint_state_broadcaster/JointStateBroadcaster",
        "imu_sensor_broadcaster": "imu_sensor_broadcaster/IMUSensorBroadcaster",
        "freeze_controller": "isaac_ros_deploy_ros2_control/FreezeController",
        "safety_controller": "openral_hal_unitree/GuardedSafetyController",
        "inference_controller": "openral_hal_unitree/GuardedInferenceController",
        "upper_body_forward_joint_command_controller": (
            "isaac_ros_deploy_ros2_control/ForwardJointCommandController"
        ),
    }


@pytest.mark.parametrize("model", _MODELS)
def test_the_safety_controllers_emergency_handover_matches_the_hal_estop(model: str) -> None:
    """Upstream's own out-of-domain swap and `/openral/estop` move the same controllers."""
    ood = _params(_yaml(_cm(model)), "safety_controller")["out_of_domain_detection"]
    hal = UnitreeHgRealHAL(robot_model=model)
    assert ood["emergency_controller"] == hal.freeze_controller
    assert ood["deactivate_controllers"] == hal.controller_names()


@pytest.mark.parametrize("model", _MODELS)
def test_the_policy_starts_blended_out(model: str) -> None:
    """Nothing in OpenRAL raises blend_ratio; the operator does, on the gantry."""
    assert _params(_yaml(_cm(model)), "safety_controller")["blend_ratio"] == 0.0


@pytest.mark.parametrize("model", _MODELS)
def test_the_blend_ramp_is_within_the_guards_cap(model: str) -> None:
    """GuardedSafetyController refuses to configure outside (0, 1.0] /s (issue #548)."""
    speed = _params(_yaml(_cm(model)), "safety_controller")["max_blend_ratio_speed"]
    assert 0.0 < speed <= 1.0


def test_the_guarded_plugin_is_exported_under_the_configured_name() -> None:
    """The type the YAML names must be the class the plugin XML exports, in a library
    CMake builds; a mismatch is a controller that fails to load on the robot."""
    xml = (_PKG / "openral_hal_unitree_controllers.xml").read_text(encoding="utf-8")
    for guard in ("GuardedSafetyController", "GuardedInferenceController"):
        assert f'name="openral_hal_unitree/{guard}"' in xml
        assert f'type="openral_hal_unitree::{guard}"' in xml
    library = re.search(r'<library path="([^"]+)"', xml)
    assert library is not None
    cmake = (_PKG / "CMakeLists.txt").read_text(encoding="utf-8")
    assert f"add_library({library.group(1)} SHARED" in cmake
    assert "controller_interface openral_hal_unitree_controllers.xml" in cmake


@pytest.mark.parametrize("model", _MODELS)
def test_the_upper_body_controller_listens_where_the_hal_publishes(model: str) -> None:
    fwd = _params(_yaml(_cm(model)), "upper_body_forward_joint_command_controller")
    # Exactly one input mode may be set upstream; OpenRAL uses the chunked one.
    assert fwd["joint_command_trajectory_topic"] == "~/joint_commands_trajectory"
    assert "joint_command_topic" not in fwd and "joint_state_topic" not in fwd
    resolved = "/upper_body_forward_joint_command_controller/joint_commands_trajectory"
    bindings = UnitreeHgRealHAL(robot_model=model).command_bindings()
    assert bindings[resolved] is ControllerKind.JOINT_COMMAND_TRAJECTORY
    # Holds its activation pose until commanded (no upstream `default_position` homing).
    assert "default_position" not in fwd


@pytest.mark.parametrize("model", _MODELS)
def test_manager_rate_is_upstreams_200_hz(model: str) -> None:
    """The README's rate table is pinned to the file, so it cannot quietly go stale."""
    assert _params(_yaml(_cm(model)), "controller_manager")["update_rate"] == 200


# ── policies/*.yaml ───────────────────────────────────────────────────────────


def test_both_policies_are_ready_and_gr00t_wbc_is_the_default() -> None:
    """GR00T Decoupled WBC won the G1 S0 bake-off; AGILE stays selectable."""
    statuses = {p.stem: _yaml(p)["openral_policy"]["status"] for p in _POLICIES}
    assert statuses == {"agile_velocity": "ready", "gr00t_decoupled_wbc": "ready"}
    desc = RobotDescription.from_yaml(str(_REPO / "robots" / "g1" / "robot.yaml"))
    assert desc.hal.parameters.defaults["s0_policy"] == "gr00t_decoupled_wbc"
    assert _load_launch_module()._DEFAULT_POLICIES["g1"] == "gr00t_decoupled_wbc"
    assert UnitreeHgRealHAL().s0_policy == "gr00t_decoupled_wbc"


@pytest.mark.parametrize(("model", "policy"), _ready_policies())
def test_a_ready_policy_splits_the_body_disjointly_between_s0_and_openral(
    model: str, policy: Path
) -> None:
    doc = _yaml(policy)
    s0 = doc["openral_policy"]["s0_joints"]
    upper = _params(doc["controllers"], "upper_body_forward_joint_command_controller")["joints"]
    manifest = set(_manifest_joints(model))
    assert not set(s0) & set(upper)
    assert set(s0) | set(upper) == manifest
    assert len(s0) + len(upper) == len(manifest)


def test_agile_split_matches_the_hal_and_the_upstream_action_set() -> None:
    doc = _yaml(_G1_CONFIG / "policies" / "agile_velocity.yaml")
    upper = _params(doc["controllers"], "upper_body_forward_joint_command_controller")["joints"]
    hal = UnitreeHgRealHAL(s0_policy="agile_velocity")
    assert upper == hal.upper_body_joints()
    assert set(doc["openral_policy"]["s0_joints"]) == set(hal.s0_joints())
    # Upstream's `safety_controller_lower_body`: "12 legs + waist roll + waist pitch.
    # Matches the AGILE action output set."
    s0 = set(doc["openral_policy"]["s0_joints"])
    assert {"waist_roll_joint", "waist_pitch_joint"} <= s0
    assert "waist_yaw_joint" not in s0
    assert len(s0) == 14


def test_gr00t_split_matches_the_bundle_action_and_the_hal() -> None:
    """GR00T's 15-D action is legs + all three waist axes; OpenRAL keeps the 14 arm joints."""
    tool = _load_tool()
    doc = _yaml(_G1_CONFIG / "policies" / "gr00t_decoupled_wbc.yaml")
    assert doc["openral_policy"]["s0_joints"] == list(tool.S0_JOINTS)
    hal = UnitreeHgRealHAL()
    assert set(hal.s0_joints()) == set(tool.S0_JOINTS)
    upper = _params(doc["controllers"], "upper_body_forward_joint_command_controller")["joints"]
    assert upper == hal.upper_body_joints()
    assert all("shoulder" in j or "elbow" in j or "wrist" in j for j in upper)
    assert doc["openral_policy"]["leapp_config"] == f"{tool.MODEL_NAME}.yaml"
    assert Path(doc["openral_policy"]["bundle_dir"]).expanduser() == tool.DEFAULT_BUNDLE_DIR
    # 200 Hz manager / decimation = the policy's 50 Hz.
    inference = _params(doc["controllers"], "inference_controller")
    assert 200 / inference["decimation"] == tool.CONTROL_HZ


def test_gr00t_arm_gains_resolve_to_the_g1_motor_gains() -> None:
    """Emulates isaac_ros_deploy gain_utils: patterns in param-name order, later match wins."""
    base = _params(_yaml(_CONTROLLER_MANAGER), "upper_body_forward_joint_command_controller")
    doc = _yaml(_G1_CONFIG / "policies" / "gr00t_decoupled_wbc.yaml")
    policy = _params(doc["controllers"], "upper_body_forward_joint_command_controller")
    expected_kp = [100.0, 100.0, 40.0, 40.0, 20.0, 20.0, 20.0] * 2
    expected_kd = [5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0] * 2
    for gain, expected in (("kp", expected_kp), ("kd", expected_kd)):
        assert _resolve_gains(base, policy, gain) == expected, gain


@pytest.mark.parametrize(("model", "policy"), _ready_policies())
def test_a_ready_policys_velocity_source_is_the_hal_twist_topic(model: str, policy: Path) -> None:
    doc = _yaml(policy)
    meta = doc["openral_policy"]
    inference = _params(doc["controllers"], "inference_controller")
    topic = meta["velocity_command_topic"]
    assert inference["source_to_topic"] == {"command/body/velocity": topic}
    assert inference["topic_input_sources"] == ["command/body/velocity"]
    # The Twist converter is upstream's default for this source; TwistStamped is the
    # alternative. The HAL publishes Twist, so this must say Twist.
    assert inference["source_message_type"] == {
        "command/body/velocity": meta["velocity_command_type"]
    }
    assert meta["velocity_command_type"] == "geometry_msgs/msg/Twist"
    hal = UnitreeHgRealHAL(robot_model=model, s0_policy=policy.stem)
    assert hal.command_bindings()[topic] is ControllerKind.TWIST
    # Writes through the safety controller like upstream's agile_velocity group.
    assert inference["command_prefix"] == "safety_controller"
    assert inference["command_suffix"] == "_raw"
    # A dead OpenRAL side must stop the walk: the input goes stale, the policy blanks.
    assert 0.0 < inference["topic_input_timeout_ms"] <= 1000.0
    # config_path is the launch's to fill; an empty one fails the controller's configure.
    assert inference["config_path"] == ""


# ── stale-input watchdog must be enforced, never silently ignored (#565) ──────


@pytest.mark.parametrize(("model", "policy"), _ready_policies())
def test_a_shipped_policys_watchdog_is_enforced_by_the_configured_type(
    model: str, policy: Path
) -> None:
    require_watchdog_enforced(load_s0_policy(policy), _yaml(_cm(model)))


def test_the_watchdog_keys_on_upstreams_inference_controller_are_refused() -> None:
    """The pinned Jazzy isaac_ros_deploy (v4.6) accepts and ignores them."""
    base = _yaml(_CONTROLLER_MANAGER)
    base["controller_manager"]["ros__parameters"]["inference_controller"]["type"] = (
        "isaac_ros_deploy_ros2_control/InferenceController"
    )
    policy = load_s0_policy(_G1_CONFIG / "policies" / "gr00t_decoupled_wbc.yaml")
    with pytest.raises(ROSConfigError, match="typed"):
        require_watchdog_enforced(policy, base)


_UPSTREAM_TYPE = "isaac_ros_deploy_ros2_control/InferenceController"


@pytest.mark.parametrize(
    "override",
    [
        # controller_manager's block in the policy file (it loads last)
        {
            "controller_manager": {
                "ros__parameters": {"inference_controller": {"type": _UPSTREAM_TYPE}}
            }
        },
        # the controller's own block: Jazzy's spawner sets the type from it too
        {"inference_controller": {"ros__parameters": {"type": _UPSTREAM_TYPE}}},
        # namespaced / wildcard keys
        {"/inference_controller": {"ros__parameters": {"type": _UPSTREAM_TYPE}}},
        {"/**": {"ros__parameters": {"inference_controller": {"type": _UPSTREAM_TYPE}}}},
        # a node nested under a wildcard namespace: `/**/inference_controller`
        {"/**": {"inference_controller": {"ros__parameters": {"type": _UPSTREAM_TYPE}}}},
    ],
    ids=["manager-block", "own-block", "namespaced", "wildcard", "nested-wildcard"],
)
def test_any_other_type_anywhere_in_the_policy_file_is_refused(
    tmp_path: Path, override: dict[str, dict[str, object]]
) -> None:
    doc = _yaml(_G1_CONFIG / "policies" / "agile_velocity.yaml")
    for key, block in override.items():
        params = block.get("ros__parameters")
        if key in doc["controllers"] and isinstance(params, dict):
            doc["controllers"][key]["ros__parameters"].update(params)
        else:
            doc["controllers"][key] = block
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ROSConfigError, match="typed"):
        require_watchdog_enforced(load_s0_policy(path), _yaml(_CONTROLLER_MANAGER))


@pytest.mark.parametrize(
    "timeout", [None, 0.0, -1.0, 500], ids=["missing", "zero", "negative", "int"]
)
def test_a_policy_without_a_positive_float_timeout_is_refused(
    tmp_path: Path, timeout: float | None
) -> None:
    """No watchdog on a real robot is refused, not silently accepted."""
    doc = _yaml(_G1_CONFIG / "policies" / "gr00t_decoupled_wbc.yaml")
    params = doc["controllers"]["inference_controller"]["ros__parameters"]
    if timeout is None:
        del params["topic_input_timeout_ms"]
    else:
        params["topic_input_timeout_ms"] = timeout
    path = tmp_path / "timeout.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ROSConfigError, match="topic_input_timeout_ms is"):
        require_watchdog_enforced(load_s0_policy(path), _yaml(_CONTROLLER_MANAGER))


# ── launch file (pure helpers; no ROS) ────────────────────────────────────────


@pytest.mark.parametrize("model", _MODELS)
def test_launch_chain_and_defaults_agree_with_the_hal_and_manifest(model: str) -> None:
    launch = _load_launch_module()
    assert launch._ROBOT_MODELS == _MODELS
    assert set(launch._XACRO) == set(_MODELS) == set(launch._DEFAULT_POLICIES)
    hal = UnitreeHgRealHAL(robot_model=model)
    assert list(launch._S0_CHAIN) == hal.controller_names()
    assert hal.freeze_controller == launch._HOLD_CONTROLLER
    desc = _manifest(model)
    assert desc.hal.real == "openral_hal.unitree_real:UnitreeHgRealHAL"
    assert desc.hal.real_bringup == f"{launch._PACKAGE}:{_LAUNCH.name}"
    assert desc.hal.parameters.defaults["robot_model"] == model
    assert desc.hal.parameters.defaults["s0_policy"] == launch._DEFAULT_POLICIES[model]
    assert hal.s0_policy == launch._DEFAULT_POLICIES[model]
    assert s0_policy_path(model, launch._DEFAULT_POLICIES[model]).is_file()


@pytest.mark.parametrize(
    ("model", "policy"), [("g1", "gr00t_decoupled_wbc"), ("h2", "agile_velocity")]
)
def test_under_deploy_the_model_and_policy_come_from_the_manifest(model: str, policy: str) -> None:
    """`deploy run` forwards no launch arguments; its `robot_yaml` names the robot."""
    resolve = _load_launch_module().resolve_model_and_policy
    robot_yaml = str(_REPO / "robots" / model / "robot.yaml")
    assert resolve("", "", robot_yaml) == (model, policy)
    # An explicit policy is honoured (the HAL then needs `--hal s0_policy=`).
    if model == "g1":
        assert resolve("", "agile_velocity", robot_yaml) == ("g1", "agile_velocity")
    # A model that contradicts the deployed manifest would drive the other robot's joint table.
    other = "h2" if model == "g1" else "g1"
    with pytest.raises(ROSConfigError, match="contradicts"):
        resolve(other, "", robot_yaml)


def test_a_hand_run_must_name_the_model() -> None:
    """No manifest, no guess: the G1 and H2 graphs drive different joint tables."""
    resolve = _load_launch_module().resolve_model_and_policy
    with pytest.raises(ROSConfigError, match="hand run must name it"):
        resolve("", "", None)
    with pytest.raises(ROSConfigError, match="not one of"):
        resolve("h7", "", None)
    assert resolve("h2", "", None) == ("h2", "agile_velocity")
    assert resolve("g1", "", None) == ("g1", "gr00t_decoupled_wbc")


def test_a_placeholder_or_missing_policy_is_refused(tmp_path: Path) -> None:
    doc = _yaml(_G1_CONFIG / "policies" / "agile_velocity.yaml")
    doc["openral_policy"]["status"] = "placeholder"
    placeholder = tmp_path / "placeholder.yaml"
    placeholder.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ROSConfigError, match="placeholder"):
        load_s0_policy(placeholder)
    with pytest.raises(ROSConfigError, match="does not exist"):
        load_s0_policy(tmp_path / "no_such_policy.yaml")


def test_the_leapp_config_resolution_order_and_the_filled_copy(tmp_path: Path) -> None:
    agile = load_s0_policy(s0_policy_path("g1", "agile_velocity"))
    leapp = tmp_path / "agile_velocity.yaml"
    leapp.write_text("models: {}\n", encoding="utf-8")

    def _no_index(pkg: str) -> str:
        raise AssertionError(f"an explicit override must not consult the package index ({pkg})")

    resolved = resolve_leapp_config(agile, str(leapp), _no_index)
    assert resolved == leapp.resolve()
    params = policy_parameters(agile, resolved)
    assert params["inference_controller"]["ros__parameters"]["config_path"] == str(resolved)
    # The loaded policy itself is untouched.
    assert agile.controllers["inference_controller"]["ros__parameters"]["config_path"] == ""
    with pytest.raises(ROSConfigError, match="does not exist"):
        resolve_leapp_config(agile, str(tmp_path / "missing.yaml"), _no_index)
    # GR00T has no data package: its locally built bundle_dir, with the builder named.
    gr00t = load_s0_policy(s0_policy_path("g1", "gr00t_decoupled_wbc"))
    assert gr00t.openral_policy.data_package is None
    expected = Path(str(gr00t.openral_policy.bundle_dir)).expanduser() / "gr00t_wbc_g1.yaml"
    if expected.is_file():
        assert resolve_leapp_config(gr00t, "") == expected
    else:
        with pytest.raises(ROSConfigError, match="build_gr00t_wbc_leapp"):
            resolve_leapp_config(gr00t, "")


def test_generates_a_launch_description_with_the_documented_arguments() -> None:
    pytest.importorskip("launch.actions", reason="`launch` not importable — source ROS 2")
    pytest.importorskip("launch_ros.actions", reason="`launch_ros` not importable — source ROS 2")
    launch = _load_launch_module()
    description = launch.generate_launch_description()
    args = {e.name for e in description.entities if e.__class__.__name__ == "DeclareLaunchArgument"}
    assert args == {
        "robot_model",
        "policy",
        "network_interface",
        "inference_controller_config_path",
    }


# ── upstream parity ───────────────────────────────────────────────────────────


def _upstream_xacro() -> Path | None:
    src = os.environ.get(_UPSTREAM_XACRO_ENV)
    rel = Path("unitree_g1/unitree_g1_description/urdf/g1_real_ros2_control.urdf.xacro")
    if src:
        return Path(src) / rel
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:
        return None
    try:
        share = get_package_share_directory("unitree_g1_description")
    except Exception:  # reason: PackageNotFoundError → not installed, skip below
        return None
    return Path(share) / "urdf" / "g1_real_ros2_control.urdf.xacro"


def test_joint_names_and_order_match_nvidias_real_g1_ros2_control_xacro() -> None:
    """Name parity (and broadcaster order) against the pinned upstream source itself.

    Runs where `unitree_g1_description` is installed or `$OPENRAL_ISAAC_ROS_ROBOTS_SRC`
    points at an isaac_ros_robots checkout (the deps.repos pin); skips otherwise.
    """
    xacro = _upstream_xacro()
    if xacro is None or not xacro.is_file():
        pytest.skip(
            "NVIDIA unitree_g1_description not installed and "
            f"${_UPSTREAM_XACRO_ENV} unset — vcs import packages/openral_hal_unitree/deps.repos"
        )
    text = xacro.read_text(encoding="utf-8")
    body = text.split('<ros2_control name="UnitreeG1System"', 1)[1].split("</ros2_control>", 1)[0]
    upstream = re.findall(r'<xacro:real_joint name="([^"]+)"', body)
    assert len(upstream) == 29
    assert set(upstream) == set(_manifest_joints())
    broadcaster = _params(_yaml(_CONTROLLER_MANAGER), "joint_state_broadcaster")["joints"]
    assert broadcaster == upstream


# ── H2: joint order, NVIDIA's hardware table, the AGILE export ────────────────

#: NVIDIA's H2 motor table: `h2_body_joint_index_table()` in
#: `unitree_hg/unitree_hg_ros2_control/src/joint_index_tables.cpp` @ 520aa94f. The H2 has
#: no NVIDIA description, so this table is the upstream contract the xacro must meet.
_H2_TABLE = Path("unitree_hg/unitree_hg_ros2_control/src/joint_index_tables.cpp")
#: The AGILE export's directory override (else the policy file's `bundle_dir`).
_H2_AGILE_ENV = "OPENRAL_H2_AGILE_DIR"


def _xacro_joints(text: str, system: str) -> list[str]:
    body = text.split(f'<ros2_control name="{system}"', 1)[1].split("</ros2_control>", 1)[0]
    return re.findall(r'<xacro:real_joint name="([^"]+)"', body)


def test_h2_whole_body_order_is_the_xacro_order_is_sdk_motor_order() -> None:
    """Manifest = xacro = every whole-body list: ankle roll before pitch, head last."""
    xacro = _xacro_joints(_H2_XACRO.read_text(encoding="utf-8"), "UnitreeH2System")
    manifest = _manifest_joints("h2")
    assert len(xacro) == 31
    assert xacro == manifest
    assert xacro.index("left_ankle_roll_joint") < xacro.index("left_ankle_pitch_joint")
    assert xacro[-2:] == ["head_pitch_joint", "head_yaw_joint"]
    doc = _yaml(_cm("h2"))
    for controller in ("joint_state_broadcaster", "safety_controller", "freeze_controller"):
        assert _params(doc, controller)["joints"] == xacro, controller
    text = _H2_XACRO.read_text(encoding="utf-8")
    assert "<plugin>unitree_hg_ros2_control/UnitreeHgSystemInterface</plugin>" in text
    assert '<param name="robot_variant">h2</param>' in text


def test_h2_xacro_joints_match_nvidias_h2_motor_table() -> None:
    """Name AND SDK-index parity with the pinned hardware interface's H2 table.

    Runs where `$OPENRAL_ISAAC_ROS_ROBOTS_SRC` points at an isaac_ros_robots checkout (the
    deps.repos pin); the installed package ships no sources, so it skips otherwise.
    """
    src = os.environ.get(_UPSTREAM_XACRO_ENV)
    if not src or not (Path(src) / _H2_TABLE).is_file():
        pytest.skip(
            f"${_UPSTREAM_XACRO_ENV} unset or not an isaac_ros_robots checkout — "
            "vcs import packages/openral_hal_unitree/deps.repos"
        )
    cpp = (Path(src) / _H2_TABLE).read_text(encoding="utf-8")
    body = cpp.split("h2_body_joint_index_table()", 1)[1].split("return table;", 1)[0]
    table = re.findall(r'\{"([a-z_]+)", (\d+)\}', body)
    assert [int(i) for _, i in table] == list(range(31))
    xacro = _xacro_joints(_H2_XACRO.read_text(encoding="utf-8"), "UnitreeH2System")
    assert [n for n, _ in table] == xacro


def test_h2_xacro_renders_31_controlled_joints_of_the_pinned_urdf() -> None:
    """`xacro` expands the include + macro; every controlled joint is a URDF joint."""
    exe = shutil.which("xacro")
    if exe is None:
        pytest.skip("xacro not on PATH — source ROS 2")
    from openral_hal._unitree_h2_assets import ensure_h2_urdf

    try:
        body = ensure_h2_urdf()
    except ROSConfigError as exc:
        pytest.skip(f"pinned H2.urdf unavailable: {exc}")
    out = subprocess.run(
        [exe, str(_H2_XACRO), f"body_urdf:={body}", "network_interface:=enp3s0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    import xml.etree.ElementTree as ET

    root = ET.fromstring(out.stdout)
    (control,) = root.findall("ros2_control")
    controlled = [j.get("name") for j in control.findall("joint")]
    assert controlled == _manifest_joints("h2")
    movable = {j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"}
    assert set(controlled) == movable
    params = {p.get("name"): p.text for p in control.find("hardware").findall("param")}  # type: ignore[union-attr] # reason: asserted present by the system block
    assert params["robot_variant"] == "h2" and params["network_interface"] == "enp3s0"
    assert root.find("link[@name='imu_in_pelvis']") is not None  # imu_sensor_broadcaster frame


def test_h2_agile_split_and_upper_body_gains() -> None:
    """AGILE owns legs + waist roll/pitch; OpenRAL keeps waist yaw, both arms and the head,
    at the gains the policy was trained with (the export's default stiffness/damping)."""
    doc = _yaml(_H2_CONFIG / "policies" / "agile_velocity.yaml")
    s0 = doc["openral_policy"]["s0_joints"]
    assert {"waist_roll_joint", "waist_pitch_joint"} <= set(s0)
    assert len(s0) == 14 and all(
        "hip" in j or "knee" in j or "ankle" in j or "waist" in j for j in s0
    )
    hal = UnitreeHgRealHAL(robot_model="h2")
    upper = _params(doc["controllers"], "upper_body_forward_joint_command_controller")
    assert upper["joints"] == hal.upper_body_joints()
    assert set(s0) == set(hal.s0_joints())
    assert {"waist_yaw_joint", "head_pitch_joint", "head_yaw_joint"} <= set(upper["joints"])
    base = _params(_yaml(_cm("h2")), "upper_body_forward_joint_command_controller")
    arm_kp = [90.0, 60.0, 20.0, 60.0, 4.0, 4.0, 4.0]
    arm_kd = [2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2]
    assert _resolve_gains(base, upper, "kp") == [300.0, *arm_kp, *arm_kp, 100.0, 100.0]
    assert _resolve_gains(base, upper, "kd") == [5.0, *arm_kd, *arm_kd, 5.0, 5.0]
    assert 200 / _params(doc["controllers"], "inference_controller")["decimation"] == 50.0


def _h2_agile_export() -> Path:
    meta = load_s0_policy(s0_policy_path("h2", "agile_velocity")).openral_policy
    override = os.environ.get(_H2_AGILE_ENV)
    bundle = Path(override) if override else Path(str(meta.bundle_dir)).expanduser()
    if not (bundle / meta.leapp_config).is_file():
        pytest.skip(
            f"H2 AGILE export not downloaded — hf download {meta.hf_repo} --revision "
            f"{meta.hf_revision} --local-dir {meta.bundle_dir} (or set ${_H2_AGILE_ENV})"
        )
    return bundle / meta.leapp_config


def test_h2_agile_policy_file_matches_the_pinned_export() -> None:
    """Split, rate, joints and gains against the export itself (pins verified first)."""
    path = _h2_agile_export()
    policy = load_s0_policy(s0_policy_path("h2", "agile_velocity"))
    if not os.environ.get(_H2_AGILE_ENV):
        assert resolve_leapp_config(policy, "") == path  # SHA-256 pins checked here
    export = _yaml(path)
    (model,) = export["models"].values()
    out = {o["name"]: o for o in model["outputs"]}
    action = out["joint_pos"]["element_names"][0]
    assert set(action) == set(policy.openral_policy.s0_joints)
    assert out["joint_pos_kp_gains"]["element_names"][0] == action
    inputs = {i["name"]: i for i in model["inputs"]}
    assert set(inputs["robot_joint_pos"]["element_names"][0]) == set(_manifest_joints("h2"))
    assert inputs["base_velocity"]["kind"] == "command/body/velocity"
    assert export["pipeline"]["configs"]["frequency"] == 50.0
    assert model["parameters"]["sha256sum"] == policy.openral_policy.sha256["agile_velocity.onnx"]
    art = export["agile"]["articulations"]["robot"]
    stiffness = dict(zip(art["joint_names"], art["default_joint_stiffness"], strict=True))
    damping = dict(zip(art["joint_names"], art["default_joint_damping"], strict=True))
    base = _params(_yaml(_cm("h2")), "upper_body_forward_joint_command_controller")
    upper = _params(policy.controllers, "upper_body_forward_joint_command_controller")
    assert _resolve_gains(base, upper, "kp") == pytest.approx(
        [stiffness[j] for j in upper["joints"]]
    )
    assert _resolve_gains(base, upper, "kd") == pytest.approx([damping[j] for j in upper["joints"]])


def test_a_bundle_that_fails_its_pins_is_refused(tmp_path: Path) -> None:
    """The Hub bundle is checked file by file; a missing one names the download command."""
    policy = load_s0_policy(s0_policy_path("h2", "agile_velocity"))
    meta = policy.openral_policy.model_copy(update={"bundle_dir": str(tmp_path)})
    moved = policy.model_copy(update={"openral_policy": meta})
    with pytest.raises(ROSConfigError, match="hf download nvidia/agile_wbc_unitree_h2"):
        resolve_leapp_config(moved, "")
    for name in meta.sha256:
        (tmp_path / name).write_text("not the export\n", encoding="utf-8")
    with pytest.raises(ROSConfigError, match="SHA-256 mismatch"):
        resolve_leapp_config(moved, "")
    # The explicit override stays the operator's escape hatch, unchecked.
    assert resolve_leapp_config(moved, str(tmp_path / meta.leapp_config)).is_file()
