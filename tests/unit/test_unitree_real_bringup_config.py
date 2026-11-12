# SPDX-License-Identifier: Apache-2.0
"""`openral_hal_unitree`'s controller configs and launch must agree with the G1 manifest
and with `UnitreeHgRealHAL`.

The bringup's YAML is a hand-maintained derivative of NVIDIA's `unitree_g1_bringup`
config (isaac_ros_robots @ 520aa94f), trimmed to the 29-DoF hands-less body; nothing
else compares it with `robots/g1/robot.yaml` or with the HAL's constants. A drift is
silent on hardware — a controller claiming a joint the hardware does not export never
activates, an upper-body joint the HAL thinks it owns but the policy also commands
fights the policy, an e-stop list that misses a controller leaves it running. These
tests read the real files (no ROS install needed); the launch-description test and the
upstream-xacro parity test run where those sources exist and skip with a reason
otherwise.
"""

from __future__ import annotations

import importlib.util
import os
import re
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
    resolve_leapp_config,
    s0_policy_path,
)

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "packages" / "openral_hal_unitree"
_LAUNCH = _PKG / "launch" / "real_bringup.launch.py"
_G1_CONFIG = _PKG / "config" / "g1"
_CONTROLLER_MANAGER = _G1_CONFIG / "controller_manager.yaml"
_POLICIES = sorted((_G1_CONFIG / "policies").glob("*.yaml"))

#: The G1 joints of NVIDIA's real-hardware ros2_control xacro
#: (`unitree_g1_description/urdf/g1_real_ros2_control.urdf.xacro` @ 520aa94f, the
#: `UnitreeG1System` block — hands excluded, they are a separate `with_hands` system).
_UPSTREAM_XACRO_ENV = "OPENRAL_ISAAC_ROS_ROBOTS_SRC"


def _yaml(path: Path) -> Any:  # reason: arbitrary YAML document
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _manifest_joints() -> list[str]:
    desc = RobotDescription.from_yaml(str(_REPO / "robots" / "g1" / "robot.yaml"))
    return [j.name for j in desc.joints]


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


def _ready_policies() -> list[Path]:
    return [p for p in _POLICIES if _yaml(p)["openral_policy"]["status"] == "ready"]


# ── controller_manager.yaml ───────────────────────────────────────────────────


def test_whole_body_controllers_cover_exactly_the_manifest_joints() -> None:
    """Broadcaster, safety and freeze each span the 29 manifest joints — no hands, no gaps."""
    doc = _yaml(_CONTROLLER_MANAGER)
    manifest = _manifest_joints()
    for controller in ("joint_state_broadcaster", "safety_controller", "freeze_controller"):
        joints = _params(doc, controller)["joints"]
        assert len(joints) == len(set(joints)), f"{controller} lists a joint twice"
        assert set(joints) == set(manifest), (
            f"{controller}: extra {sorted(set(joints) - set(manifest))}, "
            f"missing {sorted(set(manifest) - set(joints))}"
        )


def test_every_declared_controller_type_is_a_known_plugin() -> None:
    """Upstream plugins, except safety_controller, which runs through OpenRAL's guard."""
    types_ = {
        name: spec["type"]
        for name, spec in _params(_yaml(_CONTROLLER_MANAGER), "controller_manager").items()
        if isinstance(spec, dict)
    }
    assert types_ == {
        "joint_state_broadcaster": "joint_state_broadcaster/JointStateBroadcaster",
        "imu_sensor_broadcaster": "imu_sensor_broadcaster/IMUSensorBroadcaster",
        "freeze_controller": "isaac_ros_deploy_ros2_control/FreezeController",
        "safety_controller": "openral_hal_unitree/GuardedSafetyController",
        "inference_controller": "isaac_ros_deploy_ros2_control/InferenceController",
        "upper_body_forward_joint_command_controller": (
            "isaac_ros_deploy_ros2_control/ForwardJointCommandController"
        ),
    }


def test_the_safety_controllers_emergency_handover_matches_the_hal_estop() -> None:
    """Upstream's own out-of-domain swap and `/openral/estop` move the same controllers."""
    ood = _params(_yaml(_CONTROLLER_MANAGER), "safety_controller")["out_of_domain_detection"]
    hal = UnitreeHgRealHAL()
    assert ood["emergency_controller"] == hal.freeze_controller
    assert ood["deactivate_controllers"] == hal.controller_names()


def test_the_policy_starts_blended_out() -> None:
    """Nothing in OpenRAL raises blend_ratio; the operator does, on the gantry."""
    assert _params(_yaml(_CONTROLLER_MANAGER), "safety_controller")["blend_ratio"] == 0.0


def test_the_blend_ramp_is_within_the_guards_cap() -> None:
    """GuardedSafetyController refuses to configure outside (0, 1.0] /s (issue #548)."""
    speed = _params(_yaml(_CONTROLLER_MANAGER), "safety_controller")["max_blend_ratio_speed"]
    assert 0.0 < speed <= 1.0


def test_the_guarded_plugin_is_exported_under_the_configured_name() -> None:
    """The type the YAML names must be the class the plugin XML exports, in a library
    CMake builds; a mismatch is a controller that fails to load on the robot."""
    xml = (_PKG / "openral_hal_unitree_controllers.xml").read_text(encoding="utf-8")
    assert 'name="openral_hal_unitree/GuardedSafetyController"' in xml
    assert 'type="openral_hal_unitree::GuardedSafetyController"' in xml
    library = re.search(r'<library path="([^"]+)"', xml)
    assert library is not None
    cmake = (_PKG / "CMakeLists.txt").read_text(encoding="utf-8")
    assert f"add_library({library.group(1)} SHARED" in cmake
    assert "controller_interface openral_hal_unitree_controllers.xml" in cmake


def test_the_upper_body_controller_listens_where_the_hal_publishes() -> None:
    fwd = _params(_yaml(_CONTROLLER_MANAGER), "upper_body_forward_joint_command_controller")
    # Exactly one input mode may be set upstream; OpenRAL uses the chunked one.
    assert fwd["joint_command_trajectory_topic"] == "~/joint_commands_trajectory"
    assert "joint_command_topic" not in fwd and "joint_state_topic" not in fwd
    resolved = "/upper_body_forward_joint_command_controller/joint_commands_trajectory"
    bindings = UnitreeHgRealHAL().command_bindings()
    assert bindings[resolved] is ControllerKind.JOINT_COMMAND_TRAJECTORY
    # Holds its activation pose until commanded (no upstream `default_position` homing).
    assert "default_position" not in fwd


def test_manager_rate_is_upstreams_200_hz() -> None:
    """The README's rate table is pinned to the file, so it cannot quietly go stale."""
    assert _params(_yaml(_CONTROLLER_MANAGER), "controller_manager")["update_rate"] == 200


# ── policies/*.yaml ───────────────────────────────────────────────────────────


def test_both_policies_are_ready_and_gr00t_wbc_is_the_default() -> None:
    """GR00T Decoupled WBC won the G1 S0 bake-off; AGILE stays selectable."""
    statuses = {p.stem: _yaml(p)["openral_policy"]["status"] for p in _POLICIES}
    assert statuses == {"agile_velocity": "ready", "gr00t_decoupled_wbc": "ready"}
    desc = RobotDescription.from_yaml(str(_REPO / "robots" / "g1" / "robot.yaml"))
    assert desc.hal.parameters.defaults["s0_policy"] == "gr00t_decoupled_wbc"
    assert _load_launch_module()._DEFAULT_POLICY == "gr00t_decoupled_wbc"
    assert UnitreeHgRealHAL().s0_policy == "gr00t_decoupled_wbc"


@pytest.mark.parametrize("policy", _ready_policies(), ids=lambda p: p.stem)
def test_a_ready_policy_splits_the_body_disjointly_between_s0_and_openral(policy: Path) -> None:
    doc = _yaml(policy)
    s0 = doc["openral_policy"]["s0_joints"]
    upper = _params(doc["controllers"], "upper_body_forward_joint_command_controller")["joints"]
    manifest = set(_manifest_joints())
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
    import re

    base = _params(_yaml(_CONTROLLER_MANAGER), "upper_body_forward_joint_command_controller")
    doc = _yaml(_G1_CONFIG / "policies" / "gr00t_decoupled_wbc.yaml")
    policy = _params(doc["controllers"], "upper_body_forward_joint_command_controller")
    expected_kp = [100.0, 100.0, 40.0, 40.0, 20.0, 20.0, 20.0] * 2
    expected_kd = [5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0] * 2
    for gain, expected in (("kp", expected_kp), ("kd", expected_kd)):
        patterns = dict(sorted({**base[gain], **policy[gain]}.items()))  # std::map order
        resolved = []
        for joint in policy["joints"]:
            value = None
            for pattern, v in patterns.items():
                if re.fullmatch(pattern, joint):
                    value = v
            resolved.append(value)
        assert resolved == expected, gain


@pytest.mark.parametrize("policy", _ready_policies(), ids=lambda p: p.stem)
def test_a_ready_policys_velocity_source_is_the_hal_twist_topic(policy: Path) -> None:
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
    hal = UnitreeHgRealHAL(s0_policy=policy.stem)
    assert hal.command_bindings()[topic] is ControllerKind.TWIST
    # Writes through the safety controller like upstream's agile_velocity group.
    assert inference["command_prefix"] == "safety_controller"
    assert inference["command_suffix"] == "_raw"
    # A dead OpenRAL side must stop the walk: the input goes stale, the policy blanks.
    assert 0.0 < inference["topic_input_timeout_ms"] <= 1000.0
    # config_path is the launch's to fill; an empty one fails the controller's configure.
    assert inference["config_path"] == ""


# ── launch file (pure helpers; no ROS) ────────────────────────────────────────


def test_launch_chain_and_defaults_agree_with_the_hal_and_manifest() -> None:
    launch = _load_launch_module()
    assert list(launch._S0_CHAIN) == UnitreeHgRealHAL().controller_names()
    assert UnitreeHgRealHAL().freeze_controller == launch._HOLD_CONTROLLER
    desc = RobotDescription.from_yaml(str(_REPO / "robots" / "g1" / "robot.yaml"))
    assert desc.hal.real_bringup == f"{launch._PACKAGE}:{_LAUNCH.name}"
    assert desc.hal.parameters.defaults["robot_model"] == launch._DEFAULT_ROBOT_MODEL
    assert desc.hal.parameters.defaults["s0_policy"] == launch._DEFAULT_POLICY
    assert launch._DEFAULT_ROBOT_MODEL in launch._ROBOT_MODELS
    assert s0_policy_path(launch._DEFAULT_ROBOT_MODEL, launch._DEFAULT_POLICY).is_file()


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
