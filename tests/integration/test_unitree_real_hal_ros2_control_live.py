# SPDX-License-Identifier: Apache-2.0
"""The real G1 HAL drives NVIDIA's S0 controller chain — with only the motors left out.

Brings up the production `openral_hal_unitree` controller config
(`config/g1/controller_manager.yaml` + the default `policies/gr00t_decoupled_wbc.yaml`) in a real
`controller_manager`, with NVIDIA's real `SafetyController` (run through OpenRAL's
`GuardedSafetyController` plugin, as on the robot), `FreezeController` and
`ForwardJointCommandController` (isaac_ros_deploy) over `mock_components/GenericSystem`
exporting the G1's 29 joints with the same `position/velocity/effort/kp/kd` command
interfaces `UnitreeHgSystemInterface` exports — the hardware-boundary double CLAUDE.md
§1.11 allows. It loads everything inactive and activates the runtime group in one strict
switch, exactly like `real_bringup.launch.py`, and drives it through the production
`UnitreeHgRealHAL` + `RosControlTransport`.

One substitution, stated plainly: the S0 policy's LEAPP bundle needs the GR00T weights
(built per host by `tools/build_gr00t_wbc_leapp.py`) and NVIDIA's Triton runtime, and its
numerics are covered separately by `tests/unit/test_gr00t_wbc_leapp_bundle.py`, so the
controller named `inference_controller` is a real `ForwardJointCommandController` holding
the S0 joints at their activation pose. Everything the HAL touches — topic, message types, the
chain through `safety_controller`, the e-stop swap — is the production article.

Asserts: an upper-body chunk reaches the joints through the safety chain; a leg target is
refused before the wire; `/openral/estop`'s swap leaves `freeze_controller` active and
the S0 chain inactive, acknowledged by `controller_manager`; the blend guard (issue #548)
refuses a `max_blend_ratio_speed` change over the real parameter service, and re-arming the
chain after the e-stop with the policy still blended in starts it blended out.

Not `launch_testing`: the workspace pytest config disables its plugins globally (see
`tests/integration/test_estop_watchdog_graph_live.py`); the graph is orchestrated with the
same subprocess helpers as `test_real_hal_estop_ros2_control_live.py`. Gates: a sourced ROS
2 with ros2_control, plus `isaac_ros_deploy_ros2_control` and `isaac_ros_deploy_interfaces`
(build them from `packages/openral_hal_unitree/deps.repos`), and `openral_hal_unitree` built
against them (its plugin). Missing any → skip with reason.
"""

from __future__ import annotations

import importlib.util
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.integration._ros2_control_stack import (
    BRINGUP_TIMEOUT_S,
    driver_node,
    spawn,
    terminate,
    unavailable,
    wait_until,
)

_REPO = Path(__file__).resolve().parents[2]
_G1_YAML = _REPO / "robots" / "g1" / "robot.yaml"
_CONFIG = _REPO / "packages" / "openral_hal_unitree" / "config" / "g1"
_BUILD_HINT = "build packages/openral_hal_unitree/deps.repos"


def _why_not() -> str | None:
    reason = unavailable()
    if reason is not None:
        return reason
    if importlib.util.find_spec("isaac_ros_deploy_interfaces") is None:
        return f"isaac_ros_deploy_interfaces not importable — {_BUILD_HINT}"
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

    try:
        get_package_share_directory("isaac_ros_deploy_ros2_control")
    except (PackageNotFoundError, ValueError):
        return f"isaac_ros_deploy_ros2_control not installed — {_BUILD_HINT}"
    try:
        prefix = Path(get_package_share_directory("openral_hal_unitree")).parents[1]
    except (PackageNotFoundError, ValueError):
        return "openral_hal_unitree not installed — build it with the NVIDIA workspace sourced"
    if not (prefix / "lib" / "libopenral_hal_unitree_guarded_safety_controller.so").exists():
        return (
            "openral_hal_unitree was built without isaac_ros_deploy_ros2_control, so "
            "GuardedSafetyController is missing — rebuild it with the NVIDIA workspace sourced"
        )
    return None


pytestmark = pytest.mark.skipif(_why_not() is not None, reason=str(_why_not()))


def _urdf(path: Path, joints: list[str]) -> Path:
    """GenericSystem with the interface set UnitreeHgSystemInterface exports per joint."""
    links = "".join(f'  <link name="g1_link_{i}"/>\n' for i in range(len(joints) + 1))
    joint_xml = "".join(
        f'  <joint name="{j}" type="revolute"><parent link="g1_link_{i}"/>'
        f'<child link="g1_link_{i + 1}"/><axis xyz="0 0 1"/>'
        f'<limit lower="-3.14" upper="3.14" effort="100" velocity="30"/></joint>\n'
        for i, j in enumerate(joints)
    )
    control = "".join(
        f'    <joint name="{j}">'
        + "".join(f'<command_interface name="{c}"/>' for c in ("position", "velocity", "effort"))
        + '<command_interface name="kp"/><command_interface name="kd"/>'
        + "".join(f'<state_interface name="{s}"/>' for s in ("position", "velocity", "effort"))
        + "</joint>\n"
        for j in joints
    )
    path.write_text(
        f'<?xml version="1.0"?>\n<robot name="g1">\n{links}{joint_xml}'
        '  <ros2_control name="UnitreeG1System" type="system">\n'
        "    <hardware><plugin>mock_components/GenericSystem</plugin></hardware>\n"
        f"{control}  </ros2_control>\n</robot>\n"
    )
    return path


def _overrides(path: Path, s0_joints: list[str]) -> Path:
    """The default policy file's controllers, with the ONNX controller stood in (docstring)."""
    policy = yaml.safe_load((_CONFIG / "policies" / "gr00t_decoupled_wbc.yaml").read_text())
    params: dict[str, Any] = policy["controllers"]
    params["controller_manager"] = {
        "ros__parameters": {
            "inference_controller": {
                "type": "isaac_ros_deploy_ros2_control/ForwardJointCommandController"
            }
        }
    }
    params["inference_controller"] = {
        "ros__parameters": {
            "joints": s0_joints,
            "joint_command_topic": "~/hold_only",  # never published: holds activation pose
            "command_prefix": "safety_controller",
            "command_suffix": "_raw",
            "kp": {".*": 100.0},
            "kd": {".*": 2.0},
        }
    }
    path.write_text(yaml.safe_dump(params))
    return path


def _run(argv: list[str], timeout_s: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)


def _param(node: str, name: str) -> float:
    """A double parameter via `ros2 param get` ("Double value is: <v>")."""
    got = _run(["ros2", "param", "get", node, name])
    prefix = "Double value is:"
    assert got.stdout.startswith(prefix), got.stdout + got.stderr
    return float(got.stdout.removeprefix(prefix))


@pytest.fixture
def s0_graph(tmp_path: Path) -> Iterator[None]:
    from openral_core import RobotDescription
    from openral_hal.unitree_real import UnitreeHgRealHAL

    desc = RobotDescription.from_yaml(str(_G1_YAML))
    joints = [j.name for j in desc.joints]
    urdf = _urdf(tmp_path / "g1.urdf", joints)
    overrides = _overrides(tmp_path / "policy_overrides.yaml", UnitreeHgRealHAL().s0_joints())
    rsp = tmp_path / "rsp.yaml"
    rsp.write_text(
        yaml.safe_dump(
            {"robot_state_publisher": {"ros__parameters": {"robot_description": urdf.read_text()}}}
        )
    )
    import os

    env = dict(os.environ)
    params = [str(_CONFIG / "controller_manager.yaml"), str(overrides)]
    procs = [
        spawn(
            [
                "ros2",
                "run",
                "robot_state_publisher",
                "robot_state_publisher",
                "--ros-args",
                "--params-file",
                str(rsp),
            ],
            tmp_path / "rsp.log",
            env,
        ),
        spawn(
            [
                "ros2",
                "run",
                "controller_manager",
                "ros2_control_node",
                "--ros-args",
                *(a for p in params for a in ("--params-file", p)),
            ],
            tmp_path / "controller_manager.log",
            env,
        ),
    ]
    try:
        load = _run(
            [
                "ros2",
                "run",
                "controller_manager",
                "spawner",
                "joint_state_broadcaster",
                "freeze_controller",
                "safety_controller",
                "upper_body_forward_joint_command_controller",
                "inference_controller",
                *(a for p in params for a in ("--param-file", p)),
                "--inactive",
                "--controller-manager-timeout",
                str(int(BRINGUP_TIMEOUT_S)),
            ],
            timeout_s=BRINGUP_TIMEOUT_S + 30,
        )
        assert load.returncode == 0, load.stdout + load.stderr
        switch = _run(
            [
                "ros2",
                "control",
                "switch_controllers",
                "--activate",
                "joint_state_broadcaster",
                "safety_controller",
                "inference_controller",
                "upper_body_forward_joint_command_controller",
                "--strict",
            ]
        )
        assert switch.returncode == 0, switch.stdout + switch.stderr
        # The policy starts blended out; the operator's step, done here by the test.
        blend = _run(["ros2", "param", "set", "/safety_controller", "blend_ratio", "1.0"])
        assert blend.returncode == 0, blend.stdout + blend.stderr
        yield
    finally:
        for p in reversed(procs):
            terminate(p)


def test_the_hal_drives_the_upper_body_through_the_safety_chain_and_estop_freezes(
    s0_graph: None, tmp_path: Path
) -> None:
    from openral_core import RobotDescription
    from openral_core.exceptions import ROSCapabilityMismatch, ROSEStopRequested
    from openral_core.schemas import Action, ControlMode
    from openral_hal import build_hal
    from openral_hal.ros_control_transport import RosControlTransport
    from openral_hal.unitree_real import UnitreeHgRealHAL

    desc = RobotDescription.from_yaml(str(_G1_YAML))
    hal = build_hal(desc, mode="real")
    assert isinstance(hal, UnitreeHgRealHAL)
    names = [j.name for j in desc.joints]
    with driver_node("unitree_hal_live") as (node, spin):
        bindings = hal.command_bindings()
        transport = RosControlTransport(
            node,
            command_topics=list(bindings),
            joint_names=hal.ros2_control_joint_names(),
            joint_state_topic=hal.joint_state_topic,
            command_kinds=bindings,
            controller_names=hal.controller_names(),
        )
        hal.attach_transport(transport.publish, transport.state, transport.last_arrival)
        hal.attach_controller_stop(transport)
        assert wait_until(spin, lambda: not transport.missing_joints(), 20.0), (
            f"/joint_states never carried {transport.missing_joints()}"
        )
        hal.connect()

        # Issue #548: the blend speed is fixed at configure, over the real parameter service.
        speed = _run(
            ["ros2", "param", "set", "/safety_controller", "max_blend_ratio_speed", "10.0"]
        )
        # `ros2 param set` exits 0 on a refusal and reports it on stderr.
        assert "Setting parameter failed" in speed.stderr, speed.stdout + speed.stderr
        assert _param("/safety_controller", "max_blend_ratio_speed") == 1.0

        left = [n for n in hal.upper_body_joints() if n.startswith("left_")]
        row = [0.3 if n == "left_elbow_joint" else 0.0 for n in names]

        def _arm_tick() -> None:
            hal.send_action(
                Action(
                    control_mode=ControlMode.JOINT_POSITION,
                    joint_targets=[row],
                    joint_names=left,
                )
            )

        def _elbow() -> float:
            return float(transport.state()["position"][names.index("left_elbow_joint")])  # type: ignore[index]

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and abs(_elbow() - 0.3) > 0.02:
            _arm_tick()
            spin(0.05)
        assert abs(_elbow() - 0.3) <= 0.02, f"left elbow at {_elbow()}, commanded 0.3"

        with pytest.raises(ROSCapabilityMismatch):
            hal.send_action(Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[row]))

        with pytest.raises(ROSEStopRequested, match="acknowledged"):
            hal.estop()
        report = hal.last_stop_report
        assert report is not None and report.stopped, report
        states = transport.controller_states(timeout_s=5.0)
        assert states.get("freeze_controller") == "active", states
        for controller in hal.controller_names():
            assert states.get(controller) == "inactive", states

        # Issue #548: blend_ratio is still 1.0 and upstream's activation would apply it with no
        # ramp. The guard resets it to 0.0 instead of refusing — a refused activation would
        # keep this swap's deactivation of freeze_controller and leave the joints unclaimed.
        rearm = _run(
            [
                "ros2",
                "control",
                "switch_controllers",
                "--deactivate",
                "freeze_controller",
                "--activate",
                *hal.controller_names(),
                "--strict",
            ]
        )
        assert rearm.returncode == 0, rearm.stdout + rearm.stderr
        states = transport.controller_states(timeout_s=5.0)
        for controller in hal.controller_names():
            assert states.get(controller) == "active", states
        assert _param("/safety_controller", "blend_ratio") == 0.0
        cm_log = (tmp_path / "controller_manager.log").read_text(errors="replace")
        assert "blend_ratio was 1 at activation; reset to 0.0" in cm_log, cm_log[-2000:]
        transport.close()
