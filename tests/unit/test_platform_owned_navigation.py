"""ADR-0127 Decision 1: ``deploy run`` of a mobile base leaves navigation to its platform.

``deploy sim`` keeps OpenRAL's own SLAM + Nav2 + ``MobileBaseBridge``. ``deploy run`` of a
robot with ``base_joints`` (today only ``openamrobot_v2``) launches neither, and the HAL node
attaches no bridge: the platform owns ``/odom``, ``odom -> base`` and the final ``/cmd_vel``,
and OpenRAL sends ``navigate_to_pose`` goals to the platform's own server. There is no setting:
the deploy command decides, and a scene that asks for OpenRAL's SLAM or Nav2 on such a real
deploy is refused. Real manifests and scenes throughout; no ``ros2 launch`` is run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
from openral_cli.deploy_sim import LaunchInvocation, resolve_launch_invocation
from openral_core import RobotDescription
from openral_core.exceptions import ROSConfigError
from openral_hal.mobile_base_bridge import platform_owns_navigation

_REPO = Path(__file__).resolve().parents[2]


def _robot(name: str) -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO / "robots" / name / "robot.yaml"))


@pytest.mark.parametrize(
    ("robot", "hal_mode", "owned"),
    [
        ("openamrobot_v2", "real", True),
        ("openamrobot_v2", "sim", False),
        ("panda_mobile", "sim", False),
        ("openarm", "real", False),  # no mobile base
    ],
)
def test_platform_owns_navigation_follows_the_deploy_command(
    robot: str, hal_mode: str, owned: bool
) -> None:
    assert platform_owns_navigation(_robot(robot), hal_mode=hal_mode) is owned


def _real_scene(tmp_path: Path, runtime: str = "") -> Path:
    scene = tmp_path / "openamrobot_real.yaml"
    scene.write_text(
        "scene:\n  id: openamrobot_lab\n"
        "robot_id: openamrobot_v2\n"
        "runtime:\n  enable_octomap: false\n" + runtime,
        encoding="utf-8",
    )
    return scene


def _resolve(config: Path, hal_mode: str) -> LaunchInvocation:
    return resolve_launch_invocation(
        config=config,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode=hal_mode,
    )


def _launch_arg(inv: LaunchInvocation, name: str) -> str:
    values = [a.split(":=", 1)[1] for a in inv.argv_template if a.startswith(f"{name}:=")]
    assert len(values) == 1, (name, values)
    return values[0]


def test_deploy_run_launches_no_openral_slam_or_nav2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRAL_UNIT_SERIAL", "AMR-TEST-0001")
    inv = _resolve(_real_scene(tmp_path), "real")
    assert _launch_arg(inv, "enable_slam") == "false"
    assert _launch_arg(inv, "enable_nav2") == "false"


@pytest.mark.parametrize("key", ["enable_slam", "enable_nav2"])
def test_deploy_run_refuses_a_scene_asking_for_openral_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    monkeypatch.setenv("OPENRAL_UNIT_SERIAL", "AMR-TEST-0001")
    with pytest.raises(ROSConfigError, match="platform owns navigation"):
        _resolve(_real_scene(tmp_path, f"  {key}: true\n"), "real")


def test_deploy_sim_keeps_openral_slam_and_nav2() -> None:
    inv = _resolve(_REPO / "scenes" / "deploy" / "openamrobot_v2_walls.yaml", "sim")
    assert _launch_arg(inv, "enable_slam") == "true"
    assert _launch_arg(inv, "enable_nav2") == "true"


def _launch_module() -> Any:  # reason: a launch file loaded as a module
    """The real launch file, loaded by path (it is not an importable package)."""
    import sys

    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    path = _REPO / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_platform_nav", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_platform_nav"] = module
    spec.loader.exec_module(module)
    return module


def test_the_palette_waits_for_the_platforms_navigate_to_pose_without_naming_its_nodes() -> None:
    """The Nav2 rSkill reaches the palette once the platform's server is advertised.

    The platform's node names are not OpenRAL's to assume, so the helper waits on the action
    alone; with OpenRAL's own Nav2 it also waits for ``/bt_navigator`` to be active.
    """
    module = _launch_module()
    platform = module._palette_reseed_cmd(lifecycle_node="")
    own = module._palette_reseed_cmd(lifecycle_node="/bt_navigator")
    assert platform[platform.index("--action") + 1] == "/navigate_to_pose"
    assert "--lifecycle-node" not in platform
    assert own[own.index("--lifecycle-node") + 1] == "/bt_navigator"
