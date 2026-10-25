"""The launch's safety-kernel exemption-provider params (ADR-0121).

The kernel enforces every bound itself (``cpp/openral_safety_kernel/test/
test_exemption_provider_kernel.cpp``); this pins the launch half: no provider → no params
(the kernel exactly as today), a provider → its class plus the manifest's hand links as the
kernel's relaxable-link mask, and an off-sim request refused at launch. Real OpenArm and
DROID manifests, the real launch file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from openral_core import DeployRuntime, RobotDescription
from openral_core.exceptions import ROSConfigError

_REPO = Path(__file__).resolve().parents[2]
_LAUNCH = _REPO / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"
_OPENARM = _REPO / "robots" / "openarm" / "robot.yaml"
_DROID = _REPO / "robots" / "droid_franka_robotiq" / "robot.yaml"
_PROVIDER = "openral_pro_safety_exemptions/ContactExemptions"
_CELL = _REPO / "scenes" / "deploy" / "openarm_real_world_voxels.yaml"


@pytest.fixture(scope="module")
def launch_module() -> Any:
    """The real launch file, loaded by path — it is not an importable package."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_provider", _LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_provider"] = module
    spec.loader.exec_module(module)
    return module


def test_no_provider_adds_no_kernel_params(launch_module: Any) -> None:
    robot = RobotDescription.from_yaml(str(_OPENARM))
    derive = launch_module._exemption_provider_kernel_params
    assert derive(robot, provider="", hal_mode="sim") == {}
    assert derive(robot, provider="", hal_mode="real") == {}


@pytest.mark.parametrize("manifest", [_OPENARM, _DROID])
def test_a_provider_gets_the_manifest_hand_links_as_its_mask(
    launch_module: Any, manifest: Path
) -> None:
    robot = RobotDescription.from_yaml(str(manifest))
    params = launch_module._exemption_provider_kernel_params(
        robot, provider=_PROVIDER, hal_mode="sim"
    )
    assert params == {
        "exemption_provider": _PROVIDER,
        "exemption_provider_links": list(robot.hand_links()),
    }
    assert params["exemption_provider_links"], "a gripper robot always has hand links"
    wide = launch_module._exemption_provider_kernel_params(
        robot, provider=_PROVIDER, hal_mode="sim", wrist_links=2
    )
    assert wide["exemption_provider_links"] == list(robot.hand_links(wrist_links=2))


@pytest.mark.parametrize("mode", ["real", "", "SIM"])
def test_a_provider_off_sim_is_refused_at_launch(launch_module: Any, mode: str) -> None:
    robot = RobotDescription.from_yaml(str(_OPENARM))
    with pytest.raises(ROSConfigError, match="sim-only"):
        launch_module._exemption_provider_kernel_params(robot, provider=_PROVIDER, hal_mode=mode)


def _cell_invocation(tmp_path: Path, runtime: dict[str, object], hal_mode: str) -> Any:
    from openral_cli.deploy_sim import resolve_launch_invocation

    data = yaml.safe_load(_CELL.read_text(encoding="utf-8"))
    data["runtime"].update(runtime)
    scene = tmp_path / "cell.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")
    return resolve_launch_invocation(
        config=scene,
        robot_override="openarm",
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
        enable_dashboard=False,
        hal_mode=hal_mode,
    )


def test_deploy_sim_forwards_the_runtime_provider_and_deploy_run_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DeployRuntime.exemption_provider`` is how ``openral deploy sim`` selects one."""
    monkeypatch.setenv("OPENRAL_ROBOT_UNIT", "thor")
    assert DeployRuntime().exemption_provider == ""
    off = _cell_invocation(tmp_path, {}, "sim").argv_template
    assert not any(a.startswith("exemption_provider") for a in off), "none: argv unchanged"
    on = _cell_invocation(tmp_path, {"exemption_provider": _PROVIDER}, "sim").argv_template
    assert f"exemption_provider:={_PROVIDER}" in on
    with pytest.raises(ROSConfigError, match="sim-only"):
        _cell_invocation(tmp_path, {"exemption_provider": _PROVIDER}, "real")
